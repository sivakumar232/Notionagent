import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

logger = logging.getLogger("notionagent.mcp")

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}",
    re.IGNORECASE,
)

# Canonical aliases mapping planner actions to exact Notion MCP tool names
TOOL_ALIASES: Dict[str, str] = {
    "search": "API-post-search",
    "search_notion": "API-post-search",
    "find_page": "API-post-search",
    "read_page": "API-retrieve-page-markdown",
    "retrieve_page": "API-retrieve-page-markdown",
    "get_page": "API-retrieve-page-markdown",
    "create_page": "API-post-page",
    "post_page": "API-post-page",
    "new_page": "API-post-page",
    "append_block": "API-patch-block-children",
    "add_block": "API-patch-block-children",
    "update_page": "API-patch-page",
    "query_database": "API-query-data-source",
    "retrieve_database": "API-retrieve-a-database",
}


def get_notion_server_params() -> StdioServerParameters:
    """Configures the official Notion MCP server subprocess."""
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
    if not token:
        raise ValueError("NOTION_API_KEY or NOTION_TOKEN must be set in environment.")

    return StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={**os.environ, "NOTION_TOKEN": token},
    )


class NotionMCPClient:
    """Clean, production-grade MCP client for the official Notion MCP server.

    Discovers tools dynamically from the server and adapts planner arguments
    to match Notion's OpenAPI specification.
    """

    def __init__(self):
        self._server_params = get_notion_server_params()
        self._cached_tools: Optional[List[Dict[str, Any]]] = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession, None]:
        """Manages server subprocess lifecycle and MCP connection handshake."""
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Fetches and caches all MCP tools formatted for OpenAI / Azure OpenAI tool binding."""
        if self._cached_tools is not None:
            return self._cached_tools

        async with self.session() as sess:
            response = await sess.list_tools()
            self._cached_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {"type": "object", "properties": {}},
                    },
                }
                for tool in response.tools
            ]
            logger.info(f"Dynamically discovered {len(self._cached_tools)} Notion MCP tools.")
            return self._cached_tools

    def resolve_tool_name(self, name: str, available_names: List[str]) -> str:
        """Resolves tool name against aliases or available MCP tool names."""
        if name in available_names:
            return name

        # Check explicit alias
        clean_name = name.lower().replace("-", "_").strip()
        if clean_name in TOOL_ALIASES and TOOL_ALIASES[clean_name] in available_names:
            return TOOL_ALIASES[clean_name]

        # Partial matching fallback
        for candidate in available_names:
            candidate_clean = candidate.lower().replace("-", "_")
            if clean_name in candidate_clean or candidate_clean.endswith(clean_name):
                return candidate

        return name

    def adapt_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Adapts planner arguments to match Notion's OpenAPI schemas."""
        args = dict(arguments or {})

        # API-post-search expects {"query": "..."}
        if tool_name == "API-post-search":
            query = args.get("query") or args.get("title") or ""
            return {"query": str(query)}

        # API-retrieve-page-markdown expects {"page_id": "..."}
        if tool_name in ("API-retrieve-page-markdown", "API-retrieve-a-page"):
            page_id = args.get("page_id") or args.get("id") or args.get("block_id") or ""
            return {"page_id": str(page_id).replace("-", "")}

        # API-post-page expects {"parent": {...}, "properties": {...}, "children": [...]}
        if tool_name == "API-post-page":
            adapted: Dict[str, Any] = {}

            # Parent resolution
            parent = args.get("parent")
            raw_parent = args.get("parent_id") or args.get("page_id")
            default_parent = os.getenv("NOTION_PARENT_PAGE_ID")

            if isinstance(parent, dict):
                adapted["parent"] = parent
            else:
                has_uuid = bool(raw_parent and UUID_PATTERN.search(str(raw_parent)))
                target_parent = raw_parent if has_uuid else default_parent

                if target_parent and UUID_PATTERN.search(str(target_parent)):
                    adapted["parent"] = {"page_id": str(target_parent).replace("-", "")}
                else:
                    raise ValueError(
                        "Notion internal integrations require a parent page to create new pages. "
                        "Please open a page in your Notion workspace, click '...' -> 'Connect to' -> 'NotionAgent', "
                        "and set NOTION_PARENT_PAGE_ID in your .env file or mention the parent page name."
                    )

            # Properties resolution (Title)
            properties = args.get("properties")
            if isinstance(properties, dict):
                adapted["properties"] = properties
            else:
                title = str(args.get("title") or args.get("name") or "New Page")
                adapted["properties"] = {
                    "title": [{"text": {"content": title}}]
                }

            # Children blocks (Content)
            children = args.get("children")
            content = args.get("content")
            if isinstance(children, list):
                adapted["children"] = children
            elif content:
                adapted["children"] = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": str(content)}}]},
                    }
                ]

            return adapted

        # API-patch-block-children expects {"block_id": "...", "children": [...]}
        if tool_name == "API-patch-block-children":
            block_id = args.get("block_id") or args.get("page_id") or args.get("id") or ""
            content = args.get("content")
            children = args.get("children")
            if not isinstance(children, list) and content:
                children = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": str(content)}}]},
                    }
                ]
            return {
                "block_id": str(block_id).replace("-", ""),
                "children": children or [],
            }

        return args

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes a tool call on the Notion MCP server and returns the text response."""
        args = arguments or {}
        tools = await self.get_tools()
        tool_names = [t["function"]["name"] for t in tools]

        actual_tool = self.resolve_tool_name(name, tool_names)
        adapted_args = self.adapt_tool_arguments(actual_tool, args)

        logger.info(f"Executing Notion MCP Tool '{actual_tool}' with args: {adapted_args}")
        async with self.session() as sess:
            result = await sess.call_tool(actual_tool, adapted_args)

            text_outputs = []
            if hasattr(result, "content") and result.content:
                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        text_outputs.append(item.text)

            return "\n".join(text_outputs) if text_outputs else str(result)


# Shared client instance
notion_mcp_client = NotionMCPClient()

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

# Regex to detect 32-character or hyphenated UUIDs
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}",
    re.IGNORECASE,
)

# Friendly aliases mapping planner action names to official Notion MCP tool names
TOOL_ALIASES: Dict[str, str] = {
    # Search
    "search": "API-post-search",
    "search_notion": "API-post-search",
    "find_page": "API-post-search",
    # Page retrieval
    "read_page": "API-retrieve-page-markdown",
    "retrieve_page": "API-retrieve-page-markdown",
    "get_page": "API-retrieve-page-markdown",
    "get_page_raw": "API-retrieve-a-page",
    # Page creation & update
    "create_page": "API-post-page",
    "post_page": "API-post-page",
    "new_page": "API-post-page",
    "update_page": "API-patch-page",
    # Block operations
    "append_block": "API-patch-block-children",
    "add_block": "API-patch-block-children",
    "get_blocks": "API-get-block-children",
    # Database operations
    "query_database": "API-query-data-source",
    "retrieve_database": "API-retrieve-a-database",
    "create_database": "API-create-a-data-source",
    # Comments & Users
    "create_comment": "API-create-a-comment",
    "get_users": "API-get-users",
    "get_me": "API-get-self",
}


def get_notion_server_params() -> StdioServerParameters:
    """Configures the official Notion MCP server subprocess."""
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
    if not token:
        raise ValueError(
            "NOTION_TOKEN or NOTION_API_KEY must be set in environment or .env file."
        )

    return StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={
            **os.environ,
            "NOTION_TOKEN": token,
        },
    )


# Alias for backward compatibility
_build_server_params = get_notion_server_params


class NotionMCPClient:
    """Production-grade MCP client for the official Notion MCP server.

    Connects to the official `@notionhq/notion-mcp-server` over stdio (JSON-RPC 2.0).
    Features:
    - Dynamic tool discovery via MCP tools/list
    - OpenAI/Azure tool-binding compatibility format
    - Intelligent parameter adaptation for both friendly aliases and official tools
    - Resilient error handling and response extraction
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

    async def get_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetches and caches all MCP tools formatted for OpenAI / Azure OpenAI tool binding."""
        if self._cached_tools is not None and not force_refresh:
            return self._cached_tools

        try:
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
                logger.info(f"Discovered {len(self._cached_tools)} Notion MCP tools from server.")
                return self._cached_tools
        except Exception as exc:
            logger.error(f"Failed to fetch tools from Notion MCP server: {exc}", exc_info=True)
            # If server discovery fails temporarily, return cached or empty list
            return self._cached_tools or []

    async def get_tools_documentation(self) -> str:
        """Returns concise, formatted documentation of all discovered Notion MCP tools."""
        tools = await self.get_tools()
        if not tools:
            # Fallback if MCP connection is unavailable
            return "\n".join(f"- `{alias}`: Calls {mcp_name}" for alias, mcp_name in TOOL_ALIASES.items())

        alias_lookup: Dict[str, List[str]] = {}
        for alias, mcp_name in TOOL_ALIASES.items():
            alias_lookup.setdefault(mcp_name, []).append(alias)

        doc_lines: List[str] = []
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            raw_desc = fn.get("description", "").strip()
            clean_desc = raw_desc.split("Error Responses:")[0].strip()

            aliases = alias_lookup.get(name, [])
            alias_text = f" (Aliases: {', '.join(f'`{a}`' for a in aliases)})" if aliases else ""

            params = fn.get("parameters", {})
            properties = params.get("properties", {})
            required = params.get("required", [])

            prop_summaries = []
            for prop_name, prop_info in properties.items():
                p_type = prop_info.get("type", "any")
                is_req = "required" if prop_name in required else "optional"
                prop_summaries.append(f"{prop_name}: {p_type} ({is_req})")

            param_str = ", ".join(prop_summaries[:8]) if prop_summaries else "none"
            doc_lines.append(f"- `{name}`{alias_text}: {clean_desc}\n  Inputs: {param_str}")

        return "\n".join(doc_lines)

    def resolve_tool_name(self, name: str, available_names: Optional[List[str]] = None) -> str:
        """Resolves planner action names or aliases to exact Notion MCP tool names."""
        if not available_names:
            return TOOL_ALIASES.get(name.lower().strip(), name)

        # 1. Exact match
        if name in available_names:
            return name

        # 2. Known alias match
        clean_name = name.lower().replace("-", "_").strip()
        if clean_name in TOOL_ALIASES and TOOL_ALIASES[clean_name] in available_names:
            return TOOL_ALIASES[clean_name]

        # 3. Normalized matching (e.g. api_post_page -> API-post-page)
        normalized = name.lower().replace("_", "-").strip()
        for candidate in available_names:
            if candidate.lower() == normalized:
                return candidate

        # 4. Fuzzy / substring matching
        for candidate in available_names:
            cand_clean = candidate.lower().replace("-", "_")
            if clean_name in cand_clean or cand_clean.endswith(clean_name):
                return candidate

        return name

    def _normalize_id(self, raw_id: Optional[str]) -> str:
        """Strips hyphens from UUID if present, returning standard 32-char hex string."""
        if not raw_id:
            return ""
        val = str(raw_id).strip()
        match = UUID_PATTERN.search(val)
        if match:
            return match.group(0).replace("-", "")
        return val.replace("-", "")

    def adapt_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Translates and adapts high-level arguments to Notion MCP OpenAPI specifications."""
        args = dict(arguments or {})

        # 1. Search tool
        if tool_name == "API-post-search":
            query = args.get("query") or args.get("title") or args.get("search") or ""
            payload: Dict[str, Any] = {"query": str(query)}
            if "filter" in args and isinstance(args["filter"], dict):
                payload["filter"] = args["filter"]
            return payload

        # 2. Page retrieval (Markdown or raw)
        if tool_name in ("API-retrieve-page-markdown", "API-retrieve-a-page"):
            raw_id = args.get("page_id") or args.get("id") or args.get("block_id") or ""
            return {"page_id": self._normalize_id(raw_id)}

        # 3. Create Page
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
                    adapted["parent"] = {"page_id": self._normalize_id(str(target_parent))}
                else:
                    raise ValueError(
                        "Notion requires a parent page or database to create a new page. "
                        "Please specify a parent page ID or set NOTION_PARENT_PAGE_ID in your .env."
                    )

            # Properties / Title resolution
            properties = args.get("properties")
            if isinstance(properties, dict):
                adapted["properties"] = properties
            else:
                title = str(args.get("title") or args.get("name") or "Untitled")
                adapted["properties"] = {
                    "title": [{"text": {"content": title}}]
                }

            # Children blocks / Content resolution
            children = args.get("children")
            content = args.get("content")
            if isinstance(children, list):
                adapted["children"] = children
            elif content:
                adapted["children"] = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": str(content)}}]
                        },
                    }
                ]

            return adapted

        # 4. Append Block Children
        if tool_name == "API-patch-block-children":
            raw_id = args.get("block_id") or args.get("page_id") or args.get("id") or ""
            content = args.get("content")
            children = args.get("children")
            block_type = str(args.get("block_type") or "paragraph").lower()

            if not isinstance(children, list) and content:
                # Support standard block types
                block_schema = {"rich_text": [{"text": {"content": str(content)}}]}
                if block_type in ("to_do", "todo"):
                    block_obj = {"object": "block", "type": "to_do", "to_do": block_schema}
                elif block_type in ("heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"):
                    block_obj = {"object": "block", "type": block_type, block_type: block_schema}
                else:
                    block_obj = {"object": "block", "type": "paragraph", "paragraph": block_schema}
                children = [block_obj]

            return {
                "block_id": self._normalize_id(raw_id),
                "children": children or [],
            }

        # 5. Database Query
        if tool_name == "API-query-data-source":
            raw_id = args.get("data_source_id") or args.get("database_id") or args.get("id") or ""
            adapted_query: Dict[str, Any] = {"data_source_id": self._normalize_id(raw_id)}
            if "filter" in args and isinstance(args["filter"], dict):
                adapted_query["filter"] = args["filter"]
            if "sorts" in args and isinstance(args["sorts"], list):
                adapted_query["sorts"] = args["sorts"]
            return adapted_query

        # 6. Generic ID normalizer for any other Notion tool
        adapted_generic = dict(args)
        for id_field in ("page_id", "block_id", "database_id", "data_source_id", "user_id"):
            if id_field in adapted_generic and isinstance(adapted_generic[id_field], str):
                adapted_generic[id_field] = self._normalize_id(adapted_generic[id_field])

        return adapted_generic

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes a tool call on the Notion MCP server and returns the text response."""
        args = arguments or {}
        tools = await self.get_tools()
        tool_names = [t["function"]["name"] for t in tools] if tools else []

        actual_tool = self.resolve_tool_name(name, tool_names)
        adapted_args = self.adapt_tool_arguments(actual_tool, args)

        logger.info(f"Executing Notion MCP Tool '{actual_tool}' with args: {adapted_args}")

        try:
            async with self.session() as sess:
                result = await sess.call_tool(actual_tool, adapted_args)

                text_outputs: List[str] = []
                if hasattr(result, "content") and result.content:
                    for item in result.content:
                        if hasattr(item, "text") and item.text:
                            text_outputs.append(item.text)
                        elif hasattr(item, "data"):
                            text_outputs.append(str(item.data))

                # Handle error indicators returned by MCP server
                is_error = getattr(result, "isError", False)
                joined_text = "\n".join(text_outputs) if text_outputs else str(result)

                if is_error:
                    logger.error(f"Notion MCP server returned error for '{actual_tool}': {joined_text}")
                    return f"Error from Notion MCP ({actual_tool}): {joined_text}"

                return joined_text

        except Exception as exc:
            logger.error(f"Failed to execute Notion MCP tool '{actual_tool}': {exc}", exc_info=True)
            return f"Execution error in {actual_tool}: {str(exc)}"


# Shared client instance
notion_mcp_client = NotionMCPClient()

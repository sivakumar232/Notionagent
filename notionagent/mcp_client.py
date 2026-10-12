import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("notionagent.mcp")

# Mapping of common planner tool names to official Notion MCP tool names
TOOL_NAME_MAP = {
    "search": "API-post-search",
    "search_notion": "API-post-search",
    "read_page": "API-retrieve-page-markdown",
    "retrieve_page": "API-retrieve-a-page",
    "create_page": "API-post-page",
    "update_page": "API-patch-page",
    "append_block": "API-patch-block-children",
    "get_block_children": "API-get-block-children",
    "query_database": "API-query-data-source",
    "retrieve_database": "API-retrieve-a-database",
}


def get_notion_server_params() -> StdioServerParameters:
    """Configures the official Notion MCP server using stdio transport."""
    notion_token = (
        os.getenv("NOTION_TOKEN")
        or os.getenv("NOTION_API_KEY")
        or ""
    )
    if not notion_token:
        raise ValueError("Missing NOTION_API_KEY or NOTION_TOKEN in environment or .env file.")

    return StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={
            **os.environ,
            "NOTION_TOKEN": notion_token,
        },
    )


def resolve_tool_name(requested_name: str) -> str:
    """Resolves friendly tool names to official Notion MCP tool names."""
    if requested_name in TOOL_NAME_MAP:
        return TOOL_NAME_MAP[requested_name]
    return requested_name


def normalize_arguments(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensures tool arguments match Notion MCP schemas."""
    args = dict(arguments or {})

    # API-post-search expects {"query": "..."}
    if tool_name == "API-post-search":
        query = args.get("query") or args.get("title") or ""
        return {"query": str(query)}

    # API-retrieve-page-markdown expects {"page_id": "..."}
    if tool_name in ("API-retrieve-page-markdown", "API-retrieve-a-page"):
        page_id = args.get("page_id") or args.get("id") or ""
        return {"page_id": str(page_id).replace("-", "")}

    return args


class NotionMCPClient:
    """Production-grade client for the official Notion MCP server."""

    def __init__(self):
        self.server_params = get_notion_server_params()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession, None]:
        """Async context manager that manages process lifecycle and connection."""
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Discovers and returns all tools provided by the Notion MCP server."""
        async with self.session() as sess:
            response = await sess.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes a tool on the Notion MCP Server and returns parsed text."""
        actual_tool = resolve_tool_name(tool_name)
        cleaned_args = normalize_arguments(actual_tool, arguments)

        logger.info(f"Executing Notion MCP tool '{actual_tool}' with args: {cleaned_args}")
        async with self.session() as sess:
            result = await sess.call_tool(actual_tool, cleaned_args)

            text_outputs = []
            if hasattr(result, "content") and result.content:
                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        text_outputs.append(item.text)

            return "\n".join(text_outputs) if text_outputs else str(result)


# Singleton instance for simple node invocation
notion_mcp_client = NotionMCPClient()

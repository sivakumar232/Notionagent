import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool

logger = logging.getLogger("notionagent.mcp")

# Mapping of simplified/friendly planner tool names to official Notion MCP tool names
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
    """Configures the official Notion MCP server via stdio transport."""
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


class NotionMCPClient:
    """Manages connection and tool calling with the official Notion MCP server."""

    def __init__(self):
        self._exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self._available_tools: Dict[str, Any] = {}

    async def connect(self) -> None:
        """Starts the Notion MCP server subprocess and performs handshake."""
        if self.session is not None:
            return  # Already connected

        self._exit_stack = AsyncExitStack()
        try:
            params = get_notion_server_params()
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()

            # Cache tool definitions
            tools_response = await self.session.list_tools()
            self._available_tools = {t.name: t for t in tools_response.tools}
            logger.info(
                f"Connected to Notion MCP server. Discovered {len(self._available_tools)} tools."
            )
        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"Failed to connect to Notion MCP server: {e}") from e

    def resolve_tool_name(self, requested_name: str) -> str:
        """Resolves friendly names (e.g. 'search') or returns exact matching MCP tool name."""
        if requested_name in self._available_tools:
            return requested_name

        mapped_name = TOOL_NAME_MAP.get(requested_name)
        if mapped_name and mapped_name in self._available_tools:
            return mapped_name

        # Fallback: check case-insensitive match
        for tool_name in self._available_tools:
            if requested_name.lower() in tool_name.lower():
                return tool_name

        return requested_name

    def _normalize_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes common arguments to match Notion MCP OpenAPI schema."""
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

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Executes a tool on the Notion MCP Server and extracts the text content."""
        if self.session is None:
            await self.connect()

        actual_tool_name = self.resolve_tool_name(tool_name)
        cleaned_args = self._normalize_tool_arguments(actual_tool_name, arguments or {})

        logger.info(f"Executing Notion MCP Tool: '{actual_tool_name}' with {cleaned_args}")
        result = await self.session.call_tool(actual_tool_name, cleaned_args)

        # Extract text responses
        text_outputs = []
        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text") and item.text:
                    text_outputs.append(item.text)

        return "\n".join(text_outputs) if text_outputs else str(result)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available tools."""
        if self.session is None:
            await self.connect()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in self._available_tools.values()
        ]

    async def to_langchain_tools(self) -> List[StructuredTool]:
        """Converts all Notion MCP tools into LangChain StructuredTool instances."""
        if self.session is None:
            await self.connect()

        lc_tools = []
        for name, tool_meta in self._available_tools.items():
            def make_coroutine(t_name: str):
                async def _coroutine(**kwargs):
                    return await self.call_tool(t_name, kwargs)
                return _coroutine

            lc_tools.append(
                StructuredTool.from_function(
                    name=name,
                    description=tool_meta.description or f"Notion tool {name}",
                    coroutine=make_coroutine(name),
                )
            )
        return lc_tools

    async def disconnect(self) -> None:
        """Terminates session and stops the Notion MCP subprocess."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None
            self._available_tools.clear()
            logger.info("Notion MCP client disconnected cleanly.")

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("notionagent.mcp")


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

    Discovers tools dynamically from the server so the LLM gets exact schemas
    without needing brittle hardcoded dictionaries or manual argument parsers.
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

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes a tool call on the Notion MCP server and returns the text response."""
        args = arguments or {}

        # Resolve name: if exact name doesn't match, fallback to partial match (e.g. 'search' -> 'API-post-search')
        target_name = name
        tools = await self.get_tools()
        tool_names = [t["function"]["name"] for t in tools]

        if target_name not in tool_names:
            for candidate in tool_names:
                if target_name.lower().replace("_", "") in candidate.lower().replace("-", ""):
                    target_name = candidate
                    break

        logger.info(f"Executing Notion MCP Tool '{target_name}' with args: {args}")
        async with self.session() as sess:
            result = await sess.call_tool(target_name, args)

            text_outputs = []
            if hasattr(result, "content") and result.content:
                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        text_outputs.append(item.text)

            return "\n".join(text_outputs) if text_outputs else str(result)


# Shared client instance
notion_mcp_client = NotionMCPClient()

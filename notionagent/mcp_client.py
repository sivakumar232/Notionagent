"""Production-grade Dynamic Notion MCP Tool Registry

Architecture:
- Persistent stdio connection for FastAPI via lifespan context manager
- Dynamic discovery of all 24 tools via JSON-RPC 2.0 tools/list
- Direct conversion of MCP tool schemas to OpenAI function calling format
- Automatic argument adaptation and UUID normalization
- Graceful shutdown without AnyIO task group leaks
- Standalone auto-connection support for CLI & scripts
"""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

load_dotenv()
logger = logging.getLogger("notionagent.registry")

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}",
    re.IGNORECASE,
)

# Friendly convenience aliases for common planner action names
TOOL_ALIASES: Dict[str, str] = {
    "search": "API-post-search",
    "search_notion": "API-post-search",
    "find_page": "API-post-search",
    "read_page": "API-retrieve-page-markdown",
    "retrieve_page": "API-retrieve-page-markdown",
    "get_page": "API-retrieve-page-markdown",
    "get_page_raw": "API-retrieve-a-page",
    "create_page": "API-post-page",
    "post_page": "API-post-page",
    "new_page": "API-post-page",
    "update_page": "API-patch-page",
    "append_block": "API-patch-block-children",
    "add_block": "API-patch-block-children",
    "get_blocks": "API-get-block-children",
    "query_database": "API-query-data-source",
    "retrieve_database": "API-retrieve-a-database",
    "create_database": "API-create-a-data-source",
    "create_comment": "API-create-a-comment",
    "get_users": "API-get-users",
    "get_me": "API-get-self",
}


def _build_server_params() -> StdioServerParameters:
    """Configures the official Notion MCP server subprocess over stdio."""
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
    if not token:
        raise ValueError(
            "NOTION_TOKEN or NOTION_API_KEY must be set in your environment or .env file."
        )

    return StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={
            **os.environ,
            "NOTION_TOKEN": token,
        },
    )


def _mcp_tool_to_openai(tool: Tool) -> Dict[str, Any]:
    """Converts raw MCP Tool schema into standard OpenAI / Azure function definition."""
    schema = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }



class NotionToolRegistry:
    """Production-grade Notion MCP Client & Registry:
    
    1. Manages persistent session during FastAPI lifespan
    2. Dynamically discovers all tools and schemas via JSON-RPC tools/list
    3. Exposes OpenAI function formats for direct LLM binding (.bind_tools())
    4. Automatically adapts arguments and executes tool calls with low latency
    5. Fallback scoped connection for CLI testing
    """

    def __init__(self):
        self._active_session: Optional[ClientSession] = None
        self._raw_tools: List[Tool] = []
        self._openai_tools: List[Dict[str, Any]] = []
        self._tool_map: Dict[str, Tool] = {}

    @asynccontextmanager
    async def lifespan_session(self) -> AsyncGenerator["NotionToolRegistry", None]:
        """Manages a single persistent subprocess connection throughout FastAPI's lifecycle."""
        params = _build_server_params()
        logger.info("Initializing persistent Notion MCP server subprocess...")
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._active_session = session

                # Discover all tools dynamically
                response = await session.list_tools()
                self._raw_tools = response.tools
                self._tool_map = {t.name: t for t in self._raw_tools}
                self._openai_tools = [_mcp_tool_to_openai(t) for t in self._raw_tools]

                logger.info(
                    f"Discovered {len(self._raw_tools)} Notion MCP tools: "
                    f"{[t.name for t in self._raw_tools]}"
                )

                try:
                    yield self
                finally:
                    self._active_session = None
                    logger.info("Notion MCP server subprocess terminated.")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[ClientSession, None]:
        """Yields the active persistent session or temporarily creates one for CLI/tests."""
        if self._active_session is not None:
            yield self._active_session
            return

        params = _build_server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if not self._raw_tools:
                    response = await session.list_tools()
                    self._raw_tools = response.tools
                    self._tool_map = {t.name: t for t in self._raw_tools}
                    self._openai_tools = [_mcp_tool_to_openai(t) for t in self._raw_tools]
                yield session

    async def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Returns all discovered tools in OpenAI function calling format."""
        if not self._openai_tools:
            async with self.get_session():
                pass
        return self._openai_tools

    async def get_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Alias for get_openai_tools for drop-in compatibility."""
        if force_refresh or not self._openai_tools:
            async with self.get_session() as session:
                response = await session.list_tools()
                self._raw_tools = response.tools
                self._tool_map = {t.name: t for t in self._raw_tools}
                self._openai_tools = [_mcp_tool_to_openai(t) for t in self._raw_tools]
        return self._openai_tools

    def get_tool_names(self) -> List[str]:
        """Returns the list of all discovered Notion tool names."""
        return list(self._tool_map.keys())

    def _normalize_id(self, raw_id: Optional[str]) -> str:
        """Strips hyphens from UUID if present, returning standard 32-char hex string."""
        if not raw_id:
            return ""
        val = str(raw_id).strip()
        match = UUID_PATTERN.search(val)
        if match:
            return match.group(0).replace("-", "")
        return val.replace("-", "")

    def resolve_tool_name(self, name: str) -> str:
        """Resolves alias or normalized name to exact Notion MCP tool name."""
        clean_name = name.strip()
        if clean_name in self._tool_map:
            return clean_name

        lower_name = clean_name.lower().replace("-", "_")
        if lower_name in TOOL_ALIASES and TOOL_ALIASES[lower_name] in self._tool_map:
            return TOOL_ALIASES[lower_name]

        normalized = clean_name.lower().replace("_", "-")
        for candidate in self._tool_map.keys():
            if candidate.lower() == normalized:
                return candidate

        for candidate in self._tool_map.keys():
            cand_clean = candidate.lower().replace("-", "_")
            if lower_name in cand_clean or cand_clean.endswith(lower_name):
                return candidate

        return clean_name

    def adapt_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically adapts incoming parameters to Notion OpenAPI conventions."""
        args = dict(arguments or {})

        for id_field in ("page_id", "block_id", "database_id", "data_source_id", "user_id", "id"):
            if id_field in args and isinstance(args[id_field], str):
                args[id_field] = self._normalize_id(args[id_field])

        if tool_name == "API-post-search":
            query = args.get("query") or args.get("title") or args.get("search") or ""
            payload = {"query": str(query)}
            if "filter" in args and isinstance(args["filter"], dict):
                payload["filter"] = args["filter"]
            return payload

        if tool_name == "API-post-page":
            parent = args.get("parent")
            raw_parent = args.get("parent_id") or args.get("page_id")
            default_parent = os.getenv("NOTION_PARENT_PAGE_ID")

            if not isinstance(parent, dict):
                has_uuid = bool(raw_parent and UUID_PATTERN.search(str(raw_parent)))
                target_parent = raw_parent if has_uuid else default_parent
                if target_parent and UUID_PATTERN.search(str(target_parent)):
                    args["parent"] = {"page_id": self._normalize_id(str(target_parent))}
                elif default_parent and UUID_PATTERN.search(str(default_parent)):
                    args["parent"] = {"page_id": self._normalize_id(str(default_parent))}

            if "properties" not in args:
                title = str(args.get("title") or args.get("name") or "Untitled")
                args["properties"] = {"title": [{"text": {"content": title}}]}

            if "children" not in args and "content" in args:
                args["children"] = [{
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": str(args["content"])}}]},
                }]

        if tool_name == "API-patch-block-children":
            if "children" not in args and "content" in args:
                block_type = str(args.get("block_type") or "paragraph").lower()
                text_content = str(args["content"])
                schema = {"rich_text": [{"text": {"content": text_content}}]}
                if block_type in ("to_do", "todo"):
                    block_obj = {"object": "block", "type": "to_do", "to_do": schema}
                elif block_type in ("heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"):
                    block_obj = {"object": "block", "type": block_type, block_type: schema}
                else:
                    block_obj = {"object": "block", "type": "paragraph", "paragraph": schema}
                args["children"] = [block_obj]

        return args

    async def call(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes a tool on the Notion MCP server session."""
        actual_tool = self.resolve_tool_name(tool_name)
        adapted_args = self.adapt_arguments(actual_tool, arguments or {})

        logger.info(f"Calling Notion MCP Tool '{actual_tool}' with args: {adapted_args}")

        async with self.get_session() as session:
            try:
                result = await session.call_tool(actual_tool, adapted_args)

                text_outputs: List[str] = []
                if hasattr(result, "content") and result.content:
                    for item in result.content:
                        if hasattr(item, "text") and item.text:
                            text_outputs.append(item.text)
                        elif hasattr(item, "data"):
                            text_outputs.append(str(item.data))

                joined_text = "\n".join(text_outputs) if text_outputs else str(result)

                if getattr(result, "isError", False):
                    logger.error(f"Notion MCP error for '{actual_tool}': {joined_text}")
                    return f"Error from Notion MCP ({actual_tool}): {joined_text}"

                return joined_text

            except Exception as exc:
                logger.error(f"Failed to execute Notion MCP tool '{actual_tool}': {exc}", exc_info=True)
                return f"Execution error in {actual_tool}: {str(exc)}"

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Alias for call() for drop-in backward compatibility."""
        return await self.call(tool_name, arguments)

    async def get_tools_documentation(self) -> str:
        """Generates formatted documentation of all tools for LLM prompts."""
        tools = await self.get_openai_tools()
        if not tools:
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


# Singleton instances
registry = NotionToolRegistry()
notion_mcp_client = registry

__all__ = [
    "NotionToolRegistry",
    "registry",
    "notion_mcp_client",
    "TOOL_ALIASES",
]

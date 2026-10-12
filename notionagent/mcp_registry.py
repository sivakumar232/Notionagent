"""Compatibility alias module for NotionToolRegistry.

Re-exports all registry classes and singletons from notionagent.mcp_client.
"""

from notionagent.mcp_client import (
    NotionToolRegistry,
    registry,
    notion_mcp_client,
    TOOL_ALIASES,
)

__all__ = [
    "NotionToolRegistry",
    "registry",
    "notion_mcp_client",
    "TOOL_ALIASES",
]

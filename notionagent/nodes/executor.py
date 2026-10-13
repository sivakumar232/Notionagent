import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from notionagent.mcp_client import notion_mcp_client
from notionagent.state import AgentState

logger = logging.getLogger("notionagent.executor")

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}",
    re.IGNORECASE,
)


class StructuredWorkspaceContext(BaseModel):
    """Formal schema for workspace state tracking across execution steps."""
    pages: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of lowercase page titles/names to Notion page IDs"
    )
    databases: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of lowercase database titles/names to Notion database IDs"
    )
    last_page_id: Optional[str] = None
    last_database_id: Optional[str] = None
    last_found_id: Optional[str] = None
    objective: Optional[str] = None

    def update_from_raw(self, raw_dict: Dict[str, Any]) -> None:
        """Merges arbitrary dictionary data into structured state safely."""
        if not raw_dict or not isinstance(raw_dict, dict):
            return
        for key, val in raw_dict.items():
            if key == "pages" and isinstance(val, dict):
                self.pages.update(val)
            elif key == "databases" and isinstance(val, dict):
                self.databases.update(val)
            elif key in self.__dict__ and val is not None:
                setattr(self, key, val)


def _extract_title(item: Dict[str, Any]) -> Optional[str]:
    """Extracts human-readable title from Notion page or database JSON."""
    # Database title: item["title"][0]["plain_text"]
    if "title" in item and isinstance(item["title"], list) and item["title"]:
        first_title = item["title"][0]
        if isinstance(first_title, dict) and "plain_text" in first_title:
            return first_title["plain_text"].strip()

    # Page title: item["properties"]["title" or "Name"]["title"][0]["plain_text"]
    properties = item.get("properties")
    if isinstance(properties, dict):
        for prop_val in properties.values():
            if isinstance(prop_val, dict) and prop_val.get("type") == "title":
                title_list = prop_val.get("title", [])
                if isinstance(title_list, list) and title_list:
                    return title_list[0].get("plain_text", "").strip()

    return None


def _extract_entities_from_output(
    output_str: str,
    context: StructuredWorkspaceContext,
) -> None:
    """Extracts page/database IDs and maps titles to IDs for all returned items."""
    if not output_str or not isinstance(output_str, str):
        return

    try:
        data = json.loads(output_str)
        items: List[Dict[str, Any]] = []

        if isinstance(data, dict):
            # If output is a paginated search or query result
            if "results" in data and isinstance(data["results"], list):
                items = data["results"]
            # If output is a single page or database object
            elif "id" in data:
                items = [data]

        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                continue

            item_id = str(item["id"])
            context.last_found_id = item_id
            obj_type = item.get("object")
            title = _extract_title(item)

            if obj_type == "page":
                context.last_page_id = item_id
                if title:
                    context.pages[title.lower()] = item_id
            elif obj_type == "database":
                context.last_database_id = item_id
                if title:
                    context.databases[title.lower()] = item_id

    except Exception:
        # Fallback regex search for bare UUID
        match = UUID_PATTERN.search(output_str)
        if match:
            context.last_found_id = match.group(0)


def _resolve_step_input(
    tool_name: str,
    tool_input: Dict[str, Any],
    context: StructuredWorkspaceContext,
) -> Dict[str, Any]:
    """Resolves placeholder IDs using the structured entity registry."""
    resolved = dict(tool_input)

    # 1. Match by page_name or database_name if provided in tool input
    page_name = str(resolved.get("page_name") or resolved.get("title") or "").strip().lower()
    if page_name and page_name in context.pages:
        resolved.setdefault("page_id", context.pages[page_name])

    database_name = str(resolved.get("database_name") or "").strip().lower()
    if database_name and database_name in context.databases:
        resolved.setdefault("database_id", context.databases[database_name])

    # 2. Check each key for placeholder tokens (e.g. '<to_be_filled>', '{new_page_id}')
    for key, val in list(resolved.items()):
        val_str = str(val or "").strip()
        is_placeholder = (
            val is None
            or val_str == ""
            or val_str.startswith("<")
            or val_str.startswith("{")
            or val_str.endswith("}")
            or val_str.endswith(">")
            or "to_be_filled" in val_str.lower()
            or "placeholder" in val_str.lower()
            or "new_page" in val_str.lower()
            or (key in ("page_id", "block_id", "id", "parent_id", "database_id") and not UUID_PATTERN.search(val_str))
        )

        if is_placeholder:
            if key in ("page_id", "block_id", "id"):
                resolved_id = context.last_page_id or context.last_found_id
                if resolved_id:
                    resolved[key] = resolved_id
                    logger.info(f"Resolved placeholder '{key}' ({val_str}) -> '{resolved_id}'")

            elif key in ("parent_id", "database_id"):
                resolved_id = (
                    context.last_database_id
                    or context.last_found_id
                    or context.last_page_id
                )
                if resolved_id:
                    resolved[key] = resolved_id
                    logger.info(f"Resolved placeholder '{key}' ({val_str}) -> '{resolved_id}'")

    if "parent" in resolved and isinstance(resolved["parent"], dict):
        p_id = str(resolved["parent"].get("page_id") or "")
        if p_id and (p_id.startswith("{") or p_id.startswith("<") or not UUID_PATTERN.search(p_id)):
            target_id = context.last_page_id or context.last_found_id or os.getenv("NOTION_PARENT_PAGE_ID")
            if target_id:
                resolved["parent"]["page_id"] = target_id

    # 3. Smart parameter injection based on specific tool type
    clean_tool = tool_name.lower().replace("-", "_")
    if "create" in clean_tool or "post_page" in clean_tool:
        if "parent_id" not in resolved and "parent" not in resolved:
            target_parent = context.last_page_id or context.last_found_id or os.getenv("NOTION_PARENT_PAGE_ID")
            if target_parent:
                resolved["parent_id"] = target_parent
    elif "block" in clean_tool or "patch" in clean_tool or "children" in clean_tool:
        if "block_id" not in resolved:
            target_block = context.last_page_id or context.last_found_id
            if target_block:
                resolved["block_id"] = target_block
    elif "search" not in clean_tool and "user" not in clean_tool:
        if "page_id" not in resolved and context.last_page_id:
            resolved["page_id"] = context.last_page_id

    return resolved


def _unwrap_exception(exc: BaseException) -> str:
    """Recursively unpacks Python ExceptionGroup to get the underlying error message."""
    if hasattr(exc, "exceptions") and exc.exceptions:
        return "; ".join(_unwrap_exception(sub) for sub in exc.exceptions)
    return str(exc)


async def _execute_step_with_retry(
    tool_name: str,
    tool_input: Dict[str, Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Executes a single MCP tool call with automatic retries for transient errors."""
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return await notion_mcp_client.call_tool(tool_name, tool_input)
        except Exception as e:
            last_err = e
            err_msg = _unwrap_exception(e)
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed for '{tool_name}': {err_msg}"
            )
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    raise last_err or RuntimeError(f"Step '{tool_name}' failed after {max_retries} attempts.")


async def executor_node(state: AgentState) -> Dict[str, Any]:
    """4. Executor node: Executes plan steps with isolated retries and structured context."""
    plan = list(state.get("plan") or [])
    step_results = list(state.get("step_results") or [])
    current_idx = int(state.get("current_step_index") or 0)

    # Initialize structured context from existing state
    context = StructuredWorkspaceContext()
    raw_context = dict(state.get("workspace_context") or {})
    context.update_from_raw(raw_context)

    # If all steps already executed or plan is empty
    if not plan or current_idx >= len(plan):
        logger.info(f"All {len(plan)} steps executed. Proceeding to synthesizer.")
        return {
            "current_step_index": len(plan),
            "step_results": step_results,
            "workspace_context": context.model_dump(),
            "has_error": False,
        }

    # Execute remaining steps internally with step-level isolation
    for idx in range(current_idx, len(plan)):
        current_step = plan[idx]
        tool_name = current_step.get("tool", "")
        raw_input = dict(current_step.get("tool_input") or {})

        resolved_input = _resolve_step_input(tool_name, raw_input, context)

        logger.info(
            f"Executing Step {idx + 1}/{len(plan)}: [{tool_name}] "
            f"Description: '{current_step.get('description')}' with input: {resolved_input}"
        )

        try:
            output = await _execute_step_with_retry(
                tool_name=tool_name,
                tool_input=resolved_input,
                max_retries=3,
                base_delay=1.0,
            )

            # Update structured entity registry from tool output
            _extract_entities_from_output(output, context)

            step_record = {
                "step_number": current_step.get("step_number", idx + 1),
                "tool": tool_name,
                "description": current_step.get("description", ""),
                "input": resolved_input,
                "output": output,
            }
            step_results.append(step_record)

        except Exception as e:
            detailed_error = _unwrap_exception(e)
            logger.exception(f"Execution failed on step {idx + 1}: {detailed_error}")
            return {
                "current_step_index": idx,
                "step_results": step_results,
                "workspace_context": context.model_dump(),
                "has_error": True,
                "error_message": f"Step {idx + 1} ({tool_name}) failed: {detailed_error}",
            }

    return {
        "current_step_index": len(plan),
        "step_results": step_results,
        "workspace_context": context.model_dump(),
        "has_error": False,
        "error_message": None,
    }

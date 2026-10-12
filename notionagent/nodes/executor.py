import json
import logging
import re
from typing import Any, Dict
from notionagent.mcp_client import notion_mcp_client
from notionagent.state import AgentState

logger = logging.getLogger("notionagent.executor")

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}",
    re.IGNORECASE,
)


def _extract_ids_from_output(output_str: str) -> Dict[str, str]:
    """Extracts page/database IDs from Notion API responses."""
    extracted: Dict[str, str] = {}
    if not output_str or not isinstance(output_str, str):
        return extracted

    try:
        data = json.loads(output_str)
        if isinstance(data, dict):
            # Check for list results (e.g. search output)
            results = data.get("results")
            if isinstance(results, list) and len(results) > 0:
                first_item = results[0]
                if isinstance(first_item, dict) and "id" in first_item:
                    item_id = str(first_item["id"])
                    extracted["last_found_id"] = item_id
                    item_type = first_item.get("object")
                    if item_type == "page":
                        extracted["last_page_id"] = item_id
                    elif item_type == "database":
                        extracted["last_database_id"] = item_id

            # Check for single object response (e.g. create_page output)
            elif "id" in data:
                obj_id = str(data["id"])
                extracted["last_found_id"] = obj_id
                if data.get("object") == "page":
                    extracted["last_page_id"] = obj_id
                elif data.get("object") == "database":
                    extracted["last_database_id"] = obj_id

    except Exception:
        # Fallback regex search for UUID in raw text
        match = UUID_PATTERN.search(output_str)
        if match:
            extracted["last_found_id"] = match.group(0)

    return extracted


def _resolve_step_input(
    tool_name: str,
    tool_input: Dict[str, Any],
    workspace_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Replaces placeholders with real IDs discovered in previous steps."""
    resolved = dict(tool_input)

    # Check each argument for placeholder patterns like <to_be_filled...> or empty IDs
    for key, val in list(resolved.items()):
        val_str = str(val or "").strip()
        is_placeholder = (
            val is None
            or val_str == ""
            or val_str.startswith("<")
            or "to_be_filled" in val_str.lower()
            or "placeholder" in val_str.lower()
        )

        if is_placeholder:
            if key in ("page_id", "block_id", "id"):
                resolved_id = (
                    workspace_context.get("last_page_id")
                    or workspace_context.get("last_found_id")
                )
                if resolved_id:
                    resolved[key] = resolved_id
                    logger.info(f"Resolved placeholder for '{key}' -> '{resolved_id}'")

            elif key in ("parent_id", "database_id"):
                resolved_id = (
                    workspace_context.get("last_database_id")
                    or workspace_context.get("last_found_id")
                    or workspace_context.get("last_page_id")
                )
                if resolved_id:
                    resolved[key] = resolved_id
                    logger.info(f"Resolved placeholder for '{key}' -> '{resolved_id}'")

    # If the tool requires page_id but it's completely missing, inject from context
    if "page" in tool_name.lower() or "block" in tool_name.lower():
        if "page_id" not in resolved and "last_page_id" in workspace_context:
            resolved["page_id"] = workspace_context["last_page_id"]

    return resolved


def _unwrap_exception(exc: BaseException) -> str:
    """Recursively unpacks Python ExceptionGroup to get the underlying error message."""
    if hasattr(exc, "exceptions") and exc.exceptions:
        return "; ".join(_unwrap_exception(sub) for sub in exc.exceptions)
    return str(exc)


async def executor_node(state: AgentState) -> Dict[str, Any]:
    """4. Executor node: Executes one planned step at a time via Notion MCP with ID propagation."""
    plan = list(state.get("plan") or [])
    current_idx = int(state.get("current_step_index") or 0)
    step_results = list(state.get("step_results") or [])
    workspace_context = dict(state.get("workspace_context") or {})

    if current_idx >= len(plan):
        logger.info(f"All {len(plan)} steps executed. Proceeding to synthesizer.")
        return {
            "current_step_index": current_idx,
            "step_results": step_results,
            "has_error": False,
        }

    current_step = plan[current_idx]
    tool_name = current_step.get("tool", "")
    raw_input = dict(current_step.get("tool_input") or {})

    # Dynamically inject/resolve IDs from previous step outputs
    resolved_input = _resolve_step_input(tool_name, raw_input, workspace_context)

    logger.info(
        f"Executing Step {current_idx + 1}/{len(plan)}: [{tool_name}] "
        f"Description: '{current_step.get('description')}' with input: {resolved_input}"
    )

    try:
        output = await notion_mcp_client.call_tool(tool_name, resolved_input)

        # Extract any newly discovered page / database IDs into workspace_context
        extracted_ids = _extract_ids_from_output(output)
        if extracted_ids:
            workspace_context.update(extracted_ids)
            logger.info(f"Updated workspace context with discovered IDs: {extracted_ids}")

        step_record = {
            "step_number": current_step.get("step_number", current_idx + 1),
            "tool": tool_name,
            "description": current_step.get("description", ""),
            "input": resolved_input,
            "output": output,
        }
        step_results.append(step_record)

        return {
            "current_step_index": current_idx + 1,
            "step_results": step_results,
            "workspace_context": workspace_context,
            "has_error": False,
            "error_message": None,
        }

    except Exception as e:
        detailed_error = _unwrap_exception(e)
        logger.exception(f"Execution failed on step {current_idx + 1}: {detailed_error}")
        return {
            "has_error": True,
            "error_message": f"Step {current_idx + 1} ({tool_name}) failed: {detailed_error}",
            "step_results": step_results,
            "workspace_context": workspace_context,
        }

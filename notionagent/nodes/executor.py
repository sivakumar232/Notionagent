import logging
from typing import Any, Dict
from notionagent.mcp_client import notion_mcp_client
from notionagent.state import AgentState

logger = logging.getLogger("notionagent.executor")


async def executor_node(state: AgentState) -> Dict[str, Any]:
    """4. Executor node: Executes one planned step at a time via Notion MCP."""
    plan = list(state.get("plan") or [])
    current_idx = int(state.get("current_step_index") or 0)
    step_results = list(state.get("step_results") or [])
    workspace_context = dict(state.get("workspace_context") or {})

    if current_idx >= len(plan):
        logger.info(f"All {len(plan)} steps executed. Proceeding.")
        return {
            "current_step_index": current_idx,
            "step_results": step_results,
            "has_error": False,
        }

    current_step = plan[current_idx]
    tool_name = current_step.get("tool", "")
    tool_input = dict(current_step.get("tool_input") or {})

    logger.info(
        f"Executing Step {current_idx + 1}/{len(plan)}: [{tool_name}] {current_step.get('description')}"
    )

    try:
        output = await notion_mcp_client.call_tool(tool_name, tool_input)

        step_record = {
            "step_number": current_step.get("step_number", current_idx + 1),
            "tool": tool_name,
            "description": current_step.get("description", ""),
            "input": tool_input,
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
        logger.exception(f"Execution failed on step {current_idx + 1}: {e}")
        return {
            "has_error": True,
            "error_message": f"Step {current_idx + 1} ({tool_name}) failed: {str(e)}",
            "step_results": step_results,
        }

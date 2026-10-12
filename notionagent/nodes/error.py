from typing import Any, Dict
from notionagent.state import AgentState


def error_node(state: AgentState) -> Dict[str, Any]:
    """5. Error node: Reports step failure and sets final error message."""
    error_message = state.get("error_message") or "An unexpected error occurred during execution."
    return {
        "has_error": True,
        "error_message": error_message,
        "final_response": f"I could not complete the request due to an error: {error_message}",
    }

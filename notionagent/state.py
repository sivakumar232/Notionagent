from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Central LangGraph state for Notion Agent."""

    user_query: str
    is_valid_request: bool
    rejection_reason: Optional[str]
    plan: List[Dict[str, Any]]
    current_step_index: int
    has_error: bool
    error_message: Optional[str]
    final_response: Optional[str]

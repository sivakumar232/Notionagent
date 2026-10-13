from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class InputState(TypedDict):
    """Public input payload provided by the caller or UI."""

    user_query: str


class OutputState(TypedDict):
    """Public output payload returned to the caller or UI."""

    user_query: str
    is_valid_request: bool
    rejection_reason: Optional[str]
    final_response: Optional[str]
    has_error: bool
    error_message: Optional[str]
    plan: List[Dict[str, Any]]
    step_results: List[Dict[str, Any]]
    is_goal_satisfied: Optional[bool]
    reflection_count: Optional[int]


class AgentState(TypedDict):
    """Central LangGraph internal state for NotionAgent execution.
    
    Contains input, guardrail status, plan, runtime execution observations,
    workspace context (page IDs, database IDs), reflection evaluations, and error handling.
    """

    # Public user input
    user_query: str

    # Guardrail status (from Jev System 1)
    is_valid_request: bool
    rejection_reason: Optional[str]

    # Planning & Multi-step Execution
    plan: List[Dict[str, Any]]
    current_step_index: int
    step_results: List[Dict[str, Any]]
    workspace_context: Dict[str, Any]

    # Reflection & Evaluator Loop
    reflection_count: int
    max_reflections: int
    is_goal_satisfied: bool
    reflection_critique: Optional[str]

    # Error handling & Circuit breaker
    has_error: bool
    error_message: Optional[str]

    # Synthesized final response
    final_response: Optional[str]


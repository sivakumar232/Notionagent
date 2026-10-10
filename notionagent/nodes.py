from typing import Any, Dict


def guard_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """1. Guardrail node: Fast classification for valid vs. invalid request."""
    return {}


def rejection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """2. Rejection node: Handles rejected / off-topic requests."""
    return {}


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """3. Planner node: Decomposes request into sequential steps."""
    return {}


def executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """4. Executor node: Executes one step at a time with tool calling."""
    return {}


def error_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """5. Error node: Reports step failure and stops execution."""
    return {}


def synthesizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """6. Synthesizer node: Assembles final answer after all steps finish."""
    return {}

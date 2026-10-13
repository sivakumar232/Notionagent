from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from notionagent.nodes import (
    guard_node,
    rejection_node,
    planner_node,
    executor_node,
    reflector_node,
    error_node,
    synthesizer_node,
)
from notionagent.state import AgentState, InputState, OutputState


# 2. Conditional Routers
def route_after_guard(state: AgentState) -> str:
    """Routes to planner if valid, or rejection if invalid."""
    if state.get("is_valid_request", False):
        return "planner"
    return "rejection"


def route_after_executor(state: AgentState) -> str:
    """Routes to error if fatal, otherwise passes to reflector for evaluation."""
    if state.get("has_error", False) and state.get("reflection_count", 0) >= state.get("max_reflections", 3):
        return "error"
    return "reflector"


def route_after_reflector(state: AgentState) -> str:
    """Routes to synthesizer if goal satisfied or max reflections hit, otherwise loops back to planner."""
    is_satisfied = bool(state.get("is_goal_satisfied", False))
    reflection_count = int(state.get("reflection_count", 0))
    max_reflections = int(state.get("max_reflections", 3))

    if is_satisfied or reflection_count >= max_reflections:
        return "synthesizer"

    return "planner"


# 3. Graph Assembly
builder = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

# Add Nodes
builder.add_node("guard", guard_node)
builder.add_node("rejection", rejection_node)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("reflector", reflector_node)
builder.add_node("error", error_node)
builder.add_node("synthesizer", synthesizer_node)

# Add Edges
builder.add_edge(START, "guard")
builder.add_conditional_edges(
    "guard",
    route_after_guard,
    {
        "planner": "planner",
        "rejection": "rejection",
    },
)
builder.add_edge("rejection", END)

builder.add_edge("planner", "executor")

builder.add_conditional_edges(
    "executor",
    route_after_executor,
    {
        "error": "error",
        "reflector": "reflector",
    },
)

builder.add_conditional_edges(
    "reflector",
    route_after_reflector,
    {
        "planner": "planner",
        "synthesizer": "synthesizer",
    },
)

builder.add_edge("error", END)
builder.add_edge("synthesizer", END)

# Compile the Graph
graph = builder.compile()


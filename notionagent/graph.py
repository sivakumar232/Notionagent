from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from notionagent.nodes import (
    guard_node,
    rejection_node,
    context_loader_node,
    planner_node,
    executor_node,
    reflector_node,
    error_node,
    synthesizer_node,
)
from notionagent.state import AgentState, InputState, OutputState


# 2. Conditional Routers
def route_after_guard(state: AgentState) -> str:
    """Routes to context_loader if valid, or rejection if invalid."""
    if state.get("is_valid_request", False):
        return "context_loader"
    return "rejection"


def route_after_executor(state: AgentState) -> str:
    """Routes after single step execution.
    - If error and max reflections reached: route to error
    - If error and reflections left: route to reflector for critique & replanning
    - If step succeeded and more steps remain in plan: route back to executor for next step
    - If all steps in plan completed: route to reflector for final goal evaluation
    """
    has_error = bool(state.get("has_error", False))
    reflection_count = int(state.get("reflection_count", 0))
    max_reflections = int(state.get("max_reflections", 3))

    if has_error:
        if reflection_count >= max_reflections:
            return "error"
        return "reflector"

    plan = list(state.get("plan") or [])
    current_idx = int(state.get("current_step_index") or 0)

    # If there are more steps remaining in the current plan, execute next step
    if current_idx < len(plan):
        return "executor"

    # All planned steps are finished; route to reflector for goal verification
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
builder.add_node("context_loader", context_loader_node)
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
        "context_loader": "context_loader",
        "rejection": "rejection",
    },
)
builder.add_edge("rejection", END)

builder.add_edge("context_loader", "planner")
builder.add_edge("planner", "executor")

builder.add_conditional_edges(
    "executor",
    route_after_executor,
    {
        "executor": "executor",
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


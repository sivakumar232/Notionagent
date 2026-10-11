from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from notionagent.nodes import (
    guard_node,
    rejection_node,
    planner_node,
    executor_node,
    error_node,
    synthesizer_node,
)


from notionagent.state import AgentState



# 2. Conditional Routers
def route_after_guard(state: AgentState) -> str:
    """Routes to planner if valid, or rejection if invalid."""
    if state.get("is_valid_request", False):
        return "planner"
    return "rejection"


def route_after_executor(state: AgentState) -> str:
    """Routes to error, next executor step, or synthesizer."""
    if state.get("has_error", False):
        return "error"
    
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)
    
    if current_idx < len(plan):
        return "executor"
    return "synthesizer"


# 3. Graph Assembly
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("guard", guard_node)
builder.add_node("rejection", rejection_node)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
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
    }
)
builder.add_edge("rejection", END)

builder.add_edge("planner", "executor")

builder.add_conditional_edges(
    "executor",
    route_after_executor,
    {
        "error": "error",
        "executor": "executor",
        "synthesizer": "synthesizer",
    }
)
builder.add_edge("error", END)
builder.add_edge("synthesizer", END)

# Compile the Graph
graph = builder.compile()

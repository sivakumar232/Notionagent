import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from notionagent.nodes.planner import get_azure_chat_client
from notionagent.prompts import reflector_prompt_template
from notionagent.state import AgentState

logger = logging.getLogger("notionagent.reflector")


class ReflectionVerdict(BaseModel):
    """Structured evaluation verdict from the Reflector node."""
    is_goal_satisfied: bool = Field(
        description="True if the user's objective was fully achieved; False if steps failed or further action is required."
    )
    critique: Optional[str] = Field(
        default=None,
        description="Actionable explanation of why the goal is not met and what alternative tool or parameters to try next."
    )


def build_reflector_chain(client: Optional[AzureChatOpenAI] = None):
    """Constructs LCEL chain for reflection evaluation with structured output."""
    if client is None:
        client = get_azure_chat_client(temperature=0.0)

    structured_llm = client.with_structured_output(ReflectionVerdict, method="function_calling")
    return reflector_prompt_template | structured_llm


async def reflector_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates the outcomes of executed steps and decides if replanning is needed."""
    user_query = str(state.get("user_query") or "").strip()
    step_results = list(state.get("step_results") or [])
    workspace_context = dict(state.get("workspace_context") or {})
    reflection_count = int(state.get("reflection_count") or 0)
    max_reflections = int(state.get("max_reflections") or 3)
    has_error = bool(state.get("has_error") or False)
    error_message = state.get("error_message")

    logger.info(
        f"Evaluating execution results (Attempt {reflection_count + 1}/{max_reflections}) "
        f"for query: '{user_query}'"
    )

    # If no actions were taken at all and there's a hard error, stop immediately
    if not step_results and has_error:
        logger.warning("No step results and hard error present. Aborting reflection.")
        return {
            "is_goal_satisfied": False,
            "reflection_critique": error_message,
            "reflection_count": reflection_count + 1,
        }

    # Format step summary for concise evaluation
    summarized_steps = []
    for s in step_results:
        summarized_steps.append({
            "step": s.get("step_number"),
            "tool": s.get("tool"),
            "input": s.get("input"),
            "output_preview": str(s.get("output", ""))[:300],
        })

    try:
        chain = build_reflector_chain()
        verdict: ReflectionVerdict = await chain.ainvoke({
            "user_query": user_query,
            "step_results": summarized_steps,
            "workspace_context": workspace_context,
            "reflection_count": reflection_count + 1,
            "max_reflections": max_reflections,
        })

        if not isinstance(verdict, ReflectionVerdict):
            verdict = ReflectionVerdict.model_validate(verdict)

        new_reflection_count = reflection_count + 1
        logger.info(
            f"Reflection verdict: is_goal_satisfied={verdict.is_goal_satisfied}, "
            f"critique='{verdict.critique}' (attempt {new_reflection_count}/{max_reflections})"
        )

        return {
            "is_goal_satisfied": verdict.is_goal_satisfied,
            "reflection_critique": verdict.critique if not verdict.is_goal_satisfied else None,
            "reflection_count": new_reflection_count,
            "has_error": False,  # Reset error flag to give the replanned attempt a fresh chance
            "error_message": None,
        }

    except Exception as e:
        logger.exception(f"Error during reflection evaluation: {e}")
        # Graceful fallback: assume satisfied to avoid blocking synthesizer
        return {
            "is_goal_satisfied": True,
            "reflection_critique": None,
            "reflection_count": reflection_count + 1,
        }

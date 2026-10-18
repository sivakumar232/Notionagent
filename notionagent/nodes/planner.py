import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from notionagent.mcp_client import notion_mcp_client
from notionagent.prompts import planner_prompt_template
from notionagent.state import AgentState

# Setup logger
logger = logging.getLogger("notionagent.planner")

load_dotenv()


class Step(BaseModel):
    step_number: int = Field(description="Sequential step index starting from 1")
    description: str = Field(description="What this step accomplishes")
    tool: str = Field(description="Name of the tool or alias to execute")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class Plan(BaseModel):
    objective: str = Field(description="Summary of the user query")
    steps: List[Step] = Field(default_factory=list, description="Ordered list of execution steps")


def get_azure_chat_client(temperature: float = 0.0) -> AzureChatOpenAI:
    """Initializes and returns an AzureChatOpenAI client configured from environment variables."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-4.1-mini"
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI credentials missing. Please set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_API_KEY in your environment or .env file."
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        max_retries=2,
        timeout=30.0,
    )

def build_planner_chain(client: Optional[AzureChatOpenAI] = None):
    """Builds and returns the LCEL chain with Azure OpenAI structured outputs."""
    if client is None:
        client = get_azure_chat_client(temperature=0.0)

    # Use function calling method to support dynamic dictionary structures in tool_input
    structured_llm = client.with_structured_output(Plan, method="function_calling")
    return planner_prompt_template | structured_llm


async def generate_plan(
    user_query: str,
    page_outline: Optional[str] = None,
    workspace_context: Optional[Dict[str, Any]] = None,
    reflection_critique: Optional[str] = None,
    step_results: Optional[List[Dict[str, Any]]] = None,
    client: Optional[AzureChatOpenAI] = None,
) -> Plan:
    """Invokes the planner chain to generate a structured Plan with dynamic MCP tools and reflection feedback."""
    chain = build_planner_chain(client=client)

    # Fetch live tool specifications directly from the Notion MCP client
    tools_doc = await notion_mcp_client.get_tools_documentation()

    # Format feedback from previous attempt if replanning
    feedback_parts = []
    if step_results:
        feedback_parts.append("\n--- COMPLETED STEPS (DO NOT RE-RUN OR DUPLICATE) ---")
        feedback_parts.append("The following actions were ALREADY executed successfully in Notion:")
        for s in step_results:
            s_num = s.get("step_number")
            tool = s.get("tool")
            desc = s.get("description", "")
            out_preview = str(s.get("output", ""))[:200]
            feedback_parts.append(f"  • Step {s_num} [{tool}] ({desc}) -> Output: {out_preview}")
        feedback_parts.append(
            "CRITICAL: Do NOT duplicate the above steps (e.g. do NOT re-create pages or blocks that were already created). "
            "Generate ONLY the remaining steps needed to fulfill the user's objective, using the IDs from completed steps."
        )

    if reflection_critique:
        feedback_parts.append("\n--- PREVIOUS ATTEMPT FEEDBACK / CRITIQUE ---")
        feedback_parts.append(f"Critique: {reflection_critique}")
        feedback_parts.append("Please adjust your remaining plan steps to resolve this critique and fulfill the objective.")

    if feedback_parts:
        feedback_parts.append("-----------------------------------------------------")
        reflection_feedback = "\n".join(feedback_parts)
    else:
        reflection_feedback = ""

    plan = await chain.ainvoke({
        "user_query": user_query,
        "page_outline": page_outline or "(No target page outline identified)",
        "workspace_context": workspace_context or {},
        "tools_documentation": tools_doc,
        "reflection_feedback": reflection_feedback,
    })

    if not isinstance(plan, Plan):
        plan = Plan.model_validate(plan)

    return plan


async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """3. Planner node: Decomposes request into sequential steps with reflection awareness."""
    user_query = str(state.get("user_query") or "").strip()
    if not user_query:
        logger.warning("Empty user_query passed to planner_node.")
        return {
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "has_error": True,
            "error_message": "Empty query received by planner.",
        }

    page_outline = str(state.get("page_outline") or "").strip()
    workspace_context = dict(state.get("workspace_context") or {})
    step_results = list(state.get("step_results") or [])
    reflection_critique = state.get("reflection_critique")

    try:
        if reflection_critique:
            logger.info(f"Replanning for query '{user_query}' with critique: {reflection_critique}")
        else:
            logger.info(f"Generating initial plan for query: '{user_query}' with outline: {page_outline[:80]}...")

        plan = await generate_plan(
            user_query=user_query,
            page_outline=page_outline,
            workspace_context=workspace_context,
            reflection_critique=reflection_critique,
            step_results=step_results,
        )

        plan_steps = [step.model_dump() for step in plan.steps]
        workspace_context["objective"] = plan.objective

        logger.info(f"Plan created with {len(plan_steps)} steps. Objective: '{plan.objective}'")

        return {
            "plan": plan_steps,
            "current_step_index": 0,
            "step_results": step_results,
            "workspace_context": workspace_context,
            "reflection_critique": None,  # Consumed
            "has_error": False,
            "error_message": None,
        }

    except Exception as e:
        logger.exception(f"Error while generating plan in planner_node: {e}")
        return {
            "plan": [],
            "current_step_index": 0,
            "step_results": step_results,
            "has_error": True,
            "error_message": f"Planner failed to generate plan: {str(e)}",
        }

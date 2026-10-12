import logging
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from notionagent.nodes.planner import get_azure_chat_client
from notionagent.state import AgentState

logger = logging.getLogger("notionagent.synthesizer")


async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """6. Synthesizer node: Assembles final answer after all steps finish."""
    user_query = str(state.get("user_query") or "")
    step_results = list(state.get("step_results") or [])

    if not step_results:
        return {
            "final_response": "I could not find or complete any actions in your Notion workspace for this request.",
        }

    try:
        llm = get_azure_chat_client(temperature=0.3)
        messages = [
            SystemMessage(
                content="You are a helpful Notion Assistant. Synthesize a concise, clear final response to the user based on the executed actions and tool results. Highlight any created or found items."
            ),
            HumanMessage(
                content=f"User Query: {user_query}\n\nTool Execution Results:\n{step_results}\n\nFinal Answer:"
            ),
        ]
        response = await llm.ainvoke(messages)
        final_text = str(response.content).strip()
    except Exception as e:
        logger.warning(f"Synthesizer LLM fallback triggered: {e}")
        final_text = f"Successfully performed {len(step_results)} actions in your Notion workspace for: '{user_query}'."

    return {
        "final_response": final_text,
    }

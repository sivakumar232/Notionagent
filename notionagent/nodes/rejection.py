import logging
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from notionagent.state import AgentState

# Setup logger
logger = logging.getLogger("notionagent.rejection")

load_dotenv()

# Safe initialization of Azure OpenAI Client
client = None
try:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-4o-mini"
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if endpoint and api_key:
        client = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            temperature=0.3,
        )
    else:
        logger.warning("Azure OpenAI credentials not fully set. Using deterministic fallback.")
except Exception as e:
    logger.warning(f"Could not initialize AzureChatOpenAI client: {e}. Fallback active.")


REJECTION_SYSTEM_PROMPT = """You are the voice of a specialized Notion Workspace Assistant.
The user's query was filtered out by our security and scope guardrail.

Your task:
1. Politely and concisely inform the user that their request cannot be fulfilled.
2. Reference the specific reason why (e.g. off-topic, general coding, trivia, or prompt injection).
3. Briefly guide them on what you CAN do (e.g., searching Notion pages, managing sprint tasks, reviewing to-dos, tracking expenses, or organizing workspace documentation).
4. Keep your response under 3 sentences. Be helpful, professional, and friendly."""


def _get_deterministic_rejection(rejection_reason: str) -> str:
    """Fallback generator when LLM is unavailable."""
    return (
        f"I cannot process this request because it is outside the scope of your Notion workspace assistant. "
        f"({rejection_reason}) Please feel free to ask about searching documents, managing to-dos, or updating your Notion pages!"
    )


async def get_rejection(user_query: str, rejection_reason: str) -> str:
    """Asynchronously generates a polite rejection message using Azure OpenAI with fallback."""
    if client is None:
        return _get_deterministic_rejection(rejection_reason)

    try:
        messages = [
            {"role": "system", "content": REJECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User Query: '{user_query}'\n"
                    f"Guardrail Reason: '{rejection_reason}'\n\n"
                    f"Generate a polite rejection response:"
                ),
            },
        ]
        # Non-blocking async invocation
        response = await client.ainvoke(messages)
        return str(response.content)

    except Exception as e:
        logger.warning(f"Azure OpenAI generation failed ({type(e).__name__}): {e}. Using deterministic fallback.")
        return _get_deterministic_rejection(rejection_reason)


async def rejection_node(state: AgentState) -> Dict[str, Any]:
    """2. Rejection node: Handles rejected / off-topic requests and sets final_response safely."""
    try:
        user_query = str(state.get("user_query") or "")
        reason = str(state.get("rejection_reason") or "Out of workspace scope.")

        explanation = await get_rejection(user_query=user_query, rejection_reason=reason)

        return {
            "final_response": explanation,
        }
    except Exception as e:
        logger.exception(f"Unhandled exception in rejection_node: {e}")
        return {
            "final_response": "I cannot fulfill this request as it is outside the scope of the Notion workspace assistant.",
        }

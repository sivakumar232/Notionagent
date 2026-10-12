import logging
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from notionagent.prompts import rejection_prompt_template
from notionagent.state import AgentState

# Setup logger
logger = logging.getLogger("notionagent.rejection")

load_dotenv()


def get_azure_chat_client(temperature: float = 0.3) -> Optional[AzureChatOpenAI]:
    """Initializes and returns the latest LangChain AzureChatOpenAI client."""
    try:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or "gpt-4.1-mini"
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        if not endpoint or not api_key:
            logger.warning("Azure OpenAI credentials not fully set in environment. Fallback active.")
            return None

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            temperature=temperature,
            max_retries=2,
            timeout=15.0,
        )
    except Exception as e:
        logger.warning(f"Could not initialize AzureChatOpenAI client: {e}. Fallback active.")
        return None


def _get_deterministic_rejection(rejection_reason: str) -> str:
    """Fallback generator when LLM is unavailable."""
    return (
        f"I cannot process this request because it is outside the scope of your Notion workspace assistant. "
        f"({rejection_reason}) Please feel free to ask about searching documents, managing to-dos, or updating your Notion pages!"
    )


async def get_rejection(user_query: str, rejection_reason: str) -> str:
    """Asynchronously generates a polite rejection message using Azure OpenAI with fallback."""
    client = get_azure_chat_client(temperature=0.3)
    if client is None:
        return _get_deterministic_rejection(rejection_reason)

    try:
        # Pattern 1: Declarative LCEL Chain (Prompt | LLM | Parser)
        chain = rejection_prompt_template | client | StrOutputParser()
        response = await chain.ainvoke({
            "user_query": user_query,
            "rejection_reason": rejection_reason,
        })
        return str(response).strip()

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

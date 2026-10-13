import asyncio
import os
import sys
from typing import Any, AsyncGenerator, Dict, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from notionagent.graph import builder, graph

# Ensure environment variables (including LangSmith tracing flags) are loaded early
load_dotenv()


class NotionAgent:
    """Production Agent Runtime:

    Manages the lifecycle, checkpointer/threads, and execution
    of the Notion Agent LangGraph workflow.
    """

    def __init__(self, checkpointer: Optional[Any] = None):
        # Default to MemorySaver if no persistent checkpointer (e.g. Postgres) is provided
        self.checkpointer = checkpointer or MemorySaver()
        self.app = builder.compile(checkpointer=self.checkpointer)

    @classmethod
    async def create_with_postgres(cls, pool: Any) -> "NotionAgent":
        """Factory method that sets up persistent AsyncPostgresSaver checkpointer."""
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        return cls(checkpointer=checkpointer)

    def _prepare_initial_state(self, query: str) -> Dict[str, Any]:
        """Builds default initial state dictionary for a new run."""
        return {
            "user_query": query,
            "is_valid_request": False,
            "rejection_reason": None,
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "workspace_context": {},
            "reflection_count": 0,
            "max_reflections": 3,
            "is_goal_satisfied": False,
            "reflection_critique": None,
            "has_error": False,
            "error_message": None,
            "final_response": None,
        }


    async def astream(
        self, query: str, thread_id: str = "default_thread"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Asynchronously stream execution events step-by-step.

        Yields state updates after each superstep (ideal for FastAPI SSE or CLI).
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
            "run_name": "NotionAgent",
            "tags": ["notionagent"],
            "metadata": {"thread_id": thread_id},
        }
        initial_state = self._prepare_initial_state(query)

        async for event in self.app.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            yield event

    async def ainvoke(
        self, query: str, thread_id: str = "default_thread"
    ) -> Dict[str, Any]:
        """Asynchronously execute graph until completion and return final state."""
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
            "run_name": "NotionAgent",
            "tags": ["notionagent"],
            "metadata": {"thread_id": thread_id},
        }
        initial_state = self._prepare_initial_state(query)
        return await self.app.ainvoke(initial_state, config=config)

    def run(self, query: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        """Synchronous convenience method to run the agent."""
        return asyncio.run(self.ainvoke(query, thread_id=thread_id))


# -------------------------------------------------------------
# CLI Entrypoint for Local Testing / Direct Invocation
# -------------------------------------------------------------
async def _cli_main():
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else print("enter a valid query")
    )

    print("=" * 60)
    print(f"🚀 Running Notion Agent with Query: '{query}'")
    print("=" * 60)

    agent = NotionAgent()

    async for event in agent.astream(query):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        print(f"📍 Completed Node: [{node_name}] -> {node_output}")

    print("\n✅ Execution finished.")


if __name__ == "__main__":
    asyncio.run(_cli_main())

import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from notionagent.agent import NotionAgent

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("notionagent.server")

# Initialize FastAPI app
app = FastAPI(
    title="NotionAgent API",
    description="Production-grade AI Agent API for Notion Workspaces powered by LangGraph and MCP",
    version="0.1.0",
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared agent instance
agent = NotionAgent()


class ChatRequest(BaseModel):
    query: str = Field(..., description="User prompt or instruction for Notion workspace")
    thread_id: Optional[str] = Field("default_thread", description="Thread ID for conversation state persistence")


class ChatResponse(BaseModel):
    query: str
    final_response: Optional[str]
    is_valid_request: bool
    rejection_reason: Optional[str] = None
    has_error: bool = False
    error_message: Optional[str] = None
    steps_executed: int = 0
    plan: Optional[list] = None
    is_goal_satisfied: Optional[bool] = None
    reflection_count: Optional[int] = 0


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "NotionAgent API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Executes the NotionAgent workflow and returns the final state."""
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Received chat request: '{clean_query}' (thread_id: {request.thread_id})")

    try:
        final_state = await agent.ainvoke(clean_query, thread_id=request.thread_id)

        step_results = final_state.get("step_results") or []
        plan = final_state.get("plan") or []

        return ChatResponse(
            query=clean_query,
            final_response=final_state.get("final_response"),
            is_valid_request=final_state.get("is_valid_request", False),
            rejection_reason=final_state.get("rejection_reason"),
            has_error=final_state.get("has_error", False),
            error_message=final_state.get("error_message"),
            steps_executed=len(step_results),
            plan=plan,
            is_goal_satisfied=final_state.get("is_goal_satisfied"),
            reflection_count=final_state.get("reflection_count", 0),
        )
    except Exception as e:
        logger.exception(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal agent execution error: {str(e)}")


async def _sse_event_stream(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Streams node-by-node execution events as Server-Sent Events (SSE)."""
    try:
        async for event in agent.astream(query, thread_id=thread_id):
            node_name = list(event.keys())[0]
            node_data = event[node_name]

            event_payload = {
                "node": node_name,
                "data": node_data,
            }

            yield f"data: {json.dumps(event_payload)}\n\n"

        yield "data: {\"status\": \"completed\"}\n\n"
    except Exception as e:
        logger.exception(f"Error in SSE stream: {e}")
        error_payload = {"error": str(e)}
        yield f"data: {json.dumps(error_payload)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streams node-by-node execution events in real time using Server-Sent Events (SSE)."""
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Streaming chat request: '{clean_query}' (thread_id: {request.thread_id})")

    return StreamingResponse(
        _sse_event_stream(clean_query, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.py:app", host="0.0.0.0", port=8000, reload=True)

# NotionAgent 🤖

> **Autonomous, Multi-Step Notion Workspace Agent** powered by **LangGraph**, **Azure OpenAI**, and the official **Notion Model Context Protocol (MCP) Server**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture Analysis: Current vs. New](#architecture-analysis-current-vs-new)
  - [Architectural Flaws in the Initial Design](#architectural-flaws-in-the-initial-design)
  - [The New Architecture Paradigm](#the-new-architecture-paradigm)
- [Architecture Diagrams (Mermaid)](#architecture-diagrams-mermaid)
  - [1. End-to-End Pipeline & State Machine](#1-end-to-end-pipeline--state-machine)
  - [2. Component & Tool Infrastructure](#2-component--tool-infrastructure)
  - [3. Concrete Execution Sequence Diagram](#3-concrete-execution-sequence-diagram)
- [Node Responsibilities](#node-responsibilities)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Getting Started](#setup--getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Installation](#installation)
  - [Pre-flight Diagnostics](#pre-flight-diagnostics)
  - [Running the Application](#running-the-application)
    - [1. Running the FastAPI Server](#1-running-the-fastapi-server)
    - [2. Running via CLI](#2-running-via-cli)
    - [3. Programmatic Python Invocation](#3-programmatic-python-invocation)
  - [API Endpoints & Testing with cURL](#api-endpoints--testing-with-curl)
- [Implementation Roadmap](#implementation-roadmap)

---

## 📖 Overview

NotionAgent bridges the gap between natural language user requests and Notion's complex, block-hierarchical REST API. Instead of executing fragile single-turn prompt-and-call operations, NotionAgent uses an **evaluator-optimizer state machine** built on LangGraph that plans, retrieves context, executes tool calls through persistent MCP subprocesses, reflects on outcomes, and self-corrects upon errors.

---

## 🔍 Architecture Analysis: Current vs. New

### Architectural Flaws in the Initial Design

An in-depth analysis of the baseline agent reveals 5 core architectural limitations when compared to production systems like Notion AI:

| # | Flaw | Impact | Root Cause |
|---|---|---|---|
| **1** | **Context Blindness (Pre-Plan Blindness)** | The agent hallucinates IDs or duplicates headings. | Planner receives only raw query + empty memory; it has never read what blocks or pages already exist. |
| **2** | **Blind Reflection (Surface Validation)** | Agent claims success even when writes fail or nest incorrectly. | Reflector only checks JSON string outputs (truncated to 300 chars) instead of querying ground-truth Notion state. |
| **3** | **Monolithic Plan Execution** | If step 2 fails, step 1 has already altered Notion, but replanning triggers from scratch. | Executor iterates through all steps in a single batch without intermediate checkpoints. |
| **4** | **Ephemeral Workspace Memory** | Every request searches for the same pages repeatedly. | `StructuredWorkspaceContext` is stored solely in volatile request state with zero persistent caching. |
| **5** | **Prompt Tool Bloat** | High latency (TTFT) and tool selection errors. | All 24 Notion MCP tools and descriptions are dumped into every single prompt call regardless of intent. |

---

### The New Architecture Paradigm

The enhanced architecture addresses these challenges through five foundational design patterns:

1. **Read-Before-Write Pipeline (`ContextLoaderNode`)**:
   - Before the Planner generates an action plan, the target page or workspace hierarchy is inspected.
   - Converts Notion's Abstract Syntax Tree (AST) into a lean **PageOutline** (IDs, headings, containers, to-dos) injected directly into the planning prompt.

2. **Grounded Reflection (`GroundTruthVerifier`)**:
   - Post-execution, the Reflector issues targeted read calls (`API-get-block-children` or `API-retrieve-page-markdown`) to confirm that created blocks exist at the designated path.

3. **Step-by-Step Interleaved Execution**:
   - Replaces all-at-once loops with an incremental loop: `Plan -> Step N -> Reflect -> Step N+1`.
   - Allows fine-grained error isolation and local replanning without duplicating previously completed steps.

4. **Persistent Workspace Memory Store**:
   - Stores workspace schema mappings (`page_title -> page_id`, `database_title -> database_id`, and `root_entry_point`) in SQLite/Redis to avoid redundant search queries across user sessions.

5. **Intent-Filtered Tool Routing**:
   - Filters the 24 tools into specialized working sets (Read, Write/Block, Database) based on Guard classification, reducing prompt tokens by ~60%.

---

## 📊 Architecture Diagrams (Mermaid)

### 1. End-to-End Pipeline & State Machine

```mermaid
flowchart TD
    USER([👤 User Request]) --> API[⚡ FastAPI /api/chat & /api/chat/stream]
    API --> GUARD

    subgraph LangGraph["🔁 LangGraph State Machine"]
        direction TB

        GUARD["🛡️ Guard Node\n• Validates Notion intent\n• Classifies task type (Read/Write/DB)"]
        REJECT["🚫 Rejection Node\n• Polite fallback for out-of-scope requests"]

        CTXLOADER["🗺️ ContextLoader Node (NEW)\n• Reads persistent entity cache\n• Inspects target page block tree\n• Builds PageOutline (IDs + headings)\n• Injects into AgentState"]

        PLANNER["🧠 Planner Node\n• Plans with full page-tree awareness\n• Targets exact container block IDs\n• Avoids duplicate sections\n• Outputs structured Plan & Steps"]

        EXECUTOR["⚙️ Executor Node\n• Executes 1 step per cycle\n• Resolves placeholder & parent IDs\n• Invokes Notion MCP Registry\n• Catches transient errors with retries"]

        REFLECTOR["🪞 Reflector Node\n• Reads live Notion state to verify write\n• Checks: Did block appear under target?\n• Emits replan critique on failure"]

        SYNTH["📝 Synthesizer Node\n• Formats user-facing response\n• Emits Notion page links & changelog"]

        ERR["❌ Error Node\n• Circuit breaker on max retry exhaustion"]

        GUARD -- "Valid Request" --> CTXLOADER
        GUARD -- "Invalid Request" --> REJECT
        CTXLOADER --> PLANNER
        PLANNER --> EXECUTOR
        EXECUTOR --> REFLECTOR

        REFLECTOR -- "Step Failed (Retries left)" --> PLANNER
        REFLECTOR -- "Step OK (More steps remaining)" --> EXECUTOR
        REFLECTOR -- "All Steps Done / Goal Met" --> SYNTH
        EXECUTOR -- "Fatal Error" --> ERR
    end

    subgraph MEMORY["💾 Long-Term Storage"]
        STORE[("SQLite / Redis Cache\n• page_name ➔ page_id\n• db_name ➔ db_id\n• root_entry_point")]
    end

    CTXLOADER <--> MEMORY
    EXECUTOR -.->|Persist new IDs| MEMORY

    subgraph INFRA["🔧 MCP & Notion Layer"]
        REGISTRY["NotionToolRegistry\n(mcp_client.py)"]
        MCPSERVER["@notionhq/notion-mcp-server\n(npx stdio subprocess)"]
        NOTIONAPI["☁️ Notion Official REST API"]

        REGISTRY <--> MCPSERVER
        MCPSERVER <--> NOTIONAPI
    end

    EXECUTOR --> REGISTRY
    CTXLOADER --> REGISTRY
    REFLECTOR --> REGISTRY

    SYNTH --> OUT([📤 Final Response to Client])
    REJECT --> OUT
    ERR --> OUT
```

---

### 2. Component & Tool Infrastructure

```mermaid
graph LR
    subgraph ClientLayer["Application Layer"]
        FastAPI["FastAPI App (server.py)"]
        Agent["NotionAgent Runner (agent.py)"]
    end

    subgraph StateGraph["Agent Core (LangGraph)"]
        State["AgentState\n────────────────\n• user_query\n• page_structure\n• workspace_context\n• plan & step_results\n• reflection_count\n• final_response"]
    end

    subgraph Bridge["MCP Client & Registry (mcp_client.py)"]
        Registry["NotionToolRegistry Singleton\n────────────────────────\n• Lifespan session manager\n• JSON-RPC stdio client\n• Tool argument adapter\n• UUID normalizer\n• Prompt docs generator"]
    end

    subgraph MCP["Notion MCP Server"]
        Server["@notionhq/notion-mcp-server\n(npx subprocess)"]
    end

    subgraph NotionCloud["Notion Platform"]
        Notion["api.notion.com"]
    end

    FastAPI --> Agent
    Agent --> StateGraph
    StateGraph --> Bridge
    Bridge --> MCP
    MCP --> NotionCloud
```

---

### 3. Concrete Execution Sequence Diagram

The following sequence illustrates how the agent fulfills the query:  
> *"Add a task 'Submit quarterly report' under the 'Sprint 1' section of my Tasks page"*

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Server as FastAPI Server
    participant Ctx as ContextLoader
    participant Mem as Persistent Memory
    participant Plan as Planner Node
    participant Exec as Executor Node
    participant MCP as Notion MCP Server
    participant Notion as Notion REST API
    participant Refl as Reflector Node
    participant Synth as Synthesizer

    User->>Server: POST /api/chat {"query": "Add task 'Submit quarterly report' under 'Sprint 1' in Tasks"}
    Server->>Ctx: Invoke ContextLoader
    Ctx->>Mem: Lookup "tasks"
    Mem-->>Ctx: Found page_id: "3f7b67c2-..."

    Ctx->>MCP: API-get-block-children(block_id="3f7b67c2-...")
    MCP->>Notion: GET /v1/blocks/3f7b67c2-.../children
    Notion-->>MCP: [Heading2: "Sprint 1" (ID: "blk-901"), Todo: "Draft plan", ...]
    MCP-->>Ctx: Raw block objects
    Ctx->>Ctx: Parse outline: Section "Sprint 1" ➔ ID "blk-901"

    Ctx->>Plan: Ingest outline context into prompt
    Plan->>Plan: LLM selects target block "blk-901" (avoids root page append)
    Plan-->>Exec: Step 1: append_block(block_id="blk-901", content="Submit quarterly report")

    Exec->>MCP: API-patch-block-children(block_id="blk-901", children=[to_do])
    MCP->>Notion: PATCH /v1/blocks/blk-901/children
    Notion-->>MCP: 200 OK (created block "blk-902")
    MCP-->>Exec: Block created JSON

    Exec->>Refl: Hand off step result
    Refl->>MCP: API-get-block-children(block_id="blk-901") [Grounded Verification]
    MCP->>Notion: GET /v1/blocks/blk-901/children
    Notion-->>MCP: Block list containing "blk-902"
    MCP-->>Refl: Verified presence of task
    Refl-->>Synth: is_goal_satisfied = True

    Synth-->>Server: Render markdown confirmation with page reference
    Server-->>User: "✅ Successfully added 'Submit quarterly report' under 'Sprint 1' on your Tasks page."
```

---

## 🧩 Node Responsibilities

| Node | File | Primary Responsibility |
|---|---|---|
| **Guard** | [guard.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/guard.py) | Filters irrelevant or malicious prompts; classifies request intent. |
| **Rejection** | [rejection.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/rejection.py) | Generates courteous explanations when requests fall outside workspace capabilities. |
| **ContextLoader** | `context_loader.py` *(New)* | Discovers existing page structure, extracts headings, and loads workspace memory. |
| **Planner** | [planner.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/planner.py) | Generates structured sequential plans with explicit target block IDs and preconditions. |
| **Executor** | [executor.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/executor.py) | Adapts inputs, normalizes UUIDs, executes MCP tools with exponential backoff retries. |
| **Reflector** | [reflector.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/reflector.py) | Inspects actual workspace updates; loops back with corrective critiques if unsatisfied. |
| **Synthesizer** | [synthesizer.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/synthesizer.py) | Produces friendly, formatted markdown summaries of actions completed. |
| **Error** | [error.py](file:///home/sivakumar/Documents/ai-eng/notionagent/notionagent/nodes/error.py) | Handles non-recoverable execution failures gracefully. |

---

## 🛠️ Tech Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph evaluator-optimizer loop)
- **LLM Client**: [LangChain Azure OpenAI](https://github.com/langchain-ai/langchain) (`AzureChatOpenAI`)
- **Protocol**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) (`mcp>=2.3.0`)
- **Notion Server**: `@notionhq/notion-mcp-server` running over persistent `stdio`
- **Web API**: [FastAPI](https://fastapi.tiangolo.com/) with asynchronous lifespan handlers
- **Package & Environment Management**: [uv](https://github.com/astral-sh/uv)

---

## 📁 Project Structure

```
notionagent/
├── server.py                   # FastAPI server with persistent MCP lifespan
├── pyproject.toml              # Project dependencies and packaging settings
├── README.md                   # System documentation and architecture guide
├── notionagent/
│   ├── __init__.py             # Package root
│   ├── agent.py                # High-level NotionAgent invocation API
│   ├── graph.py                # LangGraph StateGraph assembly and routers
│   ├── state.py                # AgentState, InputState, and OutputState schemas
│   ├── mcp_client.py           # NotionToolRegistry MCP client & argument adapter
│   ├── mcp_registry.py         # Module compatibility aliases
│   ├── nodes/
│   │   ├── guard.py            # Guardrail node
│   │   ├── rejection.py        # Request rejection node
│   │   ├── planner.py          # Structured plan generator
│   │   ├── executor.py         # Step executor with error recovery
│   │   ├── reflector.py        # Goal satisfaction evaluator
│   │   ├── synthesizer.py      # Final response synthesizer
│   │   └── error.py            # Terminal error handler
│   └── prompts/
│       ├── planner.py          # Planner system prompts
│       ├── reflector.py        # Reflector evaluation prompts
│       ├── guard.py            # Guard system prompt
│       └── synthesizer.py      # Synthesizer prompt
```

---

## 🚀 Setup & Getting Started

### Prerequisites
- Python `>= 3.12`
- Node.js `>= 18` (required by `npx @notionhq/notion-mcp-server`)
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Notion Internal Integration Token ([Notion Developers](https://www.notion.so/my-integrations))

### Installation

Clone the repository and install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/sivakumar232/Notionagent.git
cd Notionagent

# Install dependencies into virtual environment
uv sync
```

---

### Environment Variables
Create a `.env` file in the root directory:

```env
# Notion Credentials (supports NOTION_TOKEN or NOTION_API_KEY)
NOTION_TOKEN=secret_your_notion_integration_token_here
NOTION_PARENT_PAGE_ID=your_entry_point_page_uuid_here

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-10-21

# PostgreSQL State Checkpointer (Optional - falls back to MemorySaver if unavailable)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/notionagent

# LangSmith Observability & Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=notionagent
```

---

### Pre-flight Diagnostics

Verify your Azure OpenAI configuration before running the agent:

```bash
uv run python test_azure.py
```

---

### Running the Application

You can run NotionAgent as a FastAPI service, directly from your terminal CLI, or import it into your own Python codebase.

#### 1. Running the FastAPI Server

Start the API server with live reloading:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Or run via `server.py`:

```bash
uv run python server.py
```

- **Interactive API Documentation (Swagger UI)**: Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
- **Alternative Docs (ReDoc)**: Open [http://localhost:8000/redoc](http://localhost:8000/redoc).

#### 2. Running via CLI

Run queries directly from the command line without starting the web server:

```bash
uv run python -m notionagent.agent "Create a weekly sprint meeting page under my Projects page"
```

#### 3. Programmatic Python Invocation

Integrate NotionAgent directly into your Python scripts or workflows:

```python
import asyncio
from notionagent.agent import NotionAgent
from notionagent.mcp_client import registry

async def main():
    agent = NotionAgent()

    # The Notion MCP server must run within its lifespan context
    async with registry.lifespan_session():
        # Synchronous/Batch completion:
        result = await agent.ainvoke(
            query="Add a todo item 'Review pull requests' to my Tasks page",
            thread_id="session_42",
        )
        print("Final Response:", result.get("final_response"))

        # Or stream node-by-node execution:
        async for step in agent.astream("List recent updates in workspace", thread_id="session_42"):
            node_name = list(step.keys())[0]
            print(f"Executed node: {node_name}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### API Endpoints & Testing with cURL

#### Health Check
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{"status": "healthy", "service": "NotionAgent API"}
```

#### Execute Agent Request (Synchronous)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create a todo list under my project page with items for sprint planning",
    "thread_id": "thread_abc123"
  }'
```

#### Stream Execution Events (Server-Sent Events)
Stream live node execution status, planning output, and intermediate results in real time:

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Add a meeting note under my Team Notes page",
    "thread_id": "thread_abc123"
  }'
```

---

## 🛣️ Implementation Roadmap

- [x] LangGraph multi-node state machine with reflection loop
- [x] Persistent Notion MCP stdio client via FastAPI lifespan
- [x] Dynamic tool discovery and schema conversion
- [x] Robust argument normalizer (UUID formatting, parent wrapper adaptation)
- [ ] **Phase 1**: Add `ContextLoaderNode` with live Notion block-tree parsing
- [ ] **Phase 2**: Implement Grounded Verification in `ReflectorNode` via read-after-write
- [x] **Phase 3**: Migrate Executor to single-step execution with intermediate reflection and duplicate-safe replanning
- [ ] **Phase 4**: Add SQLite persistent workspace memory across agent runs
- [ ] **Phase 5**: Intent-based tool filtering to cut prompt token consumption

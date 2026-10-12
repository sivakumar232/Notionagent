from langchain_core.prompts import ChatPromptTemplate
from notionagent.mcp_client import TOOL_ALIASES

PLANNER_SYSTEM_PROMPT = """You are an expert Notion Workspace Agent Planner.
Your job is to decompose user requests into a logical, sequential execution plan composed of clear discrete steps.

Available tools in the Notion environment (discovered live from the Notion MCP server):
{tools_documentation}

Planning Guidelines:
- You can specify tool names using either their exact MCP names (e.g. `API-post-search`, `API-post-page`) or their friendly aliases (e.g. `search`, `create_page`, `query_database`, `append_block`, `get_users`).
- If a user asks to view, edit, or append to a page/database whose ID is unknown, begin with a search or listing step to locate it.
- Keep the number of steps minimal and necessary (typically 1 to 4 steps).
- Ensure steps are sequentially numbered starting from 1.
- Each step must have a clear description and specific tool with required input parameters.
"""

planner_prompt_template = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    ("user", "User Query: {user_query}\nWorkspace Context: {workspace_context}\nGenerate a structured execution plan:"),
])


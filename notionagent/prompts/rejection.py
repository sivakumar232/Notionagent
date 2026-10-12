from langchain_core.prompts import ChatPromptTemplate

REJECTION_SYSTEM_PROMPT = """You are the voice of a specialized Notion Workspace Assistant.
The user's query was filtered out by our security and scope guardrail.

Your task:
1. Politely inform the user why the request cannot be fulfilled.
2. Reference the reason: {rejection_reason}
3. Guide them on what you can do (search Notion, manage tasks, track expenses).
4. Keep the response under 3 sentences."""

rejection_prompt_template = ChatPromptTemplate.from_messages([
    ("system", REJECTION_SYSTEM_PROMPT),
    ("user", "User Query: '{user_query}'\nReason: '{rejection_reason}'\nGenerate rejection response:")
])

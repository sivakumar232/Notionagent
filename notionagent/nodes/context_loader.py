import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from notionagent.db import entity_repo
from notionagent.mcp_client import notion_mcp_client
from notionagent.state import AgentState
from notionagent.utils.outline_parser import parse_blocks_to_outline

logger = logging.getLogger("notionagent.context_loader")

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}",
    re.IGNORECASE,
)

STOP_WORDS = {
    "add", "create", "make", "insert", "append", "new", "under", "in", "to", "on",
    "the", "a", "an", "page", "section", "block", "inside", "my", "our", "with",
}


def _extract_candidate_title(query: str) -> Optional[str]:
    """Extracts quoted page name or candidate keyword from user query."""
    # 1. Match quotes (e.g. in "todolist" or under 'Project Alpha')
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
    if quoted:
        return quoted[0].strip()

    # 2. Match patterns like "in <name> page" or "under <name> page" or "to <name>"
    match = re.search(r"(?:in|under|to|on)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+page|\s+database|$)", query, re.I)
    if match:
        candidate = match.group(1).strip()
        words = [w for w in candidate.split() if w.lower() not in STOP_WORDS]
        if words:
            return " ".join(words)

    return None


async def context_loader_node(state: AgentState) -> Dict[str, Any]:
    """2. Context Loader Node: Inspects Notion page structure before planning.

    Discovers target page ID, fetches its block hierarchy, and formats an
    outline so the Planner plans with full structural awareness.
    Uses PostgreSQL to persistently cache entities and outlines.
    """
    user_query = str(state.get("user_query") or "").strip()
    workspace_context = dict(state.get("workspace_context") or {})
    known_pages = dict(workspace_context.get("pages") or {})

    # Load persistent entities from PostgreSQL into memory context
    try:
        db_entities = await entity_repo.get_all_entities()
        if db_entities:
            known_pages.update(db_entities)
    except Exception as e:
        logger.warning(f"Could not load entities from PostgreSQL: {e}")

    target_id: Optional[str] = state.get("target_page_id")
    target_title: Optional[str] = state.get("page_title")

    # 1. Check if UUID is explicitly mentioned in the user query
    if not target_id:
        uuid_match = UUID_PATTERN.search(user_query)
        if uuid_match:
            target_id = uuid_match.group(0)
            logger.info(f"Target ID found via query UUID match: {target_id}")

    # 2. Check if a known page title matches in workspace_context or PostgreSQL
    if not target_id and known_pages:
        query_lower = user_query.lower()
        for title_key, page_id in known_pages.items():
            if title_key in query_lower:
                target_id = page_id
                target_title = title_key
                logger.info(f"Target page matched in persistent context: '{title_key}' -> {target_id}")
                break

    # 3. Check persistent DB specifically for candidate title
    candidate_title = _extract_candidate_title(user_query)
    if not target_id and candidate_title:
        cached_id = await entity_repo.get_entity_id_by_name(candidate_title)
        if cached_id:
            target_id = cached_id
            target_title = candidate_title
            logger.info(f"Target page matched via PostgreSQL query: '{candidate_title}' -> {target_id}")

    # 4. Search Notion API if candidate still unresolved
    if not target_id and candidate_title:
        try:
            logger.info(f"Searching Notion for candidate target title: '{candidate_title}'")
            search_output = await notion_mcp_client.call_tool(
                "API-post-search",
                {"query": candidate_title, "page_size": 3},
            )
            data = json.loads(search_output) if isinstance(search_output, str) else search_output
            results = data.get("results", []) if isinstance(data, dict) else []

            for item in results:
                if item.get("object") == "page":
                    target_id = item.get("id")
                    # Extract page title
                    props = item.get("properties", {})
                    for prop in props.values():
                        if isinstance(prop, dict) and prop.get("type") == "title":
                            title_list = prop.get("title", [])
                            if title_list and isinstance(title_list, list):
                                target_title = title_list[0].get("plain_text", "").strip()
                    logger.info(f"Discovered target page from search: '{target_title}' -> {target_id}")

                    # Save discovered entity persistently into PostgreSQL
                    if target_title and target_id:
                        await entity_repo.upsert_entity(target_title, target_id, "page")
                    break
        except Exception as e:
            logger.warning(f"Context search failed for candidate '{candidate_title}': {e}")

    # 5. Fallback to entry point page ID from environment if target is still unresolved
    if not target_id:
        fallback_parent = os.getenv("NOTION_PARENT_PAGE_ID")
        if fallback_parent and UUID_PATTERN.search(fallback_parent):
            target_id = fallback_parent
            logger.info(f"Falling back to NOTION_PARENT_PAGE_ID entrypoint: {target_id}")

    # 6. Retrieve block hierarchy for the target page (from PostgreSQL cache or live Notion)
    outline_text = "(No target page identified prior to planning; planner will search or create as needed)"
    raw_blocks: List[Dict[str, Any]] = []

    if target_id:
        # Check PostgreSQL cache first (15-minute TTL)
        cached_outline = await entity_repo.get_cached_outline(target_id, max_age_seconds=900)
        if cached_outline:
            outline_text = cached_outline
            logger.info(f"Loaded page outline for {target_id} from PostgreSQL cache")
        else:
            try:
                logger.info(f"Fetching block hierarchy live from Notion for: {target_id}")
                blocks_output = await notion_mcp_client.call_tool(
                    "API-get-block-children",
                    {"block_id": target_id},
                )
                b_data = json.loads(blocks_output) if isinstance(blocks_output, str) else blocks_output
                if isinstance(b_data, dict):
                    raw_blocks = b_data.get("results", [])
                elif isinstance(b_data, list):
                    raw_blocks = b_data

                outline_text = parse_blocks_to_outline(raw_blocks)
                logger.info(f"Generated outline for page {target_id} ({len(raw_blocks)} blocks)")

                # Persist fresh outline into PostgreSQL
                await entity_repo.upsert_page_outline(
                    target_id, target_title, outline_text, raw_blocks
                )

            except Exception as e:
                logger.warning(f"Failed to fetch block hierarchy for target {target_id}: {e}")
                outline_text = f"(Target page {target_id} identified, but could not retrieve blocks: {e})"

    # 7. Update workspace_context with discovered target
    if target_id:
        workspace_context["last_page_id"] = target_id
        if target_title:
            known_pages[target_title.lower()] = target_id
        workspace_context["pages"] = known_pages

    return {
        "target_page_id": target_id,
        "page_title": target_title,
        "page_outline": outline_text,
        "raw_blocks": raw_blocks,
        "workspace_context": workspace_context,
    }

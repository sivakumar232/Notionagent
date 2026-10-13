import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from notionagent.db.connection import db_manager

logger = logging.getLogger("notionagent.db.repository")


class WorkspaceEntityRepository:
    """Repository for workspace entity name-to-ID mappings and cached page outlines."""

    async def get_entity_id_by_name(self, name: str) -> Optional[str]:
        """Looks up a Notion entity ID by case-insensitive name."""
        if not name or not name.strip():
            return None

        clean_name = name.strip().lower()
        try:
            async with db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT entity_id FROM workspace_entities WHERE LOWER(name) = %s LIMIT 1",
                        (clean_name,),
                    )
                    row = await cur.fetchone()
                    if row:
                        return row["entity_id"]
        except Exception as e:
            logger.warning(f"Error fetching entity by name '{clean_name}': {e}")
        return None

    async def get_all_entities(self) -> Dict[str, str]:
        """Returns a dictionary mapping all lowercase entity names to Notion IDs."""
        mapping: Dict[str, str] = {}
        try:
            async with db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT name, entity_id FROM workspace_entities")
                    rows = await cur.fetchall()
                    for r in rows:
                        mapping[r["name"].lower()] = r["entity_id"]
        except Exception as e:
            logger.warning(f"Error fetching all entities: {e}")
        return mapping

    async def upsert_entity(
        self,
        name: str,
        entity_id: str,
        entity_type: str = "page",
        parent_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inserts or updates a workspace entity in PostgreSQL."""
        if not name or not entity_id:
            return

        clean_name = name.strip()
        clean_id = entity_id.strip()
        props_json = json.dumps(properties or {})

        try:
            async with db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO workspace_entities (name, entity_id, entity_type, parent_id, properties, last_synced_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                        ON CONFLICT (entity_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            entity_type = EXCLUDED.entity_type,
                            parent_id = COALESCE(EXCLUDED.parent_id, workspace_entities.parent_id),
                            properties = EXCLUDED.properties,
                            last_synced_at = NOW()
                        """,
                        (clean_name, clean_id, entity_type, parent_id, props_json),
                    )
                await conn.commit()
            logger.debug(f"Upserted entity into PostgreSQL: '{clean_name}' -> {clean_id}")
        except Exception as e:
            logger.warning(f"Error upserting entity '{clean_name}' ({clean_id}): {e}")

    async def get_cached_outline(
        self,
        page_id: str,
        max_age_seconds: int = 1800,
    ) -> Optional[str]:
        """Retrieves cached page outline if it exists and is younger than max_age_seconds."""
        if not page_id:
            return None

        try:
            async with db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT outline_markdown, updated_at FROM page_structures WHERE page_id = %s",
                        (page_id,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None

                    updated_at = row["updated_at"]
                    if updated_at:
                        now = datetime.now(timezone.utc)
                        age = (now - updated_at).total_seconds()
                        if age > max_age_seconds:
                            logger.info(f"Cached outline for page {page_id} expired (age: {age:.0f}s > {max_age_seconds}s)")
                            return None

                    return row["outline_markdown"]
        except Exception as e:
            logger.warning(f"Error fetching cached outline for page {page_id}: {e}")
        return None

    async def upsert_page_outline(
        self,
        page_id: str,
        title: Optional[str],
        outline_markdown: str,
        raw_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Stores or updates the structural outline of a page in PostgreSQL."""
        if not page_id or not outline_markdown:
            return

        blocks_json = json.dumps(raw_blocks or [])
        block_count = len(raw_blocks or [])

        try:
            async with db_manager.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO page_structures (page_id, title, outline_markdown, block_count, raw_blocks, updated_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                        ON CONFLICT (page_id) DO UPDATE SET
                            title = COALESCE(EXCLUDED.title, page_structures.title),
                            outline_markdown = EXCLUDED.outline_markdown,
                            block_count = EXCLUDED.block_count,
                            raw_blocks = EXCLUDED.raw_blocks,
                            updated_at = NOW()
                        """,
                        (page_id, title, outline_markdown, block_count, blocks_json),
                    )
                await conn.commit()
            logger.debug(f"Saved outline for page {page_id} to PostgreSQL ({block_count} blocks)")
        except Exception as e:
            logger.warning(f"Error saving outline for page {page_id}: {e}")


entity_repo = WorkspaceEntityRepository()

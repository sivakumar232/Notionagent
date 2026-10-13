import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

load_dotenv()
logger = logging.getLogger("notionagent.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspace_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    entity_id VARCHAR(64) NOT NULL UNIQUE,
    entity_type VARCHAR(32) NOT NULL DEFAULT 'page',
    parent_id VARCHAR(64),
    properties JSONB DEFAULT '{}'::jsonb,
    last_synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_lower_name ON workspace_entities(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_entities_id ON workspace_entities(entity_id);

CREATE TABLE IF NOT EXISTS page_structures (
    page_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(255),
    outline_markdown TEXT NOT NULL,
    block_count INT DEFAULT 0,
    raw_blocks JSONB DEFAULT '[]'::jsonb,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""


class DatabaseManager:
    """Production PostgreSQL Connection Pool & Schema Manager."""

    def __init__(self, db_url: Optional[str] = None):
        self._db_url = (
            db_url
            or os.getenv("DATABASE_URL")
            or "postgresql://postgres:postgres@localhost:5432/notionagent"
        )
        self._pool: Optional[AsyncConnectionPool] = None

    @property
    def db_url(self) -> str:
        return self._db_url

    async def initialize(self) -> None:
        """Initializes connection pool and applies table migrations."""
        if self._pool is not None:
            return

        logger.info("Initializing PostgreSQL AsyncConnectionPool...")
        # Configure pool with dict_row factory for convenience
        self._pool = AsyncConnectionPool(
            conninfo=self._db_url,
            min_size=2,
            max_size=10,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        await self._pool.open()

        # Run schema migrations
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(SCHEMA_SQL)
            await conn.commit()

        logger.info("PostgreSQL connected and schema verified successfully.")

    async def close(self) -> None:
        """Closes the connection pool cleanly."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL AsyncConnectionPool closed.")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator:
        """Yields an active database connection from the pool."""
        if self._pool is None:
            await self.initialize()
        assert self._pool is not None
        async with self._pool.connection() as conn:
            yield conn


db_manager = DatabaseManager()

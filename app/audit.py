"""MongoDB audit log and canvas content persistence.
Best-effort: never blocks the user, never raises."""

from __future__ import annotations

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient

from app.config import logger, settings

# Set by main.py during startup.
mongo_client: AsyncIOMotorClient | None = None


def set_mongo_client(client: AsyncIOMotorClient | None) -> None:
    global mongo_client
    mongo_client = client


def _coll():
    if not mongo_client:
        return None
    return mongo_client[settings.mongodb_database]["chatops_audit"]


def _canvas_coll():
    """Collection for persisting canvas content across restarts."""
    if not mongo_client:
        return None
    return mongo_client[settings.mongodb_database]["chatops_canvas"]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


async def audit_log(
    slack_user: str, slack_email: str, tool_calls: list[dict], result: str
) -> None:
    coll = _coll()
    if coll is None:
        return
    try:
        await coll.insert_one({
            "slack_user": slack_user,
            "slack_email": slack_email,
            "tool_calls": tool_calls,
            "result_preview": result[:500],
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning("Audit log write failed: %s", e)


# ---------------------------------------------------------------------------
# Canvas content persistence
# ---------------------------------------------------------------------------


async def save_canvas_content(channel_id: str, canvas_id: str, content: str) -> None:
    """Save canvas content to MongoDB. Best-effort — never raises."""
    coll = _canvas_coll()
    if coll is None:
        return
    try:
        await coll.update_one(
            {"channel_id": channel_id, "canvas_id": canvas_id},
            {
                "$set": {
                    "content": content,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
        logger.info(
            "Canvas content saved to DB: channel=%s canvas=%s (%d chars)",
            channel_id, canvas_id, len(content),
        )
    except Exception as e:
        logger.warning("Canvas DB write failed: %s", e)


async def load_canvas_content(channel_id: str, canvas_id: str) -> str:
    """Load canvas content from MongoDB. Returns empty string if not found."""
    coll = _canvas_coll()
    if coll is None:
        return ""
    try:
        doc = await coll.find_one(
            {"channel_id": channel_id, "canvas_id": canvas_id}
        )
        if doc and doc.get("content"):
            logger.info(
                "Canvas content loaded from DB: channel=%s canvas=%s (%d chars)",
                channel_id, canvas_id, len(doc["content"]),
            )
            return doc["content"]
        return ""
    except Exception as e:
        logger.warning("Canvas DB read failed: %s", e)
        return ""


async def load_all_canvas_content() -> dict[str, dict[str, str]]:
    """Load all canvas content from MongoDB at startup.
    Returns: {channel_id: {canvas_id: content}}"""
    coll = _canvas_coll()
    if coll is None:
        return {}
    try:
        result: dict[str, dict[str, str]] = {}
        async for doc in coll.find({}):
            channel_id = doc.get("channel_id", "")
            canvas_id = doc.get("canvas_id", "")
            content = doc.get("content", "")
            if channel_id and canvas_id and content:
                result.setdefault(channel_id, {})[canvas_id] = content
        logger.info(
            "Loaded canvas content from DB for %d channel(s)", len(result)
        )
        return result
    except Exception as e:
        logger.warning("Canvas DB load all failed: %s", e)
        return {}

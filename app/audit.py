"""MongoDB audit log. Best-effort: never blocks the user, never raises."""

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

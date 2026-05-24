"""FastAPI entrypoint. Wires startup/shutdown, mounts the Slack request handler,
and exposes /health. Run with `python -m app` or `uvicorn app.main:api`."""
from __future__ import annotations
import asyncio
from fastapi import FastAPI, Request
from motor.motor_asyncio import AsyncIOMotorClient
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from app import audit
from app.config import logger, settings
from app.instructions import (
    load_instructions,
    preload_canvas_from_db,
    refresh_all_canvases,
)
from app.plane import mcp, refresh_plane_members, refresh_plane_states
from app.slack_app import slack_app
from app.standup import standup_loop
# Importing handlers is what registers the Slack listeners on slack_app.
# Keep this import for its side effects.
from app import handlers  # noqa: F401

api = FastAPI()
slack_handler = AsyncSlackRequestHandler(slack_app)
_background_tasks: list[asyncio.Task] = []
CANVAS_POLL_INTERVAL_SECONDS = 3600  # 1 hour


async def canvas_poll_loop() -> None:
    """Background task: re-fetch all canvas contents every 1 hour."""
    while True:
        await asyncio.sleep(CANVAS_POLL_INTERVAL_SECONDS)
        try:
            await refresh_all_canvases(settings.slack_bot_token)
            logger.info("Canvas content auto-refreshed (polling)")
        except Exception as e:
            logger.warning("Canvas poll failed: %s", e)


@api.post("/slack/events")
async def slack_events(request: Request):
    return await slack_handler.handle(request)


@api.get("/health")
async def health():
    return {"status": "ok", "mcp_servers": list(mcp.sessions.keys())}


@api.on_event("startup")
async def on_startup() -> None:
    if settings.mongodb_uri:
        client = AsyncIOMotorClient(settings.mongodb_uri)
        try:
            await client.admin.command("ping")
            audit.set_mongo_client(client)
            logger.info("MongoDB connected for audit log and canvas persistence")
        except Exception as e:
            logger.warning("MongoDB ping failed; audit log and canvas persistence disabled: %s", e)

    await mcp.start()
    await refresh_plane_members()
    await refresh_plane_states()
    load_instructions()

    # Step 1: Load canvas content from MongoDB instantly (no Slack API call needed)
    try:
        await preload_canvas_from_db()
        logger.info("Canvas content preloaded from MongoDB")
    except Exception as e:
        logger.warning("Canvas preload from DB failed: %s", e)

    # Step 2: Refresh from Slack API in background (updates DB + memory)
    try:
        await refresh_all_canvases(settings.slack_bot_token)
        logger.info("Canvas content refreshed from Slack API at startup")
    except Exception as e:
        logger.warning("Canvas refresh from Slack API failed: %s", e)

    # Start background polling every 1 hour
    _background_tasks.append(asyncio.create_task(canvas_poll_loop()))
    logger.info(
        "Canvas polling started (interval: %ds)", CANVAS_POLL_INTERVAL_SECONDS
    )

    if 0 <= settings.standup_hour_utc <= 23:
        _background_tasks.append(asyncio.create_task(standup_loop()))


@api.on_event("shutdown")
async def on_shutdown() -> None:
    for t in _background_tasks:
        t.cancel()
    await mcp.stop()
    if audit.mongo_client:
        audit.mongo_client.close()


def main() -> None:
    import uvicorn
    uvicorn.run(
        "app.main:api",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

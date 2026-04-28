"""RemoteStar ChatOps bot.

A Slack bot that lets the team manage Plane tickets through natural language.
Built on the official Plane MCP server. An LLM picks the right MCP tool from
the user's text. Designed so adding more MCP servers (GitHub, etc.) later is
just registering them in MCP_SERVERS, no other code changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    slack_bot_token: str
    slack_signing_secret: str

    openai_api_key: str

    plane_api_key: str
    plane_workspace_slug: str = "remotestar"
    plane_base_url: str = "https://plane.remotestar.io/api"
    plane_project_recruiter: str
    plane_project_candidate: str

    mongodb_uri: str | None = None
    mongodb_database: str = "remotestar_candidate"

    allowed_channel_ids: str = ""
    port: int = 9001
    log_level: str = "INFO"

    @property
    def allowed_channels(self) -> set[str]:
        return {c.strip() for c in self.allowed_channel_ids.split(",") if c.strip()}


settings = Settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger("chatops")


# ---------------------------------------------------------------------------
# MCP server registry — add more services here later
# ---------------------------------------------------------------------------

MCP_SERVERS: dict[str, StdioServerParameters] = {
    "plane": StdioServerParameters(
        command="uvx",
        args=["plane-mcp-server", "stdio"],
        env={
            "PLANE_API_KEY": settings.plane_api_key,
            "PLANE_WORKSPACE_SLUG": settings.plane_workspace_slug,
            "PLANE_BASE_URL": settings.plane_base_url,
            "PATH": os.environ.get("PATH", ""),
        },
    ),
    # Future: "github": StdioServerParameters(...)
}


class MCPManager:
    """Spawns and holds MCP subprocesses, exposing a flat tool registry."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tool_index: dict[str, tuple[str, Any]] = {}  # prefixed_name -> (server, tool)

    async def start(self) -> None:
        for name, params in MCP_SERVERS.items():
            try:
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session

                tools = (await session.list_tools()).tools
                for tool in tools:
                    prefixed = f"{name}__{tool.name}"
                    self._tool_index[prefixed] = (name, tool)
                logger.info("MCP server %r ready with %d tools", name, len(tools))
            except Exception as e:
                logger.error("Failed to start MCP server %r: %s", name, e, exc_info=True)

    async def stop(self) -> None:
        await self._stack.aclose()

    def openai_tools(self) -> list[dict]:
        """Convert MCP tools to OpenAI tools schema."""
        result = []
        for prefixed, (_, tool) in self._tool_index.items():
            result.append({
                "type": "function",
                "function": {
                    "name": prefixed,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return result

    async def call(self, prefixed_name: str, args: dict) -> str:
        if prefixed_name not in self._tool_index:
            return json.dumps({"error": f"unknown tool: {prefixed_name}"})
        server, tool = self._tool_index[prefixed_name]
        session = self._sessions.get(server)
        if not session:
            return json.dumps({"error": f"MCP server not running: {server}"})
        try:
            result = await session.call_tool(tool.name, args)
            content = result.content
            if isinstance(content, list):
                return "\n".join(getattr(c, "text", str(c)) for c in content)
            return str(content)
        except Exception as e:
            logger.error("Tool call failed: %s.%s — %s", server, tool.name, e, exc_info=True)
            return json.dumps({"error": str(e)})


mcp = MCPManager()


# ---------------------------------------------------------------------------
# Audit log (MongoDB) — best effort, never blocks the user
# ---------------------------------------------------------------------------

mongo_client: AsyncIOMotorClient | None = None


def get_audit_collection():
    if not mongo_client:
        return None
    return mongo_client[settings.mongodb_database]["chatops_audit"]


async def audit_log(slack_user: str, slack_email: str, tool_calls: list[dict], result: str) -> None:
    coll = get_audit_collection()
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
# Agent: LLM + MCP tool-calling loop
# ---------------------------------------------------------------------------

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


SYSTEM_PROMPT = f"""You are RemoteStar's ChatOps assistant in Slack. You help the team manage Plane tickets through natural language.

## Workspace context
- Plane workspace slug: {settings.plane_workspace_slug}
- Two projects available:
  - **CANDIDATE** (id: {settings.plane_project_candidate}) — for the candidate-facing app, profiles, jobs, interviews, matching
  - **RECRUITER** (id: {settings.plane_project_recruiter}) — for the recruiter dashboard, hiring flows, ATS integration

## How to pick the project
Try to infer from keywords:
- candidate, profile, signup, interview, jobs, matching, resume → CANDIDATE
- recruiter, hiring, ATS, dashboard, scraper, talent → RECRUITER
- Both / cross-cutting / unclear → ASK the user "Which project: CANDIDATE or RECRUITER?" before creating
- If the user explicitly says a project, use that

## Attribution
Every Plane issue you create or update via `create_issue`-style tools must include this footer in the description:

```
---
Created via ChatOps by {{user_email}} at {{timestamp_iso}}
```

Replace user_email and timestamp_iso with the actual values from the request context.

## Tool naming
Tools are prefixed with `<server>__<tool>`. For Plane tools, use the `plane__*` names exactly.

## User assistance
- Be concise. Reply in Slack-friendly markdown (no headings, just text + links).
- After creating an issue, return the issue URL so the user can click it.
- If a tool call fails, explain the error in plain English and suggest a fix.
- For ambiguous requests, ASK rather than assuming.
"""


async def agent_loop(history: list[dict], user_email: str, user_slack_id: str) -> str:
    """history is a list of {role, content} dicts. Last item is the current user request."""
    now_iso = datetime.now(timezone.utc).isoformat()
    system = SYSTEM_PROMPT + f"\n\n## Current request\n- User email: {user_email}\n- User Slack ID: {user_slack_id}\n- Timestamp: {now_iso}\n"

    messages: list[dict] = [
        {"role": "system", "content": system},
        *history,
    ]

    tools = mcp.openai_tools()
    if not tools:
        return "I'm not connected to any backends right now. Try again in a moment."

    tool_call_log: list[dict] = []
    max_iterations = 8

    for _ in range(max_iterations):
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            final = msg.content or "Done."
            await audit_log(user_slack_id, user_email, tool_call_log, final)
            return final

        # Append assistant's tool-calling message
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            logger.info("Tool call: %s args=%s", tc.function.name, args)
            result = await mcp.call(tc.function.name, args)
            tool_call_log.append({"name": tc.function.name, "args": args, "result_preview": result[:200]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    fallback = "I tried but couldn't complete the request in a reasonable number of steps. Try rephrasing?"
    await audit_log(user_slack_id, user_email, tool_call_log, fallback)
    return fallback


# ---------------------------------------------------------------------------
# Slack handlers
# ---------------------------------------------------------------------------

slack_app = AsyncApp(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    process_before_response=True,
)


def is_allowed_channel(channel_id: str) -> bool:
    if not settings.allowed_channels:
        return True  # if not configured, allow everywhere
    return channel_id in settings.allowed_channels


async def resolve_user_email(client, user_id: str) -> str:
    try:
        info = await client.users_info(user=user_id)
        return info["user"]["profile"].get("email") or "unknown@remotestar.io"
    except Exception as e:
        logger.warning("Failed to resolve user email for %s: %s", user_id, e)
        return "unknown@remotestar.io"


def strip_bot_mention(text: str) -> str:
    """Remove leading <@U123> mention from app_mention text."""
    if not text:
        return ""
    # Strip all leading <@...> mentions (sometimes there are multiple)
    while True:
        text = text.strip()
        if text.startswith("<@") and ">" in text:
            text = text.split(">", 1)[1]
        else:
            break
    return text.strip()


_bot_user_id_cache: str | None = None


async def get_bot_user_id(client) -> str:
    global _bot_user_id_cache
    if _bot_user_id_cache:
        return _bot_user_id_cache
    try:
        auth = await client.auth_test()
        _bot_user_id_cache = auth["user_id"]
        return _bot_user_id_cache
    except Exception as e:
        logger.warning("Failed to resolve bot user ID: %s", e)
        return ""


async def fetch_thread_history(client, channel: str, thread_ts: str, limit: int = 30) -> list[dict]:
    """Fetch messages in a thread and convert to LLM message format."""
    try:
        result = await client.conversations_replies(channel=channel, ts=thread_ts, limit=limit)
        bot_uid = await get_bot_user_id(client)
        history: list[dict] = []
        for msg in result.get("messages", []):
            text = strip_bot_mention(msg.get("text", "") or "")
            if not text:
                continue
            is_bot = msg.get("bot_id") or msg.get("user") == bot_uid
            role = "assistant" if is_bot else "user"
            history.append({"role": role, "content": text})
        return history
    except Exception as e:
        logger.warning("Failed to fetch thread history for %s: %s", thread_ts, e)
        return []


# Slash command handler
async def slash_ack(ack):
    await ack()


async def slash_lazy(command, respond, client):
    if not is_allowed_channel(command["channel_id"]):
        await respond(text="I only work in approved channels.", response_type="ephemeral")
        return
    text = command.get("text", "").strip()
    if not text:
        await respond(text="Try: `/cs create a ticket: API returning 500s`", response_type="ephemeral")
        return
    email = await resolve_user_email(client, command["user_id"])
    try:
        result = await agent_loop([{"role": "user", "content": text}], email, command["user_id"])
    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        result = f"Something went wrong: {e}"
    await respond(text=result, response_type="in_channel")


slack_app.command("/cs")(ack=slash_ack, lazy=[slash_lazy])


# Mention handler
async def mention_ack(ack):
    await ack()


async def mention_lazy(event, client):
    # Reply in the existing thread if there is one, else start a new thread
    reply_ts = event.get("thread_ts") or event["ts"]

    if not is_allowed_channel(event["channel"]):
        await client.chat_postMessage(
            channel=event["channel"],
            thread_ts=reply_ts,
            text="I only work in approved channels.",
        )
        return

    text = strip_bot_mention(event.get("text", "") or "")
    if not text:
        await client.chat_postMessage(
            channel=event["channel"],
            thread_ts=reply_ts,
            text="Mention me with an instruction. Example: `@chatops create a ticket: API down`",
        )
        return

    # If this is in an existing thread, fetch history for context
    if event.get("thread_ts"):
        history = await fetch_thread_history(client, event["channel"], event["thread_ts"])
        # The latest message is already in the history (this mention's text).
        # Make sure history isn't empty.
        if not history:
            history = [{"role": "user", "content": text}]
    else:
        history = [{"role": "user", "content": text}]

    email = await resolve_user_email(client, event["user"])
    try:
        result = await agent_loop(history, email, event["user"])
    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        result = f"Something went wrong: {e}"
    await client.chat_postMessage(
        channel=event["channel"],
        thread_ts=reply_ts,
        text=result,
    )


slack_app.event("app_mention")(ack=mention_ack, lazy=[mention_lazy])


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI()
slack_handler = AsyncSlackRequestHandler(slack_app)


@api.post("/slack/events")
async def slack_events(request: Request):
    return await slack_handler.handle(request)


@api.get("/health")
async def health():
    return {"status": "ok", "mcp_servers": list(mcp._sessions.keys())}


@api.on_event("startup")
async def on_startup() -> None:
    global mongo_client
    if settings.mongodb_uri:
        mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
        try:
            await mongo_client.admin.command("ping")
            logger.info("MongoDB connected for audit log")
        except Exception as e:
            logger.warning("MongoDB ping failed; audit log disabled: %s", e)
            mongo_client = None
    await mcp.start()


@api.on_event("shutdown")
async def on_shutdown() -> None:
    await mcp.stop()
    if mongo_client:
        mongo_client.close()


def main() -> None:
    import uvicorn
    uvicorn.run("app:api", host="0.0.0.0", port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()

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
import re
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

import aiohttp
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


PLANE_HOST = settings.plane_base_url.rstrip("/")

# Cached list of Plane workspace members: [{id, email, display_name}, ...]
plane_members_cache: list[dict] = []


# ---------------------------------------------------------------------------
# Local (chatops__*) tools — implemented in this process, exposed to the LLM
# alongside MCP tools. We use these for things the Plane API key can't do
# (e.g. server-side filtering by assignee, which goes through an OAuth-only
# endpoint we get 403 on).
# ---------------------------------------------------------------------------


LOCAL_TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "chatops__list_assigned_tickets",
            "description": (
                "List Plane work items assigned to a specific user, filtered server-side. "
                "ALWAYS prefer this over plane__list_work_items for any 'list X's tickets' / "
                "'list my tickets' / 'show me what is assigned to ...' request, because "
                "passing assignee_ids to plane__list_work_items routes through Plane's "
                "advanced-search endpoint which is forbidden for our API key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignee_email": {
                        "type": "string",
                        "description": "Plane email of the assignee. For 'my tickets', use the requesting user's email shown in the system prompt.",
                    },
                    "project": {
                        "type": "string",
                        "enum": ["CANDIDATE", "RECRUITER", "ALL"],
                        "description": "Limit to one project, or ALL (default).",
                    },
                },
                "required": ["assignee_email"],
            },
        },
    }
]


async def _chatops_list_assigned_tickets(args: dict) -> str:
    email = (args.get("assignee_email") or "").lower().strip()
    if not email:
        return json.dumps({"error": "assignee_email is required"})
    target = next(
        (m for m in plane_members_cache if (m.get("email") or "").lower() == email),
        None,
    )
    if not target:
        return json.dumps({
            "error": f"No Plane member with email {email!r}.",
            "available_emails": [m["email"] for m in plane_members_cache],
        })
    target_uuid = target["id"]
    proj_filter = (args.get("project") or "ALL").upper()

    project_pairs: list[tuple[str, str]] = []
    if proj_filter in ("RECRUITER", "ALL"):
        project_pairs.append(("RECRUITER", settings.plane_project_recruiter))
    if proj_filter in ("CANDIDATE", "ALL"):
        project_pairs.append(("CANDIDATE", settings.plane_project_candidate))

    matches: list[dict] = []
    PER_PAGE = 100
    for ident, proj_id in project_pairs:
        for page in range(20):  # safety: at most 2000 issues per project
            params: dict = {
                "project_id": proj_id,
                "per_page": PER_PAGE,
                "fields": "id,name,sequence_id,assignees,state",
            }
            if page > 0:
                # Plane MCP cursor format: "{per_page}:{page_index}:0"
                params["cursor"] = f"{PER_PAGE}:{page}:0"
            text = await mcp.call("plane__list_work_items", params)
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                logger.warning("list_work_items returned non-JSON for %s page %d", ident, page)
                break
            # The Plane MCP unwraps the response to a bare list; older versions
            # might return {"results": [...]}. Handle both.
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("results") or []
            else:
                items = []
            if not items:
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                if target_uuid in (item.get("assignees") or []):
                    matches.append({
                        "key": f"{ident}-{item.get('sequence_id')}",
                        "name": item.get("name"),
                        "id": item.get("id"),
                        "project": ident,
                        "state_id": item.get("state"),
                    })
            if len(items) < PER_PAGE:
                break  # last page
    logger.info("chatops__list_assigned_tickets: %s -> %d match(es)", email, len(matches))
    return json.dumps({
        "assignee_email": email,
        "assignee_name": target.get("display_name"),
        "count": len(matches),
        "tickets": matches,
    })


LOCAL_TOOL_HANDLERS = {
    "chatops__list_assigned_tickets": _chatops_list_assigned_tickets,
}


async def refresh_plane_members() -> None:
    """Fetch workspace members from Plane and cache."""
    global plane_members_cache
    try:
        result_text = await mcp.call("plane__get_workspace_members", {})
        # The tool returns JSON-ish text. Parse what we can.
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            # Sometimes content is a list of TextContent already concatenated; try to extract
            data = []
        members = []
        items = data if isinstance(data, list) else data.get("members", []) if isinstance(data, dict) else []
        for m in items:
            if not isinstance(m, dict):
                continue
            members.append({
                "id": m.get("id") or m.get("user_id") or m.get("member_id"),
                "email": m.get("email"),
                "display_name": m.get("display_name") or m.get("first_name") or m.get("email"),
            })
        plane_members_cache = [m for m in members if m.get("id") and m.get("email")]
        logger.info("Cached %d Plane workspace members", len(plane_members_cache))
    except Exception as e:
        logger.warning("Failed to fetch Plane members: %s", e)


def build_system_prompt() -> str:
    members_block = ""
    if plane_members_cache:
        rows = "\n".join(f"- `{m['email']}` → `{m['id']}` ({m.get('display_name', '')})" for m in plane_members_cache)
        members_block = f"\n## Plane workspace members (email → user_id)\nUse this map to resolve assignees. The `assignees` field on `plane__create_work_item` expects a list of Plane user_id values (UUIDs), NOT emails or Slack IDs.\n\n{rows}\n"
    else:
        members_block = "\n## Plane workspace members\nIf you need to assign someone, call `plane__get_workspace_members` first to get their Plane user_id (UUID).\n"

    return f"""You are RemoteStar's ChatOps assistant in Slack. You help the team manage Plane tickets through natural language.

## Workspace context
- Plane workspace slug: `{settings.plane_workspace_slug}`
- Plane host: `{PLANE_HOST}`
- Two projects available:
  - **CANDIDATE** (id: `{settings.plane_project_candidate}`) — candidate-facing app, profiles, jobs, interviews, matching, signup, resume
  - **RECRUITER** (id: `{settings.plane_project_recruiter}`) — recruiter dashboard, hiring flows, ATS integration, talent, scrapers

## How to pick the project (be decisive, don't over-ask)
- candidate, candidates, profile, signup, interview, jobs, matching, resume → CANDIDATE
- recruiter, recruiters, hiring, ATS, scraper, talent, dashboard → RECRUITER
- If the user explicitly says a project, use it without confirming
- If genuinely ambiguous, ask "CANDIDATE or RECRUITER?"
{members_block}
## Assigning tickets
- The user's message may contain emails (e.g., `rudy@remotestar.io`) — these come from Slack `@mentions` already resolved to emails.
- For `plane__create_work_item` and update tools, the `assignees` field expects a list of Plane user_id UUIDs. Look up the email in the workspace members map above to find the UUID.
- If you can't find a matching member, tell the user clearly: "I couldn't find a Plane user with email X — please check they're in the workspace."
- If the user only gave a name (not email), look up by display_name in the members map; if multiple match, ask which one.

## Listing and searching work items (READ THIS CAREFULLY)
Our API key has a hard limitation: ANY filter parameter on `plane__list_work_items` (assignee_ids, state_ids, state_groups, priorities, label_ids, type_ids, cycle_ids, module_ids, created_by_ids, query, workspace_search, etc.) routes through Plane's `/work-items/advanced-search/` endpoint which returns HTTP 403 for our key. Do NOT pass any of those filters — the call will always fail.

What works:
- **`plane__list_work_items` with ONLY `project_id`** (no other filters) — returns all issues in that project. Use pagination (`per_page`, `cursor`) for large projects.
- **`plane__search_work_items` with a `query`** — free-text workspace-wide search across name and description. Use this when the user gives a topic like "find tickets about login bug".
- **`plane__retrieve_work_item_by_identifier`** with `project_identifier` (RECRUITER or CANDIDATE) and `issue_identifier` (the integer sequence number) — for "show me RECRUITER-106" lookups.

How to handle common requests:
- "list my tickets" / "list <user>'s tickets" / "what is assigned to X" → ALWAYS use `chatops__list_assigned_tickets`. It takes an `assignee_email` and (optionally) a `project` and returns only the matching items — server-side filtering, no token waste. Resolve `<user>` to an email first using the workspace members list above (or the requesting user's email for "my tickets").
- "find tickets about X" → use `plane__search_work_items(query="X")`.
- "show me RECRUITER-106" → use `plane__retrieve_work_item_by_identifier`.
- Only fall back to `plane__list_work_items(project_id=...)` (no filters) when you genuinely need every issue in a project.

Project identifiers for `retrieve_work_item_by_identifier`:
- `RECRUITER` → recruiter project (UUID: `{settings.plane_project_recruiter}`)
- `CANDIDATE` → candidate project (UUID: `{settings.plane_project_candidate}`)

## Issue URL format (CRITICAL)
Self-hosted Plane URL format — use this exactly, never `plane.com`:

`{PLANE_HOST}/{settings.plane_workspace_slug}/projects/<PROJECT_ID>/issues/<ISSUE_ID>/`

After `plane__create_work_item` succeeds, extract the new issue's id from the tool result and construct this URL.

## Issue description
For `description_html`, format as HTML with the user's content followed by an attribution footer:

```html
<p>{{user_message}}</p>
<hr/>
<p><em>Created via ChatOps by {{user_email}} at {{timestamp_iso}}</em></p>
```

## Slack file attachments (IMPORTANT)
- If the user message mentions that files were attached in Slack (you'll see `[The user attached N file(s) in Slack: ...]`), the host application will upload those files AND embed them inline in the issue's description automatically AFTER your tool calls finish.
- Do NOT include any `<img>` tags in `description_html`. Do NOT make up image URLs like `https://plane.remotestar.io/path/to/image.png` — the system inserts the real `<img>` tags itself.
- Do NOT apologize or say "I can't attach files." Just operate on the right work item; uploads happen after.
- Attachments are bound to the LAST work item you created or updated. So if the user says "attach this image to PROJ-123", call `plane__update_work_item` (or `plane__retrieve_work_item_by_identifier` first to get its UUID) and stop — do not delete and recreate.

## Never use placeholder strings
Never pass literal strings like `<OLD_TICKET_ID>`, `<TYPE_ID>`, `<PROJECT_ID>`, etc. as tool arguments. They are not valid IDs and will produce 404s. If you don't have a real UUID, call the appropriate `list_*`, `search_work_items`, or `retrieve_work_item_by_identifier` tool first to obtain one.

## Tool naming
Tools are prefixed with `<server>__<tool>`. For Plane tools, use the `plane__*` names.

## User assistance
- Be concise. Slack-friendly markdown (no headings).
- After creating an issue, give the URL using the format above.
- If a tool fails, explain the error in plain English and suggest a fix.
"""


async def agent_loop(
    history: list[dict], user_email: str, user_slack_id: str
) -> tuple[str, dict | None]:
    """history is a list of {role, content} dicts. Last item is the current user request.

    Returns (final_text, created_issue) where created_issue is
    {project_id, issue_id} from the most recent successful plane__create_work_item
    call, or None if no issue was created.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    system = build_system_prompt() + f"\n\n## Current request\n- User email: {user_email}\n- User Slack ID: {user_slack_id}\n- Timestamp: {now_iso}\n"

    messages: list[dict] = [
        {"role": "system", "content": system},
        *history,
    ]

    tools = mcp.openai_tools() + LOCAL_TOOL_DEFS
    if not tools:
        return "I'm not connected to any backends right now. Try again in a moment.", None

    tool_call_log: list[dict] = []
    created_issue: dict | None = None
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
            return final, created_issue

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
            if tc.function.name in LOCAL_TOOL_HANDLERS:
                try:
                    result = await LOCAL_TOOL_HANDLERS[tc.function.name](args)
                except Exception as e:
                    logger.error("Local tool %s crashed: %s", tc.function.name, e, exc_info=True)
                    result = json.dumps({"error": str(e)})
            else:
                result = await mcp.call(tc.function.name, args)
            tool_call_log.append({"name": tc.function.name, "args": args, "result_preview": result[:200]})

            # Capture the issue id so we can attach files after the loop.
            # We track both create_work_item (new issue) and update_work_item
            # (so users can attach files to an existing ticket via the bot).
            if tc.function.name == "plane__create_work_item":
                try:
                    data = json.loads(result)
                    if isinstance(data, dict) and data.get("id") and args.get("project_id"):
                        created_issue = {"project_id": args["project_id"], "issue_id": data["id"]}
                except (json.JSONDecodeError, TypeError):
                    pass
            elif tc.function.name == "plane__update_work_item":
                proj = args.get("project_id")
                wid = args.get("work_item_id") or args.get("issue_id")
                if proj and wid:
                    created_issue = {"project_id": proj, "issue_id": wid}
            elif tc.function.name == "plane__delete_work_item":
                # If the LLM just deleted the issue we were tracking, drop it so
                # we don't try to upload attachments to a now-deleted ticket.
                wid = args.get("work_item_id") or args.get("issue_id")
                if created_issue and created_issue.get("issue_id") == wid:
                    logger.info("Cleared tracked issue %s because LLM deleted it", wid)
                    created_issue = None

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    fallback = "I tried but couldn't complete the request in a reasonable number of steps. Try rephrasing?"
    await audit_log(user_slack_id, user_email, tool_call_log, fallback)
    return fallback, created_issue


# ---------------------------------------------------------------------------
# Slack file -> Plane issue attachment bridge
#
# The Plane MCP server has no attachment tools (verified against all 109).
# Plane self-hosted uses a 3-step S3-presigned flow:
#   1. POST issue-attachments/ with {name,size,type} -> presigned upload_data + asset_id
#   2. POST upload_data.url multipart with the fields + file bytes
#   3. PATCH issue-attachments/{asset_id}/ with {is_uploaded: true}
# ---------------------------------------------------------------------------


def _looks_like_uuid(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


async def _attach_one_file(
    session: aiohttp.ClientSession,
    slack_file: dict,
    project_id: str,
    issue_id: str,
) -> dict | None:
    """Upload one Slack file to Plane. Returns {asset_url, name, mime} on success, else None."""
    name = slack_file.get("name") or "attachment"
    mime = slack_file.get("mimetype") or "application/octet-stream"
    download_url = slack_file.get("url_private_download") or slack_file.get("url_private")
    if not download_url:
        logger.warning("Slack file %r has no download URL; skipping", name)
        return None

    async with session.get(
        download_url,
        headers={"Authorization": f"Bearer {settings.slack_bot_token}"},
    ) as r:
        if r.status != 200:
            logger.warning("Slack download for %r returned %s", name, r.status)
            return None
        data = await r.read()
    size = len(data)

    plane_headers = {"X-API-Key": settings.plane_api_key, "Content-Type": "application/json"}
    create_url = (
        f"{PLANE_HOST}/api/v1/workspaces/{settings.plane_workspace_slug}"
        f"/projects/{project_id}/issues/{issue_id}/issue-attachments/"
    )

    async with session.post(
        create_url,
        json={"name": name, "size": size, "type": mime},
        headers=plane_headers,
    ) as r:
        if r.status not in (200, 201):
            body = await r.text()
            logger.warning("Plane create-attachment %r failed: %s %s", name, r.status, body[:300])
            return None
        meta = await r.json()

    upload = meta.get("upload_data") or {}
    asset_id = meta.get("asset_id")
    asset_url = meta.get("asset_url")
    if not upload.get("url") or not asset_id:
        logger.warning("Plane response missing upload_data/asset_id for %r: %s", name, meta)
        return None

    form = aiohttp.FormData()
    for k, v in upload["fields"].items():
        form.add_field(k, v)
    form.add_field("file", data, filename=name, content_type=mime)
    async with session.post(upload["url"], data=form) as r:
        if r.status not in (200, 201, 204):
            body = await r.text()
            logger.warning("Storage upload for %r failed: %s %s", name, r.status, body[:300])
            return None

    async with session.patch(
        f"{create_url}{asset_id}/",
        json={"is_uploaded": True},
        headers=plane_headers,
    ) as r:
        if r.status not in (200, 204):
            body = await r.text()
            logger.warning("Mark-uploaded for %r failed: %s %s", name, r.status, body[:300])
            return None

    logger.info("Attached %r (%d bytes) to issue %s", name, size, issue_id)
    return {"asset_url": asset_url, "name": name, "mime": mime}


async def attach_slack_files_to_plane_issue(
    slack_files: list[dict], project_id: str, issue_id: str
) -> tuple[list[dict], int]:
    """Upload all Slack files. Returns (list of successful uploads, total attempted).

    After uploading, the successful image attachments are also embedded inline in
    the issue's description_html so they render in the issue body, not just as a
    separate Attachments section."""
    if not slack_files:
        return [], 0
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_attach_one_file(session, f, project_id, issue_id) for f in slack_files],
            return_exceptions=True,
        )
    successful: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Attachment task raised: %s", r)
        elif isinstance(r, dict):
            successful.append(r)

    if successful:
        await _embed_attachments_in_description(project_id, issue_id, successful)

    return successful, len(slack_files)


async def _embed_attachments_in_description(
    project_id: str, issue_id: str, attachments: list[dict]
) -> None:
    """Append <img> / file links into the issue's description_html so attachments
    render inline. Images become <img>; everything else becomes a text link."""
    plane_headers = {"X-API-Key": settings.plane_api_key, "Content-Type": "application/json"}
    issue_url = (
        f"{PLANE_HOST}/api/v1/workspaces/{settings.plane_workspace_slug}"
        f"/projects/{project_id}/issues/{issue_id}/"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(issue_url, headers=plane_headers) as r:
                if r.status != 200:
                    logger.warning("Could not fetch issue for description embed: %s", r.status)
                    return
                issue = await r.json()
            current = issue.get("description_html") or ""

            blocks = []
            for a in attachments:
                full_url = f"{PLANE_HOST}{a['asset_url']}"
                if (a.get("mime") or "").startswith("image/"):
                    blocks.append(
                        f'<p><img src="{full_url}" alt="{a["name"]}" /></p>'
                    )
                else:
                    blocks.append(
                        f'<p><a href="{full_url}">{a["name"]}</a></p>'
                    )
            embed_html = "".join(blocks)

            # Insert before the attribution footer (<hr/>) if present, else append.
            if "<hr/>" in current:
                new_desc = current.replace("<hr/>", embed_html + "<hr/>", 1)
            else:
                new_desc = current + embed_html

            async with session.patch(
                issue_url,
                headers=plane_headers,
                json={"description_html": new_desc},
            ) as r:
                if r.status not in (200, 204):
                    body = await r.text()
                    logger.warning("Description embed PATCH failed: %s %s", r.status, body[:300])
                    return
            logger.info("Embedded %d attachment(s) inline in issue %s", len(attachments), issue_id)
    except Exception as e:
        logger.warning("Description embed crashed: %s", e, exc_info=True)


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


async def resolve_slack_mentions(client, text: str) -> str:
    """Convert `<@U...>` Slack user mentions into emails so the LLM can map to Plane users."""
    if not text or "<@" not in text:
        return text
    user_ids = set(re.findall(r"<@([UW][A-Z0-9]+)>", text))
    if not user_ids:
        return text
    for uid in user_ids:
        email = await resolve_user_email(client, uid)
        text = text.replace(f"<@{uid}>", email)
    return text


def strip_bot_mention(text: str) -> str:
    """Remove the leading <@BOT> mention from app_mention text. Only strips the first one;
    subsequent <@USER> mentions are preserved so they can be resolved to emails."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("<@") and ">" in text:
        text = text.split(">", 1)[1]
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


async def collect_thread_files(client, channel: str, thread_ts: str, limit: int = 30) -> list[dict]:
    """Scan a thread for any uploaded files (skipping the bot's own messages)."""
    try:
        result = await client.conversations_replies(channel=channel, ts=thread_ts, limit=limit)
        if not result.get("ok"):
            return []
        bot_uid = await get_bot_user_id(client)
        files: list[dict] = []
        for msg in result.get("messages", []):
            if msg.get("user") == bot_uid or msg.get("bot_id"):
                continue
            for f in msg.get("files") or []:
                files.append(f)
        return files
    except Exception as e:
        logger.warning("Failed to collect thread files: %s", e, exc_info=True)
        return []


async def fetch_thread_history(client, channel: str, thread_ts: str, limit: int = 30) -> list[dict]:
    """Fetch messages in a thread and convert to LLM message format."""
    try:
        result = await client.conversations_replies(channel=channel, ts=thread_ts, limit=limit)
        if not result.get("ok"):
            logger.warning("conversations.replies returned ok=false: %s", result.data)
            return []
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
        logger.warning("Failed to fetch thread history for %s: %s", thread_ts, e, exc_info=True)
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
    text = await resolve_slack_mentions(client, text)
    email = await resolve_user_email(client, command["user_id"])
    try:
        result, _created = await agent_loop([{"role": "user", "content": text}], email, command["user_id"])
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
    files = event.get("files") or []

    # If the user is in a thread and didn't attach files directly to this message,
    # pick up any files uploaded earlier in the thread.
    if not files and event.get("thread_ts"):
        files = await collect_thread_files(client, event["channel"], event["thread_ts"])
        if files:
            logger.info("Collected %d file(s) from thread context", len(files))

    logger.info(
        "Mention received: user=%s channel=%s text=%r files=%d",
        event.get("user"), event.get("channel"), text[:120], len(files),
    )
    for f in files:
        logger.info(
            "  file: name=%r mime=%r size=%s has_url=%s",
            f.get("name"), f.get("mimetype"), f.get("size"),
            bool(f.get("url_private_download") or f.get("url_private")),
        )

    if not text and not files:
        await client.chat_postMessage(
            channel=event["channel"],
            thread_ts=reply_ts,
            text="Mention me with an instruction. Example: `@chatops create a ticket: API down`",
        )
        return

    text = await resolve_slack_mentions(client, text)

    # Hint the LLM that attachments are present so it knows to create an issue.
    # The LLM doesn't upload them; we do that after the loop returns.
    if files:
        names = ", ".join(f.get("name") or "file" for f in files)
        text = (text or "Create a ticket for this.") + (
            f"\n\n[The user attached {len(files)} file(s) in Slack: {names}. "
            "These will be auto-attached to the new Plane issue after you create it.]"
        )

    # If this is in an existing thread, fetch history for context
    if event.get("thread_ts"):
        history = await fetch_thread_history(client, event["channel"], event["thread_ts"])
        # Replace the last user message with the mention-resolved version
        if history and history[-1]["role"] == "user":
            history[-1]["content"] = text
        else:
            history.append({"role": "user", "content": text})
    else:
        history = [{"role": "user", "content": text}]

    email = await resolve_user_email(client, event["user"])
    created_issue: dict | None = None
    try:
        result, created_issue = await agent_loop(history, email, event["user"])
    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        result = f"Something went wrong: {e}"

    if files:
        if created_issue and _looks_like_uuid(created_issue.get("issue_id")) and _looks_like_uuid(created_issue.get("project_id")):
            uploaded, total = await attach_slack_files_to_plane_issue(
                files, created_issue["project_id"], created_issue["issue_id"]
            )
            ok = len(uploaded)
            if total:
                if ok == total:
                    result += f"\n\nAttached {ok} file{'s' if ok != 1 else ''} inline in the issue."
                else:
                    result += f"\n\nAttached {ok}/{total} files (some failed; check logs)."
        elif created_issue:
            logger.warning("Skipping upload: invalid issue ids %s", created_issue)
            result += "\n\n_(Couldn't attach files: the ticket ID I picked up wasn't a valid UUID.)_"
        else:
            result += (
                f"\n\n_(I saw {len(files)} attachment(s) but didn't operate on a Plane issue, "
                "so I didn't upload them. Mention me again asking to create or update a specific ticket.)_"
            )

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
    await refresh_plane_members()


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

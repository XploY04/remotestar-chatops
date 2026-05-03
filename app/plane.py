"""Plane plumbing: MCP subprocess manager, members/states caches,
the chatops__list_assigned_tickets local tool, and the Slack-file-to-Plane
attachment bridge (3-step S3 presigned upload + inline-image embed).

The Plane MCP server has no attachment tools (verified across all 109), so the
attachment flow bypasses MCP and hits Plane's REST API directly."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import AsyncExitStack
from typing import Any

import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import PLANE_HOST, logger, settings


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

    @property
    def sessions(self) -> dict[str, ClientSession]:
        return self._sessions

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
# Caches refreshed at startup
# ---------------------------------------------------------------------------

# Plane workspace members: [{id, email, display_name}, ...]
plane_members_cache: list[dict] = []

# Plane states: state_uuid -> {name, group, project_id}
plane_states_cache: dict[str, dict] = {}


async def refresh_plane_members() -> None:
    """Fetch workspace members from Plane and cache."""
    global plane_members_cache
    try:
        result_text = await mcp.call("plane__get_workspace_members", {})
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            data = []
        members: list[dict] = []
        items = data if isinstance(data, list) else (
            data.get("members", []) if isinstance(data, dict) else []
        )
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


async def refresh_plane_states() -> None:
    """Cache state metadata per project: state_uuid -> {name, group, project_id}."""
    global plane_states_cache
    new_cache: dict[str, dict] = {}
    for proj_id in (settings.plane_project_recruiter, settings.plane_project_candidate):
        try:
            text = await mcp.call("plane__list_states", {"project_id": proj_id})
            data = json.loads(text)
            items = data if isinstance(data, list) else (
                data.get("results") or [] if isinstance(data, dict) else []
            )
            for s in items:
                if isinstance(s, dict) and s.get("id"):
                    new_cache[s["id"]] = {
                        "name": s.get("name"),
                        "group": s.get("group"),
                        "project_id": proj_id,
                    }
        except Exception as e:
            logger.warning("Failed to fetch states for project %s: %s", proj_id, e)
    plane_states_cache = new_cache
    logger.info("Cached %d Plane states across both projects", len(plane_states_cache))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def looks_like_uuid(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


def pick_state_for_group(project_id: str, target_group: str) -> str | None:
    """Pick the canonical state UUID for a (project_id, state_group). If multiple
    states share the group (e.g. Staging/Production/Done are all 'completed'),
    prefer the one named Done > In Progress > Backlog > anything else."""
    candidates = [
        (sid, info) for sid, info in plane_states_cache.items()
        if info.get("project_id") == project_id
        and (info.get("group") or "").lower() == target_group
    ]
    if not candidates:
        return None
    PREF_NAMES = {"done", "in progress", "todo", "backlog", "cancelled"}
    for sid, info in candidates:
        if (info.get("name") or "").strip().lower() in PREF_NAMES:
            return sid
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Local (chatops__*) tools
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
                "advanced-search endpoint which is forbidden for our API key. "
                "By default returns only Todo (state_group=unstarted) tickets."
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
                    "state_groups": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["backlog", "unstarted", "started", "completed", "cancelled"],
                        },
                        "description": (
                            "Which Plane state groups to include. Default ['unstarted'] which is 'Todo'. "
                            "Pass ['unstarted','started'] for 'open' / 'pending'. "
                            "Pass [] (empty array) to include every state."
                        ),
                    },
                },
                "required": ["assignee_email"],
            },
        },
    }
]


async def chatops_list_assigned_tickets(args: dict) -> str:
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

    state_groups = args.get("state_groups")
    if state_groups is None:
        state_groups = ["unstarted"]  # default = Todo
    state_filter = {s.lower() for s in state_groups if isinstance(s, str)}

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
                if target_uuid not in (item.get("assignees") or []):
                    continue
                state_id = item.get("state")
                state_meta = plane_states_cache.get(state_id) or {}
                state_group = (state_meta.get("group") or "").lower()
                if state_filter and state_group not in state_filter:
                    continue
                seq = item.get("sequence_id")
                issue_id = item.get("id")
                matches.append({
                    "key": f"{ident}-{seq}",
                    "name": item.get("name"),
                    "url": (
                        f"{PLANE_HOST}/{settings.plane_workspace_slug}"
                        f"/projects/{proj_id}/issues/{issue_id}/"
                    ),
                    "project": ident,
                    "state": state_meta.get("name"),
                    "state_group": state_group or None,
                })
            if len(items) < PER_PAGE:
                break  # last page
    logger.info(
        "chatops__list_assigned_tickets: %s state_groups=%s -> %d match(es)",
        email, sorted(state_filter) if state_filter else "ALL", len(matches),
    )
    return json.dumps({
        "assignee_email": email,
        "assignee_name": target.get("display_name"),
        "state_groups": sorted(state_filter) if state_filter else "ALL",
        "count": len(matches),
        "tickets": matches,
    })


LOCAL_TOOL_HANDLERS = {
    "chatops__list_assigned_tickets": chatops_list_assigned_tickets,
}


# ---------------------------------------------------------------------------
# Slack file -> Plane issue attachment bridge
#
# The Plane MCP has no attachment tools; Plane self-hosted uses 3-step S3:
#   1. POST issue-attachments/ {name,size,type} -> presigned upload_data + asset_id
#   2. POST upload_data.url multipart with the presigned fields + file bytes
#   3. PATCH issue-attachments/{asset_id}/ {is_uploaded: true}
# ---------------------------------------------------------------------------


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

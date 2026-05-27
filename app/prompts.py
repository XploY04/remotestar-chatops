"""System prompt builder, help text variants, and Slack mrkdwn coercion."""

from __future__ import annotations

import re

from app.config import PLANE_HOST, settings
from app.instructions import get_instructions
from app.plane import mcp, plane_members_cache, plane_states_cache


HELP_TEXT_PLANE = """*RemoteStar ChatOps* — what I can do here:

*Tickets*
- `@chatops list my tickets` — your Todo tickets (default). Add "open" for in-progress too.
- `@chatops list <user>'s tickets` — pending work for someone (mention them with @)
- `@chatops create a ticket: <description>` — new ticket, auto-routed by keywords
- `@chatops add a comment to RECRUITER-109: <text>` — comment on a ticket
- `@chatops close RECRUITER-109` — mark done
- `@chatops show me RECRUITER-109` — fetch a specific ticket

*Search*
- `@chatops find tickets about <topic>` — full-text search across both projects

*Attachments*
- Drop a screenshot in the same message OR earlier in the thread; I upload it to the issue and embed it inline in the description.

*Reactions* (on my own messages)
- :white_check_mark: → mark Done
- :construction: → mark In Progress
- :back: → move to Backlog
- :x: → mark Cancelled
"""


HELP_TEXT_CHATBOT = """*RemoteStar ChatOps* in this channel:

I'm a general-purpose assistant for the team. Just mention me with a question or task and I'll help. I have your channel's context loaded so I know what you work on.

Examples:
- `@chatops draft three subject lines for our launch email`
- `@chatops summarize what's been said in this thread`
- `@chatops what's the right tone for a Series A announcement vs a feature drop?`

I don't have Plane tools or external integrations in this channel — just conversation.
"""


HELP_TEXT_MIXPANEL = """*RemoteStar ChatOps* in this channel:

I'm scoped to Mixpanel here. I can run queries, list/edit events and properties, manage dashboards, and pull session replays via the Mixpanel MCP. No Plane tools.

Examples:
- `@chatops how many signups in the last 7 days?`
- `@chatops show me the candidate-signup funnel for the last 30 days`
- `@chatops list dashboards`
- `@chatops which events were tracked yesterday and how many times?`

*Limits*
- Shared 600 requests/hour across everyone using me here.
- Read tools run immediately; destructive tools (delete, bulk-edit) ask you to confirm first.
"""


def is_help_text(text: str) -> bool:
    t = (text or "").strip().lower().rstrip("?").strip()
    return t in {"help", "what can you do", "what do you do", "commands"}


def help_text_for(mode: str) -> str:
    if mode == "chatbot":
        return HELP_TEXT_CHATBOT
    if mode == "mixpanel":
        return HELP_TEXT_MIXPANEL
    return HELP_TEXT_PLANE


_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_MD_BOLD_RE = re.compile(r"\*\*([^\n*][^\n]*?)\*\*")
_MD_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.*?)[ \t]*$", re.MULTILINE)


def to_slack_mrkdwn(text: str) -> str:
    if not text:
        return text
    text = _MD_LINK_RE.sub(r"<\2|\1>", text)
    text = _MD_BOLD_RE.sub(r"*\1*", text)
    text = _MD_HEADING_RE.sub(r"*\1*", text)
    return text


_SLACK_FORMATTING_BLOCK = """## Slack message formatting (mrkdwn, NOT standard markdown)
Slack uses its own variant of markdown. Output your replies in that syntax — never standard CommonMark/GitHub markdown.
- Bold: `*bold*` (single asterisks). NEVER `**bold**` — Slack shows the asterisks literally.
- Italic: `_italic_`.
- Inline code: `` `code` ``. Code block: triple backticks, no language tag.
- Links: `<https://example.com|label>`. NEVER `[label](https://example.com)` — Slack shows the brackets literally.
- Bullets: start the line with `•` or `-`. Headings (`#`, `##`) are NOT supported — use bold instead.
- Strikethrough: `~text~`."""


def _channel_block(channel_id):
    body = get_instructions(channel_id).strip() if channel_id else ""
    if not body:
        return ""
    return (
        "\n## Channel context\nThe instructions below are specific to this Slack channel and override "
        "the generic guidance above when there's a conflict.\n\n"
        + body + "\n"
    )


def _build_plane_prompt(channel_id):
    members_block = ""
    if plane_members_cache:
        rows = "\n".join(
            "- `" + m["email"] + "` → `" + m["id"] + "` (" + m.get("display_name", "") + ")"
            for m in plane_members_cache
        )
        members_block = (
            "\n## Plane workspace members (email → user_id)\n"
            "Use this map to resolve assignees. The `assignees` field on `plane__create_work_item` "
            "expects a list of Plane user_id values (UUIDs), NOT emails or Slack IDs.\n\n"
            + rows + "\n"
        )
    else:
        members_block = (
            "\n## Plane workspace members\n"
            "If you need to assign someone, call `plane__get_workspace_members` first to get their Plane user_id (UUID).\n"
        )

    GROUP_ORDER = ["backlog", "unstarted", "started", "completed", "cancelled"]
    states_block = ""
    if plane_states_cache:
        project_to_states = {}
        for sid, info in plane_states_cache.items():
            project_to_states.setdefault(info.get("project_id") or "", []).append({
                "id": sid,
                "name": info.get("name") or "",
                "group": (info.get("group") or "").lower(),
            })
        for project_id in project_to_states:
            project_to_states[project_id].sort(
                key=lambda s: (
                    GROUP_ORDER.index(s["group"]) if s["group"] in GROUP_ORDER else 99,
                    s["name"].lower(),
                )
            )

        sections = []
        project_label = {
            settings.plane_project_candidate: "CANDIDATE",
            settings.plane_project_recruiter: "RECRUITER",
        }
        for project_id, label in project_label.items():
            states = project_to_states.get(project_id) or []
            if not states:
                continue
            lines = ["### " + label]
            for s in states:
                lines.append("- " + s["name"] + " (" + s["group"] + ") → `" + s["id"] + "`")
            sections.append("\n".join(lines))

        states_block = (
            "\n## Plane states (per project) — for `plane__update_work_item`\n"
            "The `state` field expects a state **UUID**, NEVER the name. Look it up here. "
            "If the user names a state that's not in this list (e.g. 'Dev', 'QA'), DO NOT make one up — "
            "tell them the state doesn't exist in that project and list the valid options.\n\n"
            + "\n\n".join(sections)
            + "\n"
        )

    mixpanel_block = ""
    if "mixpanel" in mcp.sessions:
        mixpanel_block = (
            "\n## Mixpanel analytics (read-only in this channel)\n"
            "You can answer product-analytics questions via Mixpanel MCP "
            "tools prefixed `mixpanel__`. In plane-mode channels (like this "
            "one) only the read-only Mixpanel tools are exposed. "
            "Mutating ops are not available here; tell the user "
            "to go to a Mixpanel-mode channel like #tech for those.\n"
        )

    workspace_slug = settings.plane_workspace_slug
    candidate_id = settings.plane_project_candidate
    recruiter_id = settings.plane_project_recruiter

    return (
        "You are RemoteStar's ChatOps assistant in Slack. You help the team manage Plane tickets through natural language.\n\n"
        "## Workspace context\n"
        "- Plane workspace slug: `" + workspace_slug + "`\n"
        "- Plane host: `" + PLANE_HOST + "`\n"
        "- Two projects available:\n"
        "  - **CANDIDATE** (id: `" + candidate_id + "`) — candidate-facing app, profiles, jobs, interviews, matching, signup, resume\n"
        "  - **RECRUITER** (id: `" + recruiter_id + "`) — recruiter dashboard, hiring flows, ATS integration, talent, scrapers\n\n"
        "## How to pick the project (be decisive, don't over-ask)\n"
        "- candidate, candidates, profile, signup, interview, jobs, matching, resume → CANDIDATE\n"
        "- recruiter, recruiters, hiring, ATS, scraper, talent, dashboard → RECRUITER\n"
        "- If the user explicitly says a project, use it without confirming\n"
        "- If genuinely ambiguous, ask \"CANDIDATE or RECRUITER?\"\n"
        + members_block + states_block + mixpanel_block +
        "\n## Setting state and labels on update_work_item\n"
        "- `state` → must be a state UUID from the per-project list above, not a name.\n"
        "- `labels` → must be a list of label UUIDs, not label names. Call `plane__list_labels` with the project_id first.\n"
        "- If you can't find a matching state/label, tell the user what valid options exist; do NOT fabricate a UUID.\n\n"
        "## Assigning tickets\n"
        "- The user's message may contain emails — these come from Slack `@mentions` already resolved to emails.\n"
        "- For create and update tools, the `assignees` field expects a list of Plane user_id UUIDs.\n"
        "- If you can't find a matching member, tell the user clearly.\n\n"
        "## Listing and searching work items\n"
        "Our API key has a hard limitation: ANY filter parameter on `plane__list_work_items` routes through Plane's advanced-search endpoint which returns HTTP 403. Do NOT pass any filters.\n\n"
        "What works:\n"
        "- `plane__list_work_items` with ONLY `project_id` (no other filters)\n"
        "- `plane__search_work_items` with a `query` — free-text workspace-wide search\n"
        "- `plane__retrieve_work_item_by_identifier` with `project_identifier` (RECRUITER or CANDIDATE) and `issue_identifier`\n\n"
        "How to handle common requests:\n"
        "- \"list my tickets\" / \"list X's tickets\" → ALWAYS use `chatops__list_assigned_tickets`\n"
        "- \"find tickets about X\" → use `plane__search_work_items(query=\"X\")`\n"
        "- \"show me RECRUITER-106\" → use `plane__retrieve_work_item_by_identifier`\n\n"
        "## Issue URL format\n"
        "Self-hosted Plane URL: `" + PLANE_HOST + "/" + workspace_slug + "/projects/<PROJECT_ID>/issues/<ISSUE_ID>/`\n\n"
        "## Slack file attachments\n"
        "- If files were attached, the host application will upload them automatically AFTER your tool calls finish.\n"
        "- Do NOT include any img tags. Do NOT make up image URLs.\n"
        "- Do NOT apologize. Just operate on the right work item.\n\n"
        "## Never use placeholder strings\n"
        "Never pass literal strings like `<OLD_TICKET_ID>` as tool arguments. If you don't have a real UUID, call list/search/retrieve first.\n\n"
        "## Tool naming\n"
        "Tools are prefixed with `<server>__<tool>`. For Plane tools, use the `plane__*` names.\n\n"
        + _SLACK_FORMATTING_BLOCK + "\n\n"
        "When listing tickets, render each as one line.\n\n"
        "## User assistance\n"
        "- Be concise. After creating an issue, give the URL.\n"
        "- If a tool fails, explain the error in plain English and suggest a fix.\n"
        + _channel_block(channel_id)
    )


def _build_chatbot_prompt(channel_id):
    return (
        "You are RemoteStar's ChatOps assistant in Slack, helping this team with their work. "
        "The channel context below is your PRIMARY source of truth — it contains this team's playbooks, policies, and processes. "
        "When a user asks a question, search the channel context FIRST and answer from it directly and confidently. "
        "Do NOT say 'I don't have access to that information' if the answer is in the channel context below — read it again. "
        "Only fall back to general knowledge if the channel context genuinely doesn't cover the topic. "
        "You do not have Plane, GitHub, or other tool integrations in this channel, so if the user asks you to PERFORM an action that requires an external system, say so plainly.\n\n"
        + _SLACK_FORMATTING_BLOCK + "\n"
        + _channel_block(channel_id)
    )


def _build_mixpanel_prompt(channel_id):
    return (
        "You are RemoteStar's ChatOps assistant in Slack, scoped to Mixpanel analytics. "
        "Your job is to answer product-analytics questions by calling Mixpanel MCP tools (prefixed `mixpanel__`) "
        "until you have a real answer, then summarize it for the user in Slack. "
        "You do NOT have Plane, GitHub, or any other integration in this channel.\n\n"
        "## Workflow: always Discover, then Query, then Summarize\n\n"
        "1. Discover the right project, event, and property names using `Get-Projects`, `Get-Events`, `List-Properties`.\n"
        "2. Query to get the actual numbers using `Run-Query` (call `Get-Query-Schema` first for non-trivial queries).\n"
        "3. Summarize in one or two Slack lines.\n\n"
        "`Get-Projects` alone is NEVER a sufficient response. Keep going until you have a number.\n\n"
        "## Never fabricate a refusal\n\n"
        "The ONLY acceptable reason to refuse is a real error message from a tool you actually called. Quote it and stop.\n\n"
        "## Date discipline\n\n"
        "Use the current UTC timestamp from the system message for time windows.\n\n"
        "## Project routing\n\n"
        "- Candidate: signups, profiles, jobs, matching, resume, interview, applies → Candidate project\n"
        "- Recruiter: recruiter dashboard, ATS flows, hiring funnel, talent search → Recruiter project\n\n"
        "## Confirmation before destructive operations\n\n"
        "Before any mutating tool (Delete-*, Bulk-Edit-*, Edit-Event, etc.), write a one-line summary and ask \"OK to proceed?\"\n\n"
        "## Rate limit\n\n"
        "Shared 600 Mixpanel requests per hour. On a 429, stop and surface the error.\n\n"
        + _SLACK_FORMATTING_BLOCK + "\n"
        + _channel_block(channel_id)
    )


def build_system_prompt(channel_id, mode):
    if mode == "chatbot":
        return _build_chatbot_prompt(channel_id)
    if mode == "mixpanel":
        return _build_mixpanel_prompt(channel_id)
    return _build_plane_prompt(channel_id)

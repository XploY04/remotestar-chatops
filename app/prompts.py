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


# --- Slack mrkdwn coercion ---------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_MD_BOLD_RE = re.compile(r"\*\*([^\n*][^\n]*?)\*\*")
_MD_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.*?)[ \t]*$", re.MULTILINE)


def to_slack_mrkdwn(text: str) -> str:
    """Best-effort conversion of common standard-markdown patterns the LLM
    sometimes emits into Slack's mrkdwn dialect. Safety net only — the system
    prompt also tells the LLM to write mrkdwn directly."""
    if not text:
        return text
    text = _MD_LINK_RE.sub(r"<\2|\1>", text)
    text = _MD_BOLD_RE.sub(r"*\1*", text)
    text = _MD_HEADING_RE.sub(r"*\1*", text)
    return text


# --- System prompt -----------------------------------------------------------

_SLACK_FORMATTING_BLOCK = """## Slack message formatting (mrkdwn, NOT standard markdown)
Slack uses its own variant of markdown. Output your replies in that syntax — never standard CommonMark/GitHub markdown.
- Bold: `*bold*` (single asterisks). NEVER `**bold**` — Slack shows the asterisks literally.
- Italic: `_italic_`.
- Inline code: `` `code` ``. Code block: triple backticks, no language tag.
- Links: `<https://example.com|label>`. NEVER `[label](https://example.com)` — Slack shows the brackets literally.
- Bullets: start the line with `•` or `-`. Headings (`#`, `##`) are NOT supported — use bold instead.
- Strikethrough: `~text~`."""


def _channel_block(channel_id: str | None) -> str:
    body = get_instructions(channel_id).strip() if channel_id else ""
    if not body:
        return ""
    # The body from get_instructions() already contains its own
    # "Team Knowledge Base" header and authoritative framing, so we
    # don't wrap it in another "Channel context" header — that would
    # nest headings and dilute the source-of-truth signal.
    return f"\n{body}\n"


def _build_plane_prompt(channel_id: str | None) -> str:
    members_block = ""
    if plane_members_cache:
        rows = "\n".join(
            f"- `{m['email']}` → `{m['id']}` ({m.get('display_name', '')})"
            for m in plane_members_cache
        )
        members_block = (
            "\n## Plane workspace members (email → user_id)\n"
            "Use this map to resolve assignees. The `assignees` field on `plane__create_work_item` "
            "expects a list of Plane user_id values (UUIDs), NOT emails or Slack IDs.\n\n"
            f"{rows}\n"
        )
    else:
        members_block = (
            "\n## Plane workspace members\n"
            "If you need to assign someone, call `plane__get_workspace_members` first to get their Plane user_id (UUID).\n"
        )

    # Per-project states block, sorted by group order so the LLM can pick fluently.
    GROUP_ORDER = ["backlog", "unstarted", "started", "completed", "cancelled"]
    states_block = ""
    if plane_states_cache:
        project_to_states: dict[str, list[dict]] = {}
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
            lines = [f"### {label}"]
            for s in states:
                lines.append(f"- {s['name']} ({s['group']}) → `{s['id']}`")
            sections.append("\n".join(lines))

        states_block = (
            "\n## Plane states (per project) — for `plane__update_work_item`\n"
            "The `state` field expects a state **UUID**, NEVER the name. Look it up here. "
            "If the user names a state that's not in this list (e.g. 'Dev', 'QA'), DO NOT make one up — "
            "tell them the state doesn't exist in that project and list the valid options.\n\n"
            + "\n\n".join(sections)
            + "\n"
        )

    # Mixpanel availability is dynamic: the MCP subprocess may or may not
    # be connected at request time. Only mention it in the prompt when the
    # session is actually live, otherwise the LLM might try to call
    # mixpanel__* tools that aren't in the tools array.
    mixpanel_block = ""
    if "mixpanel" in mcp.sessions:
        mixpanel_block = (
            "\n## Mixpanel analytics (read-only in this channel)\n"
            "You can answer product-analytics questions via Mixpanel MCP "
            "tools prefixed `mixpanel__`. In plane-mode channels (like this "
            "one) only the read-only Mixpanel tools are exposed: `Run-Query`, "
            "`Get-Query-Schema`, `Get-Report`, `Display-Query`, `Get-Projects`, "
            "`Get-Events`, `List-Properties`, `Get-Property-Values`, "
            "`Search-Entities`, `Get-Business-Context`, `Get-Issues`, "
            "`List-Metrics`, `Get-Metric`, `List-Dashboards`, `Get-Dashboard`. "
            "Mutating ops (Edit-*, Delete-*, Bulk-Edit-*, Create-Metric, "
            "Update-Feature-Flag, etc.) are not available here; tell the user "
            "to go to a Mixpanel-mode channel like #tech for those.\n\n"
            "Workflow for any analytics question: `Get-Projects` to find the "
            "project, `Get-Events` (and `List-Properties` if needed) to find "
            "the event, then `Run-Query` (call `Get-Query-Schema` first for "
            "non-trivial queries). `Get-Projects` alone is never a sufficient "
            "answer; keep going until you have a number.\n\n"
            "Never refuse based on speculation about regions or permissions. "
            "Only surface a refusal when a tool you actually called returned a "
            "real error message; quote that message. Rate limit is 600 "
            "Mixpanel requests/hour across all users; on a 429, stop.\n"
        )

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
{members_block}{states_block}{mixpanel_block}
## Setting state and labels on update_work_item
- `state` → must be a state UUID from the per-project list above, not a name. If the user says "set to Done in CANDIDATE", look up Done's UUID under CANDIDATE and pass that.
- `labels` → must be a list of label UUIDs, not label names. We don't have a labels lookup table cached. If the user asks to add/remove labels by name, call `plane__list_labels` with the project_id first to get the UUIDs, then pass those.
- If you can't find a matching state/label, tell the user what valid options exist; do NOT fabricate a UUID.

## Assigning tickets
- The user's message may contain emails (e.g., `rudy@remotestar.io`) — these come from Slack `@mentions` already resolved to emails.
- For `plane__create_work_item` and update tools, the `assignees` field expects a list of Plane user_id UUIDs. Look up the email in the workspace members map above to find the UUID.
- If you can't find a matching member, tell the user clearly: "I couldn't find a Plane user with email X — please check they're in the workspace."
- If the user only gave a name (not email), look up by display_name in the members map; if multiple match, ask which one.

## Listing and searching work items (READ THIS CAREFULLY)
Our API key has a hard limitation: ANY filter parameter on `plane__list_work_items` (assignee_ids, state_ids, state_groups, priorities, label_ids, type_ids, cycle_ids, module_ids, created_by_ids, query, workspace_search, etc.) routes through Plane's `/work-items/advanced-search/` endpoint which returns HTTP 403 for our key. Do NOT pass any of those filters — the call will always fail.

What works:
- **`plane__list_work_items` with ONLY `project_id`** (no other filters) — returns all issues in that project. Use pagination (`per_page`, `cursor`) for large projects. **ALWAYS pass `fields="id,name,sequence_id,state,assignees"`** — without it the response includes full `description_html` for every issue and a few hundred issues will blow past the LLM context window. The `name` field is required by the MCP's schema; the others keep the payload small.
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

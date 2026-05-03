"""System prompt builder, help text variants, and Slack mrkdwn coercion."""

from __future__ import annotations

import re

from app.config import PLANE_HOST, settings
from app.instructions import get_instructions
from app.plane import plane_members_cache


HELP_TEXT_PLANE = """*RemoteStar ChatOps* — what I can do here:

*Tickets*
• `@chatops list my tickets` — your Todo tickets (default). Add "open" for in-progress too.
• `@chatops list <user>'s tickets` — pending work for someone (mention them with @)
• `@chatops create a ticket: <description>` — new ticket, auto-routed by keywords
• `@chatops add a comment to RECRUITER-109: <text>` — comment on a ticket
• `@chatops close RECRUITER-109` — mark done
• `@chatops show me RECRUITER-109` — fetch a specific ticket

*Search*
• `@chatops find tickets about <topic>` — full-text search across both projects

*Attachments*
• Drop a screenshot in the same message OR earlier in the thread; I upload it to the issue and embed it inline in the description.

*Reactions* (on my own messages)
• :white_check_mark: → mark Done
• :construction: → mark In Progress
• :back: → move to Backlog
• :x: → mark Cancelled
"""


HELP_TEXT_CHATBOT = """*RemoteStar ChatOps* in this channel:

I'm a general-purpose assistant for the team. Just mention me with a question or task and I'll help. I have your channel's context loaded so I know what you work on.

Examples:
• `@chatops draft three subject lines for our launch email`
• `@chatops summarize what's been said in this thread`
• `@chatops what's the right tone for a Series A announcement vs a feature drop?`

I don't have Plane tools or external integrations in this channel — just conversation.
"""


def is_help_text(text: str) -> bool:
    t = (text or "").strip().lower().rstrip("?").strip()
    return t in {"help", "what can you do", "what do you do", "commands"}


def help_text_for(mode: str) -> str:
    return HELP_TEXT_CHATBOT if mode == "chatbot" else HELP_TEXT_PLANE


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
    return (
        "\n## Channel context\nThe instructions below are specific to this Slack channel and override "
        "the generic guidance above when there's a conflict.\n\n"
        f"{body}\n"
    )


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

{_SLACK_FORMATTING_BLOCK}

When listing tickets, render each as one line:
`• *RECRUITER-109* — <https://plane.remotestar.io/.../issues/.../|the issue title> _(Todo)_`

## User assistance
- Be concise. After creating an issue, give the URL using the format above.
- If a tool fails, explain the error in plain English and suggest a fix.
{_channel_block(channel_id)}"""


def _build_chatbot_prompt(channel_id: str | None) -> str:
    return f"""You are RemoteStar's ChatOps assistant in Slack, acting as a general-purpose helper for this team. You do NOT have access to Plane, GitHub, or any other integration in this channel — answer from your general knowledge plus the channel context below. If the user asks for something that requires an external system you cannot reach, say so plainly and suggest where they should go instead.

{_SLACK_FORMATTING_BLOCK}
{_channel_block(channel_id)}"""


def build_system_prompt(channel_id: str | None, mode: str) -> str:
    if mode == "chatbot":
        return _build_chatbot_prompt(channel_id)
    return _build_plane_prompt(channel_id)

"""Slack listeners: app_mention, message (DM), slash command, reaction_added.

Mode is resolved per request via app.instructions.resolve_mode(channel). If a
channel has no instructions file, the bot drops the event silently (no fallback
to a default mode)."""

from __future__ import annotations

import json
import re

from app.agent import agent_loop
from app.config import logger
from app.instructions import resolve_mode
from app.plane import (
    attach_slack_files_to_plane_issue,
    looks_like_uuid,
    mcp,
    pick_state_for_group,
    plane_states_cache,
)
from app.prompts import help_text_for, is_help_text, to_slack_mrkdwn
from app.slack_app import slack_app


# ---------------------------------------------------------------------------
# Small Slack helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared agent flow for both @mentions and DMs
# ---------------------------------------------------------------------------


async def handle_user_request(
    client,
    *,
    channel: str,
    user_id: str,
    text: str,
    files: list[dict],
    thread_ts: str | None,
    reply_ts: str | None,
    mode: str,
) -> None:
    # Help short-circuit — deterministic, fast, no LLM call.
    if is_help_text(text) and not files:
        await client.chat_postMessage(channel=channel, thread_ts=reply_ts, text=help_text_for(mode))
        return

    text = await resolve_slack_mentions(client, text)

    # In Plane mode, if files came in, hint the LLM so it knows to create an issue.
    # In chatbot mode, files are silently ignored — V1 has no vision/upload there.
    if files and mode == "plane":
        names = ", ".join(f.get("name") or "file" for f in files)
        text = (text or "Create a ticket for this.") + (
            f"\n\n[The user attached {len(files)} file(s) in Slack: {names}. "
            "These will be auto-attached to the new Plane issue after you create it.]"
        )

    if thread_ts:
        history = await fetch_thread_history(client, channel, thread_ts)
        if history and history[-1]["role"] == "user":
            history[-1]["content"] = text
        else:
            history.append({"role": "user", "content": text})
    else:
        history = [{"role": "user", "content": text}]

    email = await resolve_user_email(client, user_id)
    created_issue: dict | None = None
    try:
        result, created_issue = await agent_loop(history, email, user_id, channel_id=channel, mode=mode)
    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        result = f"Something went wrong: {e}"

    # Attachments: only in plane mode.
    if files and mode == "plane":
        if (
            created_issue
            and looks_like_uuid(created_issue.get("issue_id"))
            and looks_like_uuid(created_issue.get("project_id"))
        ):
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
        channel=channel,
        thread_ts=reply_ts,
        text=to_slack_mrkdwn(result),
    )


# ---------------------------------------------------------------------------
# Slash command — /cs
# ---------------------------------------------------------------------------


async def slash_ack(ack):
    await ack()


async def slash_lazy(command, respond, client):
    channel = command["channel_id"]
    mode = resolve_mode(channel)
    if mode is None:
        await respond(text="I'm not configured for this channel.", response_type="ephemeral")
        return
    text = command.get("text", "").strip()
    if not text:
        await respond(text="Try: `/cs help`", response_type="ephemeral")
        return
    text = await resolve_slack_mentions(client, text)
    email = await resolve_user_email(client, command["user_id"])

    if is_help_text(text):
        await respond(text=help_text_for(mode), response_type="ephemeral")
        return

    try:
        result, _created = await agent_loop(
            [{"role": "user", "content": text}],
            email,
            command["user_id"],
            channel_id=channel,
            mode=mode,
        )
    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        result = f"Something went wrong: {e}"
    await respond(text=to_slack_mrkdwn(result), response_type="in_channel")


slack_app.command("/cs")(ack=slash_ack, lazy=[slash_lazy])


# ---------------------------------------------------------------------------
# @-mention handler
# ---------------------------------------------------------------------------


async def mention_ack(ack):
    await ack()


async def mention_lazy(event, client):
    channel = event["channel"]
    mode = resolve_mode(channel)
    if mode is None:
        logger.info("Mention in unconfigured channel %s — ignoring", channel)
        return

    reply_ts = event.get("thread_ts") or event["ts"]

    text = strip_bot_mention(event.get("text", "") or "")
    files = event.get("files") or []

    # In plane mode, pick up files dropped earlier in the thread (so users can
    # mention us in a follow-up). In chatbot mode, attachments are ignored
    # entirely so don't bother scanning.
    if mode == "plane" and not files and event.get("thread_ts"):
        files = await collect_thread_files(client, channel, event["thread_ts"])
        if files:
            logger.info("Collected %d file(s) from thread context", len(files))

    logger.info(
        "Mention received: user=%s channel=%s mode=%s text=%r files=%d",
        event.get("user"), channel, mode, text[:120], len(files),
    )

    if not text and not files:
        await client.chat_postMessage(
            channel=channel,
            thread_ts=reply_ts,
            text="Mention me with an instruction. Try `@chatops help`.",
        )
        return

    await handle_user_request(
        client,
        channel=channel,
        user_id=event["user"],
        text=text,
        files=files,
        thread_ts=event.get("thread_ts"),
        reply_ts=reply_ts,
        mode=mode,
    )


slack_app.event("app_mention")(ack=mention_ack, lazy=[mention_lazy])


# ---------------------------------------------------------------------------
# DM handler — `message` event filtered to channel_type=im
# ---------------------------------------------------------------------------


async def dm_ack(ack):
    await ack()


async def dm_lazy(event, client):
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id") or event.get("subtype"):
        return  # ignore bots and message edits/joins/etc
    bot_uid = await get_bot_user_id(client)
    if event.get("user") == bot_uid:
        return

    channel = event["channel"]
    mode = resolve_mode(channel)
    if mode is None:
        logger.info("DM mode not configured (no instructions/{plane,chatbot}/dm.md) — ignoring")
        return

    text = (event.get("text") or "").strip()
    files = event.get("files") or []
    if not text and not files:
        return

    logger.info(
        "DM received: user=%s mode=%s text=%r files=%d",
        event.get("user"), mode, text[:120], len(files),
    )

    await handle_user_request(
        client,
        channel=channel,
        user_id=event["user"],
        text=text,
        files=files,
        thread_ts=event.get("thread_ts"),
        reply_ts=event.get("thread_ts"),  # keep flat unless already in a thread
        mode=mode,
    )


slack_app.event("message")(ack=dm_ack, lazy=[dm_lazy])


# ---------------------------------------------------------------------------
# Reaction-driven status changes — only in Plane-mode channels
# ---------------------------------------------------------------------------


EMOJI_TO_STATE_GROUP: dict[str, str] = {
    "white_check_mark": "completed",
    "heavy_check_mark": "completed",
    "construction": "started",
    "hammer_and_wrench": "started",
    "back": "backlog",
    "x": "cancelled",
    "no_entry_sign": "cancelled",
}

# Match a Plane issue URL (works for both the encoded and decoded forms Slack stores)
_ISSUE_URL_RE = re.compile(
    r"plane\.remotestar\.io/[^/\s>|]+/projects/([0-9a-f-]{36})/issues/([0-9a-f-]{36})",
    re.IGNORECASE,
)


async def reaction_ack(ack):
    await ack()


async def reaction_lazy(event, client):
    emoji = event.get("reaction")
    target_group = EMOJI_TO_STATE_GROUP.get(emoji)
    if not target_group:
        return

    item = event.get("item") or {}
    if item.get("type") != "message":
        return
    channel = item.get("channel")
    msg_ts = item.get("ts")
    if not channel or not msg_ts:
        return

    # Only act in plane-mode channels (DMs included if dm.md is plane).
    if resolve_mode(channel) != "plane":
        return

    try:
        history = await client.conversations_history(
            channel=channel, latest=msg_ts, limit=1, inclusive=True
        )
        msgs = history.get("messages") or []
        if not msgs:
            return
        msg = msgs[0]
    except Exception as e:
        logger.warning("Could not fetch reacted message: %s", e)
        return

    bot_uid = await get_bot_user_id(client)
    if msg.get("user") != bot_uid and not msg.get("bot_id"):
        return  # only act on our own messages

    text = msg.get("text") or ""
    m = _ISSUE_URL_RE.search(text)
    if not m:
        return
    project_id, issue_id = m.group(1), m.group(2)

    state_id = pick_state_for_group(project_id, target_group)
    if not state_id:
        logger.warning("No %s state cached for project %s", target_group, project_id)
        return

    logger.info(
        "Reaction :%s: from %s -> set state group %s on %s/%s",
        emoji, event.get("user"), target_group, project_id, issue_id,
    )
    result = await mcp.call("plane__update_work_item", {
        "project_id": project_id,
        "work_item_id": issue_id,
        "state": state_id,
    })
    state_name = (plane_states_cache.get(state_id) or {}).get("name") or target_group
    try:
        json.loads(result)  # if it parses, the update worked
        await client.chat_postMessage(
            channel=channel,
            thread_ts=msg_ts,
            text=f"Marked as *{state_name}* via :{emoji}: from <@{event.get('user')}>.",
        )
    except (json.JSONDecodeError, TypeError):
        logger.warning("update_work_item returned unexpected result: %s", result[:200])


slack_app.event("reaction_added")(ack=reaction_ack, lazy=[reaction_lazy])

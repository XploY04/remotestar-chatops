"""Slack listeners: app_mention, message (DM), slash command, reaction_added,
canvas_updated."""

from __future__ import annotations

import base64
import json
import re

import aiohttp
from app.agent import agent_loop
from app.config import logger, settings
from app.instructions import (
    get_channel_id_for_canvas,
    refresh_canvas,
    resolve_mode,
)
from app.plane import (
    attach_slack_files_to_plane_issue,
    looks_like_uuid,
    mcp,
    pick_state_for_group,
    plane_states_cache,
)
from app.prompts import help_text_for, is_help_text, to_slack_mrkdwn
from app.slack_app import slack_app


async def resolve_user_email(client, user_id: str) -> str:
    try:
        info = await client.users_info(user=user_id)
        return info["user"]["profile"].get("email") or "unknown@remotestar.io"
    except Exception as e:
        logger.warning("Failed to resolve user email for %s: %s", user_id, e)
        return "unknown@remotestar.io"


async def resolve_slack_mentions(client, text: str) -> str:
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


async def download_image_as_base64(file_url: str, bot_token: str) -> tuple[str, str] | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                file_url,
                headers={"Authorization": f"Bearer {bot_token}"}
            ) as resp:
                if resp.status != 200:
                    logger.warning("Image download failed: HTTP %s", resp.status)
                    return None
                data = await resp.read()
                mime_type = resp.content_type or "image/png"
                b64 = base64.b64encode(data).decode("utf-8")
                return b64, mime_type
    except Exception as e:
        logger.warning("Image download exception: %s", e, exc_info=True)
        return None


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
    if mode in ("plane", "mixpanel") and is_help_text(text) and not files:
        await client.chat_postMessage(channel=channel, thread_ts=reply_ts, text=help_text_for(mode))
        return

    text = await resolve_slack_mentions(client, text)

    # Vision: detect image and download as base64
    image_data: dict | None = None
    if files:
        for f in files:
            mime = f.get("mimetype", "")
            if mime.startswith("image/"):
                url = f.get("url_private_download") or f.get("url_private")
                if url:
                    result = await download_image_as_base64(url, settings.slack_bot_token)
                    if result:
                        image_data = {"b64": result[0], "mime": result[1]}
                        logger.info("Vision image ready: %s", f.get("name"))
                        break

    if files and mode == "plane" and not image_data:
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
        result, created_issue = await agent_loop(
            history, email, user_id,
            channel_id=channel,
            mode=mode,
            image_data=image_data,
        )
    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        result = f"Something went wrong: {e}"

    if files and mode == "plane" and not image_data:
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
    if mode in ("plane", "mixpanel") and is_help_text(text):
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



async def handle_recap_request(client, *, channel: str, days: int, reply_ts: str) -> None:
    import time
    import openai
    oldest = str(time.time() - days * 24 * 60 * 60)
    try:
        result = await client.conversations_history(channel=channel, oldest=oldest, limit=500)
        msgs = result.get("messages") or []
        lines = []
        for m in reversed(msgs):
            if m.get("bot_id") or m.get("subtype"):
                continue
            msg_text = (m.get("text") or "").strip()
            if msg_text:
                lines.append(msg_text)
        if not lines:
            await client.chat_postMessage(channel=channel, thread_ts=reply_ts,
                text=f"No messages found in the last {days} days.")
            return
        full_text = "\n".join(lines)
        prompt = (
            "Here are the last " + str(days) + " days of messages from a Slack channel.\n"
            "Summarize the key highlights in clear bullet points. Be concise.\n\n"
            + full_text[:12000]
        )
        oai = openai.AsyncOpenAI()
        response = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        summary = response.choices[0].message.content
        await client.chat_postMessage(
            channel=channel,
            thread_ts=reply_ts,
            text=":brain: *Last " + str(days) + " days recap:*\n\n" + summary
        )
    except Exception as e:
        logger.warning("Recap request failed: %s", e, exc_info=True)
        await client.chat_postMessage(channel=channel, thread_ts=reply_ts,
            text="Sorry, failed to generate recap: " + str(e))


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
    if mode == "plane" and not files and event.get("thread_ts"):
        files = await collect_thread_files(client, channel, event["thread_ts"])
        if files:
            logger.info("Collected %d file(s) from thread context", len(files))
    logger.info("Mention received: user=%s channel=%s mode=%s text=%r files=%d",
        event.get("user"), channel, mode, text[:120], len(files))
    if not text and not files:
        await client.chat_postMessage(channel=channel, thread_ts=reply_ts,
            text="Mention me with an instruction. Try `@chatops help`.")
        return

    # Recap intent — ask user for time range
    recap_keywords = ["recap", "summarize", "summary", "what happened", "last week", "catch me up"]
    if any(kw in text.lower() for kw in recap_keywords) and "days" not in text.lower():
        await client.chat_postMessage(
            channel=channel,
            thread_ts=reply_ts,
            text="Sure! Would you like a recap for the last *7 days*, *10 days*, or *20 days*? Just reply with the number."
        )
        return

    # User replied with a number of days for recap
    days_match = re.search(r"\b(7|10|20)\b", text)
    if days_match:
        days = int(days_match.group(1))
        await handle_recap_request(client, channel=channel, days=days, reply_ts=reply_ts)
        return

    await handle_user_request(client, channel=channel, user_id=event["user"],
        text=text, files=files, thread_ts=event.get("thread_ts"), reply_ts=reply_ts, mode=mode)


slack_app.event("app_mention")(ack=mention_ack, lazy=[mention_lazy])


async def dm_ack(ack):
    await ack()


async def dm_lazy(event, client):
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id") or event.get("subtype"):
        return
    bot_uid = await get_bot_user_id(client)
    if event.get("user") == bot_uid:
        return
    channel = event["channel"]
    mode = resolve_mode(channel)
    if mode is None:
        logger.info("DM mode not configured — ignoring")
        return
    text = (event.get("text") or "").strip()
    files = event.get("files") or []
    if not text and not files:
        return
    logger.info("DM received: user=%s mode=%s text=%r files=%d",
        event.get("user"), mode, text[:120], len(files))
    await handle_user_request(client, channel=channel, user_id=event["user"],
        text=text, files=files, thread_ts=event.get("thread_ts"),
        reply_ts=event.get("thread_ts"), mode=mode)


slack_app.event("message")(ack=dm_ack, lazy=[dm_lazy])


EMOJI_TO_STATE_GROUP: dict[str, str] = {
    "white_check_mark": "completed",
    "heavy_check_mark": "completed",
    "construction": "started",
    "hammer_and_wrench": "started",
    "back": "backlog",
    "x": "cancelled",
    "no_entry_sign": "cancelled",
}

_ISSUE_URL_RE = re.compile(
    r"plane\.remotestar\.io/[^/\s>|]+/projects/([0-9a-f-]{36})/issues/([0-9a-f-]{36})",
    re.IGNORECASE,
)

# n8n Company Brain webhook URL
BRAIN_WEBHOOK_URL = "https://aadi1974.app.n8n.cloud/webhook/1ad2e233-fc45-4d22-93da-dd2fdff4acaa"


async def reaction_ack(ack):
    await ack()


async def reaction_lazy(event, client):
    emoji = event.get("reaction")
    item = event.get("item") or {}
    if item.get("type") != "message":
        return
    channel = item.get("channel")
    msg_ts = item.get("ts")
    if not channel or not msg_ts:
        return

    # 🧠 Company Brain — send to n8n webhook
    if emoji == "brain":
        # Ignore reactions from the bot itself to prevent duplicates
        bot_uid = await get_bot_user_id(client)
        if event.get("user") == bot_uid:
            return
        try:
            import time
            oldest = str(time.time() - 20 * 24 * 60 * 60)  # 20 days ago
            history = await client.conversations_history(
                channel=channel, oldest=oldest, limit=500
            )
            msgs = history.get("messages") or []
            # Filter out bot messages and empty texts
            lines = []
            for m in reversed(msgs):  # oldest first
                if m.get("bot_id") or m.get("subtype"):
                    continue
                text = (m.get("text") or "").strip()
                if text:
                    lines.append(text)
            if lines:
                full_text = "\n".join(lines)
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        BRAIN_WEBHOOK_URL,
                        json={"text": full_text, "channel": channel, "ts": msg_ts}
                    )
                logger.info("Sent last 20 days (%d messages) to Company Brain n8n webhook", len(lines))
                # Confirm to the user
                await client.chat_postMessage(
                    channel=channel,
                    text=f"🧠 Saving last 20 days of this channel to Company Brain... ({len(lines)} messages)"
                )
        except Exception as e:
            logger.warning("Brain webhook failed: %s", e, exc_info=True)
        return

    # Plane status changes
    target_group = EMOJI_TO_STATE_GROUP.get(emoji)
    if not target_group:
        return
    if resolve_mode(channel) != "plane":
        return
    try:
        history = await client.conversations_history(channel=channel, latest=msg_ts, limit=1, inclusive=True)
        msgs = history.get("messages") or []
        if not msgs:
            return
        msg = msgs[0]
    except Exception as e:
        logger.warning("Could not fetch reacted message: %s", e)
        return
    bot_uid = await get_bot_user_id(client)
    if msg.get("user") != bot_uid and not msg.get("bot_id"):
        return
    text = msg.get("text") or ""
    m = _ISSUE_URL_RE.search(text)
    if not m:
        return
    project_id, issue_id = m.group(1), m.group(2)
    state_id = pick_state_for_group(project_id, target_group)
    if not state_id:
        logger.warning("No %s state cached for project %s", target_group, project_id)
        return
    result = await mcp.call("plane__update_work_item", {"project_id": project_id, "work_item_id": issue_id, "state": state_id})
    state_name = (plane_states_cache.get(state_id) or {}).get("name") or target_group
    try:
        json.loads(result)
        await client.chat_postMessage(channel=channel, thread_ts=msg_ts,
            text=f"Marked as *{state_name}* via :{emoji}: from <@{event.get('user')}>.")
    except (json.JSONDecodeError, TypeError):
        logger.warning("update_work_item returned unexpected result: %s", result[:200])


slack_app.event("reaction_added")(ack=reaction_ack, lazy=[reaction_lazy])


async def canvas_ack(ack):
    await ack()


async def canvas_lazy(event, client):
    canvas_id = event.get("canvas_id")
    if not canvas_id:
        return
    channel_id = get_channel_id_for_canvas(canvas_id)
    if not channel_id:
        logger.info("Canvas %s updated but not mapped to any channel — ignoring", canvas_id)
        return
    logger.info("Canvas %s updated — refreshing context for channel %s", canvas_id, channel_id)
    try:
        await refresh_canvas(channel_id, settings.slack_bot_token)
    except Exception as e:
        logger.warning("Failed to refresh canvas %s for channel %s: %s", canvas_id, channel_id, e)


slack_app.event("canvas_updated")(ack=canvas_ack, lazy=[canvas_lazy])

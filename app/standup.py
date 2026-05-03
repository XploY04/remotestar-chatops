"""Daily standup cron — DMs each Plane workspace member their pending tickets
(Todo + In Progress) at settings.standup_hour_utc."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from app.config import logger, settings
from app.plane import chatops_list_assigned_tickets, plane_members_cache
from app.slack_app import slack_app


async def run_standup_once() -> None:
    """For each cached Plane member, DM them their pending tickets."""
    if not plane_members_cache:
        logger.info("Standup skipped: no cached members")
        return
    sent = 0
    for member in plane_members_cache:
        email = member.get("email")
        if not email:
            continue
        try:
            lookup = await slack_app.client.users_lookupByEmail(email=email)
        except Exception as e:
            logger.info("Standup: no Slack account for %s (%s)", email, e)
            continue
        slack_user_id = (lookup.get("user") or {}).get("id")
        if not slack_user_id:
            continue

        result_text = await chatops_list_assigned_tickets({
            "assignee_email": email,
            "state_groups": ["unstarted", "started"],
        })
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            continue
        tickets = data.get("tickets") or []
        if not tickets:
            continue

        lines = [f"*Daily standup* — you have {len(tickets)} pending ticket(s):"]
        for t in tickets:
            state = t.get("state") or "?"
            lines.append(
                f"• *{t['key']}* — <{t['url']}|{t.get('name') or '(no title)'}> _({state})_"
            )
        msg = "\n".join(lines)
        try:
            await slack_app.client.chat_postMessage(channel=slack_user_id, text=msg)
            sent += 1
        except Exception as e:
            logger.warning("Standup: failed to DM %s: %s", email, e)
    logger.info("Standup: posted to %d user(s)", sent)


async def standup_loop() -> None:
    """Sleep until the next standup_hour_utc, run, repeat."""
    target_hour = settings.standup_hour_utc
    if not (0 <= target_hour <= 23):
        return
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        sleep_secs = (next_run - now).total_seconds()
        logger.info("Standup scheduled for %s UTC (in %.0f s)", next_run.isoformat(), sleep_secs)
        try:
            await asyncio.sleep(sleep_secs)
        except asyncio.CancelledError:
            return
        try:
            await run_standup_once()
        except Exception as e:
            logger.error("Standup run failed: %s", e, exc_info=True)

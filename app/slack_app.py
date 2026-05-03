"""The shared AsyncApp instance. Lives in its own module so handlers and the
standup cron can import it without any circular dependency on main."""

from __future__ import annotations

from slack_bolt.async_app import AsyncApp

from app.config import settings


slack_app = AsyncApp(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    process_before_response=True,
)

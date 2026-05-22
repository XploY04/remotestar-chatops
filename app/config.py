"""Settings, loaded from .env via pydantic-settings, plus the shared logger."""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    port: int = 9001
    log_level: str = "INFO"

    # Default mode for channels (and DMs) that don't have a specific
    # instructions file. "plane", "chatbot", or "mixpanel" turns on the
    # fallback so the bot responds in any channel it's invited to. Unset
    # (default) keeps the original strict opt-in behavior.
    default_channel_mode: str | None = None

    # Hosted Mixpanel MCP server, proxied via `npx mcp-remote`. Requires
    # Node 20 at /opt/node20/bin and one-time OAuth bootstrap that writes
    # credentials to ~/.mcp-auth. See README "Adding Mixpanel".
    mixpanel_mcp_enabled: bool = False
    mixpanel_mcp_url: str = "https://mcp.mixpanel.com/mcp"
    # Where Node 20 is installed on the host. The system `node` on rs is
    # still on 18 for other services; this lets only the mcp-remote
    # subprocess use a newer Node without affecting them.
    node20_bin: str = "/opt/node20/bin"

    # Daily standup cron: hour (UTC) at which to DM each member their pending tickets.
    # 0..23 enables, -1 disables.
    standup_hour_utc: int = -1


settings = Settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger("chatops")

PLANE_HOST = settings.plane_base_url.rstrip("/")

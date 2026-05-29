"""Settings, loaded from .env via pydantic-settings, plus the shared logger."""
from __future__ import annotations
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    slack_bot_token: str
    slack_signing_secret: str
    slack_user_token: str | None = None
    openai_api_key: str
    pinecone_api_key: str | None = None
    plane_api_key: str
    plane_workspace_slug: str = "remotestar"
    plane_base_url: str = "https://plane.remotestar.io/api"
    plane_project_recruiter: str
    plane_project_candidate: str
    mongodb_uri: str | None = None
    mongodb_database: str = "remotestar_candidate"
    port: int = 9001
    log_level: str = "INFO"
    default_channel_mode: str | None = None
    mixpanel_mcp_enabled: bool = False
    mixpanel_mcp_url: str = "https://mcp.mixpanel.com/mcp"
    node20_bin: str = "/opt/node20/bin"
    standup_hour_utc: int = -1


settings = Settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger("chatops")
PLANE_HOST = settings.plane_base_url.rstrip("/")

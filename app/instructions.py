"""Per-channel instructions, loaded from the `instructions/` directory.

Layout:

    instructions/
    ├── plane/
    │   ├── <channel_id>.md        # one markdown file per Plane-mode channel
    │   └── dm.md                  # if present, DMs run in plane mode
    └── chatbot/
        ├── <channel_id>.md        # one markdown file per chatbot-mode channel
        └── dm.md                  # if present, DMs run in chatbot mode

Mode is implied by the parent directory. Filename is the Slack channel ID with
a `.md` extension; the file's contents are appended verbatim to the system
prompt as that channel's custom context.

Default behavior is strict opt-in: a channel without a file gets no response.
Setting DEFAULT_CHANNEL_MODE=chatbot (or plane) in the env switches on a
fallback so any channel the bot is invited to gets handled with that mode
when no specific file is present. Per-channel files always win over the
fallback.

Canvas integration:
    Certain channels can be linked to one or more Slack canvases. Canvas
    content is fetched at startup and refreshed every 12 hours via polling.
    Content is persisted to MongoDB so it survives bot restarts instantly.
"""

from __future__ import annotations

import aiohttp
from pathlib import Path

from app.config import logger, settings


# Repo root → /instructions  (this module lives at /app/instructions.py)
INSTRUCTIONS_DIR = Path(__file__).resolve().parent.parent / "instructions"

PLANE_DIR = INSTRUCTIONS_DIR / "plane"
CHATBOT_DIR = INSTRUCTIONS_DIR / "chatbot"

# In-memory cache populated by load_instructions(): channel_id -> (mode, body)
_cache: dict[str, tuple[str, str]] = {}
# Resolved DM mode: "plane" / "chatbot" / None
_dm_mode: str | None = None
_dm_body: str = ""
# Optional fallback for channels (and DMs) with no specific file.
_default_mode: str | None = None

# ---------------------------------------------------------------------------
# Canvas integration
# ---------------------------------------------------------------------------

# Maps channel_id -> list of (canvas_id, cached_content)
# Each channel can have multiple canvases.
_canvas_cache: dict[str, list[tuple[str, str]]] = {
    "C0846QDN39D": [
        ("F0B2DDMBKGB", ""),   # BD Messaging Library canvas
    ],
    "C09LJEUACG0": [
        ("F0B2WQBFBJA", ""),   # Marketing canvas 1
        ("F0AM8THAM2A", ""),   # Marketing canvas 2
        ("F0APDRL8C8H", ""),   # Marketing canvas 3
    ],
    "C02MH67ENV8": [
        ("F05UJRK9BAS", ""),   # Internal team canvas 1
        ("F06PL5X1FLJ", ""),   # Internal team canvas 2
        ("F09K18T8LQ5", ""),   # Internal team canvas 3
        ("F0AFN7A34L8", ""),   # Internal team canvas 4
        ("F09JFD22UKW", ""),   # Internal team canvas 5
    ],
}


async def fetch_canvas_content(canvas_id: str, bot_token: str) -> str:
    """Fetch canvas content from Slack API and return as plain text."""
    url = "https://slack.com/api/canvases.sections.lookup"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "canvas_id": canvas_id,
        "criteria": {
            "section_types": [
                "any_header",
                "bulleted_list",
                "numbered_list",
                "paragraph",
                "table",
                "code",
            ]
        },
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.warning(
                        "Canvas fetch failed for %s: %s", canvas_id, data.get("error")
                    )
                    return ""
                sections = data.get("sections", [])
                content = "\n\n".join(
                    s.get("content", {}).get("markdown", "")
                    for s in sections
                    if s.get("content", {}).get("markdown", "").strip()
                )
                return content.strip()
    except Exception as e:
        logger.warning("Canvas fetch exception for %s: %s", canvas_id, e, exc_info=True)
        return ""


async def refresh_canvas(channel_id: str, bot_token: str) -> None:
    """Re-fetch all canvas content for a channel, update memory + MongoDB."""
    from app.audit import save_canvas_content
    if channel_id not in _canvas_cache:
        return
    canvases = _canvas_cache[channel_id]
    updated = []
    for canvas_id, old_content in canvases:
        content = await fetch_canvas_content(canvas_id, bot_token)
        if content:
            updated.append((canvas_id, content))
            # Save to MongoDB for persistence across restarts
            await save_canvas_content(channel_id, canvas_id, content)
            logger.info(
                "Canvas refreshed for channel %s (canvas_id=%s, %d chars)",
                channel_id, canvas_id, len(content),
            )
        else:
            # Keep old content if fetch failed
            updated.append((canvas_id, old_content))
            logger.warning(
                "Canvas fetch failed for channel %s (canvas_id=%s) — keeping old content",
                channel_id, canvas_id,
            )
    _canvas_cache[channel_id] = updated


async def preload_canvas_from_db() -> None:
    """At startup: load all canvas content from MongoDB into memory cache.
    This means the bot has canvas knowledge instantly on restart
    without waiting for the next Slack API fetch."""
    from app.audit import load_all_canvas_content
    db_content = await load_all_canvas_content()
    for channel_id, canvases in _canvas_cache.items():
        channel_db = db_content.get(channel_id, {})
        updated = []
        for canvas_id, _ in canvases:
            saved_content = channel_db.get(canvas_id, "")
            updated.append((canvas_id, saved_content))
            if saved_content:
                logger.info(
                    "Canvas preloaded from DB: channel=%s canvas=%s (%d chars)",
                    channel_id, canvas_id, len(saved_content),
                )
        _canvas_cache[channel_id] = updated


async def refresh_all_canvases(bot_token: str) -> None:
    """Fetch all mapped canvases from Slack API and persist to MongoDB."""
    for channel_id in list(_canvas_cache.keys()):
        await refresh_canvas(channel_id, bot_token)


def get_canvas_content(channel_id: str) -> str:
    """Return all cached canvas content for a channel combined."""
    canvases = _canvas_cache.get(channel_id)
    if not canvases:
        return ""
    parts = [content for _, content in canvases if content.strip()]
    return "\n\n---\n\n".join(parts)


def get_channel_id_for_canvas(canvas_id: str) -> str | None:
    """Return the channel_id mapped to a given canvas_id, or None."""
    for channel_id, canvases in _canvas_cache.items():
        for cid, _ in canvases:
            if cid == canvas_id:
                return channel_id
    return None


# ---------------------------------------------------------------------------
# Instructions loader
# ---------------------------------------------------------------------------


def load_instructions() -> None:
    """Walk the instructions directory and populate the cache."""
    global _dm_mode, _dm_body, _default_mode
    _cache.clear()
    _dm_mode = None
    _dm_body = ""
    _default_mode = None

    configured = (settings.default_channel_mode or "").strip().lower() or None
    if configured in ("plane", "chatbot"):
        _default_mode = configured
    elif configured:
        logger.warning(
            "DEFAULT_CHANNEL_MODE=%r is not 'plane' or 'chatbot'; ignoring",
            settings.default_channel_mode,
        )

    if not INSTRUCTIONS_DIR.exists():
        logger.warning(
            "instructions/ directory not found at %s — bot will ignore every channel",
            INSTRUCTIONS_DIR,
        )
        return

    plane_dm = PLANE_DIR / "dm.md"
    chatbot_dm = CHATBOT_DIR / "dm.md"

    for mode, dir_path in (("plane", PLANE_DIR), ("chatbot", CHATBOT_DIR)):
        if not dir_path.exists():
            continue
        for md_file in dir_path.iterdir():
            if not md_file.is_file() or md_file.suffix != ".md":
                continue
            stem = md_file.stem
            if stem == "dm":
                continue
            body = md_file.read_text(encoding="utf-8").strip()
            if stem in _cache:
                logger.warning(
                    "Channel %s has files in both plane/ and chatbot/; ignoring %s",
                    stem, md_file,
                )
                continue
            _cache[stem] = (mode, body)

    if plane_dm.exists() and chatbot_dm.exists():
        logger.warning(
            "Both instructions/plane/dm.md and instructions/chatbot/dm.md exist; using plane/"
        )
    if plane_dm.exists():
        _dm_mode = "plane"
        _dm_body = plane_dm.read_text(encoding="utf-8").strip()
    elif chatbot_dm.exists():
        _dm_mode = "chatbot"
        _dm_body = chatbot_dm.read_text(encoding="utf-8").strip()

    summary = (
        ", ".join(f"{cid}={mode}" for cid, (mode, _) in _cache.items()) or "(none)"
    )
    logger.info(
        "Loaded %d channel instruction file(s): %s. DM mode: %s. Default fallback: %s",
        len(_cache),
        summary,
        _dm_mode or "ignored",
        _default_mode or "off",
    )


def reload_instructions() -> None:
    """Re-scan disk."""
    load_instructions()


def resolve_mode(channel_id: str | None) -> str | None:
    """Return 'plane' / 'chatbot' / None for a Slack channel ID."""
    if not channel_id:
        return None
    if channel_id.startswith("D"):
        return _dm_mode or _default_mode
    entry = _cache.get(channel_id)
    if entry:
        return entry[0]
    return _default_mode


def get_instructions(channel_id: str | None) -> str:
    """Return the channel's instruction body (markdown text).
    Combines .md file content + live canvas content if available."""
    if not channel_id:
        return ""

    # Get .md file content
    if channel_id.startswith("D"):
        md_content = _dm_body
    else:
        entry = _cache.get(channel_id)
        md_content = entry[1] if entry else ""

    # Append live canvas content if available
    canvas_content = get_canvas_content(channel_id)
    if canvas_content:
        return (
            f"{md_content}\n\n"
            f"## Live Canvas Content (auto-synced from Slack)\n"
            f"{canvas_content}"
        ).strip()

    return md_content

"""Per-channel instructions, loaded from the `instructions/` directory.

Layout:

    instructions/
    ├── plane/
    │   ├── <channel_id>.md        # one markdown file per Plane-mode channel
    │   └── dm.md                  # if present, DMs run in plane mode
    ├── chatbot/
    │   ├── <channel_id>.md        # one markdown file per chatbot-mode channel
    │   └── dm.md                  # if present, DMs run in chatbot mode
    └── mixpanel/
        ├── <channel_id>.md        # one markdown file per Mixpanel-mode channel
        └── dm.md                  # if present, DMs run in mixpanel mode

Mode is implied by the parent directory. Filename is the Slack channel ID with
a `.md` extension; the file's contents are appended verbatim to the system
prompt as that channel's custom context.

Default behavior is strict opt-in: a channel without a file gets no response.
Setting DEFAULT_CHANNEL_MODE=chatbot (or plane / mixpanel) in the env
switches on a fallback so any channel the bot is invited to gets handled
with that mode when no specific file is present. Per-channel files always
win over the fallback.

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
MIXPANEL_DIR = INSTRUCTIONS_DIR / "mixpanel"

# Modes recognised by the loader and DEFAULT_CHANNEL_MODE validator.
# Tuple order also defines dm.md resolution priority when multiple
# `<mode>/dm.md` files exist (plane > mixpanel > chatbot).
_MODE_DIRS: tuple[tuple[str, Path], ...] = (
    ("plane", PLANE_DIR),
    ("mixpanel", MIXPANEL_DIR),
    ("chatbot", CHATBOT_DIR),
)
_VALID_MODES = {m for m, _ in _MODE_DIRS}

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
    """Fetch canvas content from Slack API using files.info."""
    url = "https://slack.com/api/files.info"
    headers = {
        "Authorization": f"Bearer {bot_token}",
    }
    params = {"file": canvas_id}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.warning(
                        "Canvas fetch failed for %s: %s", canvas_id, data.get("error")
                    )
                    return ""
                file_data = data.get("file", {})
                content = (
                    file_data.get("plain_text") or
                    file_data.get("preview") or
                    ""
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
            await save_canvas_content(channel_id, canvas_id, content)
            logger.info(
                "Canvas refreshed for channel %s (canvas_id=%s, %d chars)",
                channel_id, canvas_id, len(content),
            )
        else:
            updated.append((canvas_id, old_content))
            logger.warning(
                "Canvas fetch failed for channel %s (canvas_id=%s) — keeping old content",
                channel_id, canvas_id,
            )
    _canvas_cache[channel_id] = updated


async def preload_canvas_from_db() -> None:
    """At startup: load all canvas content from MongoDB into memory cache."""
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
    if configured in _VALID_MODES:
        _default_mode = configured
    elif configured:
        logger.warning(
            "DEFAULT_CHANNEL_MODE=%r is not one of %s; ignoring",
            settings.default_channel_mode, sorted(_VALID_MODES),
        )

    if not INSTRUCTIONS_DIR.exists():
        logger.warning(
            "instructions/ directory not found at %s — bot will ignore every channel",
            INSTRUCTIONS_DIR,
        )
        return

    dm_files = [(mode, dir_path / "dm.md") for mode, dir_path in _MODE_DIRS]

    for mode, dir_path in _MODE_DIRS:
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
                    "Channel %s has files in multiple mode directories; "
                    "first match wins, ignoring %s",
                    stem, md_file,
                )
                continue
            _cache[stem] = (mode, body)

    # dm.md resolution: priority order from _MODE_DIRS (plane > mixpanel > chatbot).
    present_dms = [(mode, path) for mode, path in dm_files if path.exists()]
    if len(present_dms) > 1:
        winner_mode, _ = present_dms[0]
        rest = ", ".join(f"instructions/{m}/dm.md" for m, _ in present_dms[1:])
        logger.warning(
            "Multiple dm.md files present; using instructions/%s/dm.md, ignoring %s",
            winner_mode, rest,
        )
    if present_dms:
        _dm_mode, dm_path = present_dms[0]
        _dm_body = dm_path.read_text(encoding="utf-8").strip()

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
    """Return 'plane' / 'chatbot' / 'mixpanel' / None for a Slack channel ID."""
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

    if channel_id.startswith("D"):
        md_content = _dm_body
    else:
        entry = _cache.get(channel_id)
        md_content = entry[1] if entry else ""

    canvas_content = get_canvas_content(channel_id)
    if canvas_content:
        return (
            f"{md_content}\n\n"
            f"## Live Canvas Content (auto-synced from Slack)\n"
            f"{canvas_content}"
        ).strip()

    return md_content

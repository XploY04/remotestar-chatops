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

There are no defaults. If a channel has no file under either directory, the
bot will not respond there. If `dm.md` is missing under both, DMs are ignored.
"""

from __future__ import annotations

from pathlib import Path

from app.config import logger


# Repo root → /instructions  (this module lives at /app/instructions.py)
INSTRUCTIONS_DIR = Path(__file__).resolve().parent.parent / "instructions"

PLANE_DIR = INSTRUCTIONS_DIR / "plane"
CHATBOT_DIR = INSTRUCTIONS_DIR / "chatbot"

# In-memory cache populated by load_instructions(): channel_id -> (mode, body)
_cache: dict[str, tuple[str, str]] = {}
# Resolved DM mode: "plane" / "chatbot" / None
_dm_mode: str | None = None
_dm_body: str = ""


def load_instructions() -> None:
    """Walk the instructions directory and populate the cache."""
    global _dm_mode, _dm_body
    _cache.clear()
    _dm_mode = None
    _dm_body = ""

    if not INSTRUCTIONS_DIR.exists():
        logger.warning("instructions/ directory not found at %s — bot will ignore every channel", INSTRUCTIONS_DIR)
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
                continue  # handled below
            body = md_file.read_text(encoding="utf-8").strip()
            if stem in _cache:
                # Same channel ID configured in both modes — refuse to guess.
                logger.warning(
                    "Channel %s has files in both plane/ and chatbot/; ignoring %s",
                    stem, md_file,
                )
                continue
            _cache[stem] = (mode, body)

    # DM resolution: prefer plane/dm.md if both exist.
    if plane_dm.exists() and chatbot_dm.exists():
        logger.warning("Both instructions/plane/dm.md and instructions/chatbot/dm.md exist; using plane/")
    if plane_dm.exists():
        _dm_mode = "plane"
        _dm_body = plane_dm.read_text(encoding="utf-8").strip()
    elif chatbot_dm.exists():
        _dm_mode = "chatbot"
        _dm_body = chatbot_dm.read_text(encoding="utf-8").strip()

    summary = ", ".join(f"{cid}={mode}" for cid, (mode, _) in _cache.items()) or "(none)"
    logger.info(
        "Loaded %d channel instruction file(s): %s. DM mode: %s",
        len(_cache), summary, _dm_mode or "ignored",
    )


def reload_instructions() -> None:
    """Re-scan disk. Wire to a slash command or admin endpoint if hot-reload is needed."""
    load_instructions()


def resolve_mode(channel_id: str | None) -> str | None:
    """Return 'plane' / 'chatbot' / None for a Slack channel ID. None means
    the bot should ignore this channel entirely."""
    if not channel_id:
        return None
    if channel_id.startswith("D"):
        return _dm_mode
    entry = _cache.get(channel_id)
    return entry[0] if entry else None


def get_instructions(channel_id: str | None) -> str:
    """Return the channel's instruction body (markdown text). Empty string if
    the channel has no file."""
    if not channel_id:
        return ""
    if channel_id.startswith("D"):
        return _dm_body
    entry = _cache.get(channel_id)
    return entry[1] if entry else ""

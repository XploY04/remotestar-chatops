"""
app/gmail_digest.py
Read-only access to the shared RemoteStar inbox (Google Workspace).
Auth: service account with domain-wide delegation, scope gmail.readonly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from email.utils import parseaddr
from html import unescape

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config import logger, settings

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SA_KEY_FILE = settings.google_sa_key_file
SHARED_ADDRESS = settings.gmail_shared_address

BODY_CHAR_LIMIT = 1500
DEFAULT_MAX_EMAILS = 25

_TAG_RE = re.compile(r"<[^>]+>")
_ON_WROTE_RE = re.compile(r"\nOn .{0,160}? wrote:.*", re.DOTALL)
_QUOTE_MARKERS = ("-----Original Message-----", "________________________________")


def _service():
    if not SA_KEY_FILE or not SHARED_ADDRESS:
        raise RuntimeError(
            "GOOGLE_SA_KEY_FILE and GMAIL_SHARED_ADDRESS must be set to read the inbox."
        )
    creds = service_account.Credentials.from_service_account_file(
        SA_KEY_FILE, scopes=GMAIL_SCOPES, subject=SHARED_ADDRESS
    )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", "replace")


def _extract_body(payload: dict) -> str:
    plain, html = "", ""

    def walk(part: dict) -> None:
        nonlocal plain, html
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if mime == "text/plain" and data and not plain:
            plain = _decode(data)
        elif mime == "text/html" and data and not html:
            html = _decode(data)
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    return plain or (_TAG_RE.sub(" ", unescape(html)) if html else "")


def _clean(text: str) -> str:
    text = _ON_WROTE_RE.sub("", text)
    for marker in _QUOTE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) > BODY_CHAR_LIMIT:
        text = text[:BODY_CHAR_LIMIT].rstrip() + " …[truncated]"
    return text


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _fetch_sync(since_minutes: int, max_emails: int) -> list[dict]:
    svc = _service()
    hours = max(1, round(since_minutes / 60))
    query = f"in:inbox newer_than:{hours}h"
    resp = (
        svc.users().messages()
        .list(userId="me", q=query, maxResults=max_emails)
        .execute()
    )
    out: list[dict] = []
    for m in resp.get("messages", []):
        msg = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        name, addr = parseaddr(_header(headers, "From"))
        out.append({
            "from_name": name or addr or "(unknown sender)",
            "from_email": addr,
            "subject": _header(headers, "Subject") or "(no subject)",
            "date": _header(headers, "Date"),
            "body": _clean(_extract_body(payload)),
        })
    return out


async def fetch_recent_emails(
    since_minutes: int = 60, max_emails: int = DEFAULT_MAX_EMAILS
) -> str:
    max_emails = min(max_emails, DEFAULT_MAX_EMAILS)
    try:
        emails = await asyncio.to_thread(_fetch_sync, since_minutes, max_emails)
    except Exception as e:
        logger.exception("inbox fetch failed")
        return json.dumps({"error": f"Could not read the shared inbox: {e}"})
    return json.dumps({
        "mailbox": SHARED_ADDRESS,
        "window_minutes": since_minutes,
        "count": len(emails),
        "emails": emails,
    })

"""
memory.py — Save important Slack messages to Pinecone (Company Brain).

Usage:
    from app.memory import save_message_to_brain

    # Call this when someone reacts with 🧠 to a Slack message
    await save_message_to_brain(text="BD strategy: focus on fintech...", source="slack")
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from openai import AsyncOpenAI
from pinecone import Pinecone

from app.config import logger, settings

# ── Clients ──────────────────────────────────────────────────────────────────

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

# Add PINECONE_API_KEY and PINECONE_INDEX to your .env / settings
pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index)  # e.g. "remotestar-brain"


# ── Main function ─────────────────────────────────────────────────────────────


async def save_message_to_brain(text: str, source: str = "slack") -> bool:
    """
    Summarize a message with GPT, embed it, and store in Pinecone.
    Returns True on success, False on failure.
    """
    if not text or not text.strip():
        return False

    try:
        # Step 1: Summarize the message
        summary_resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a note-taking assistant for RemoteStar. "
                        "Summarize the following Slack message into 1–2 clear sentences. "
                        "Extract the key decision, strategy, or information. "
                        "Be concise. Do not add commentary."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        summary = summary_resp.choices[0].message.content or text

        # Step 2: Embed the summary
        embed_resp = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=summary,
        )
        vector = embed_resp.data[0].embedding

        # Step 3: Store in Pinecone
        doc_id = hashlib.md5(text.encode()).hexdigest()  # stable ID per unique message
        index.upsert(vectors=[{
            "id": doc_id,
            "values": vector,
            "metadata": {
                "summary": summary,
                "original": text[:500],  # keep original truncated for reference
                "source": source,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        }])

        logger.info("Saved to brain: %s (source=%s)", summary[:80], source)
        return True

    except Exception as e:
        logger.warning("Failed to save message to brain: %s", e, exc_info=True)
        return False

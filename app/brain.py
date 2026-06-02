"""
brain.py — Search the Company Brain (Pinecone) for relevant knowledge.

Usage:
    from app.brain import search_brain

    results = await search_brain("what is our BD strategy?")
    # Returns a formatted string ready to inject into the system prompt
"""

from __future__ import annotations

from openai import AsyncOpenAI
from pinecone import Pinecone

from app.config import logger, settings

# ── Clients ───────────────────────────────────────────────────────────────────

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index)


# ── Main function ─────────────────────────────────────────────────────────────


async def search_brain(query: str, top_k: int = 5) -> str:
    """
    Search Pinecone for knowledge relevant to the query.
    Returns a formatted string to inject into Jarvis's context.
    Returns empty string if nothing found or on error.
    """
    if not query or not query.strip():
        return ""

    try:
        # Step 1: Embed the query
        embed_resp = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        vector = embed_resp.data[0].embedding

        # Step 2: Search Pinecone
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = results.get("matches") or []
        # Only use matches with reasonable similarity (score > 0.5)
        good_matches = [m for m in matches if m.get("score", 0) > 0.5]

        if not good_matches:
            return ""

        # Step 3: Format results as clean context
        lines = ["## Company Brain (from saved Slack knowledge)"]
        for i, match in enumerate(good_matches, 1):
            summary = match.get("metadata", {}).get("summary", "")
            saved_at = match.get("metadata", {}).get("saved_at", "")[:10]  # just the date
            if summary:
                lines.append(f"{i}. {summary} _(saved {saved_at})_")

        logger.info("Brain search returned %d results for: %s", len(good_matches), query[:60])
        return "\n".join(lines)

    except Exception as e:
        logger.warning("Brain search failed: %s", e, exc_info=True)
        return ""

"""LLM tool-calling loop. Mode-aware:
- "plane":    Plane MCP toolset + chatops__* local tools + a curated read-only
              subset of Mixpanel tools.
- "mixpanel": full Mixpanel MCP toolset (45 tools), no Plane.
- "chatbot":  no tools, single completion call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openai import AsyncOpenAI

from app.audit import audit_log
from app.config import logger, settings
from app.plane import LOCAL_TOOL_DEFS, LOCAL_TOOL_HANDLERS, mcp
from app.prompts import build_system_prompt


openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

OPENAI_TOOLS_MAX = 128
PLANE_MODE_MIXPANEL_SUBSET = frozenset({
    "mixpanel__Run-Query",
    "mixpanel__Get-Query-Schema",
    "mixpanel__Get-Report",
    "mixpanel__Display-Query",
    "mixpanel__Get-Projects",
    "mixpanel__Get-Events",
    "mixpanel__List-Properties",
    "mixpanel__Get-Property-Values",
    "mixpanel__Search-Entities",
    "mixpanel__Get-Business-Context",
    "mixpanel__Get-Issues",
    "mixpanel__List-Metrics",
    "mixpanel__Get-Metric",
    "mixpanel__List-Dashboards",
    "mixpanel__Get-Dashboard",
})


async def agent_loop(
    history: list[dict],
    user_email: str,
    user_slack_id: str,
    channel_id: str | None,
    mode: str,
    image_data: dict | None = None,  # vision support
) -> tuple[str, dict | None]:
    """history is a list of {role, content} dicts. Last item is the current user request.

    image_data: optional dict with keys 'b64' (base64) and 'mime' (mime type).
    When present, the last user message is sent to GPT-4o with the image attached.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    system = (
        build_system_prompt(channel_id, mode)
        + f"\n\n## Current request\n- User email: {user_email}"
        + f"\n- User Slack ID: {user_slack_id}\n- Timestamp: {now_iso}\n"
    )

    # ── Build messages with optional vision ───────────────────────────────────
    if image_data and history:
        last_text = history[-1].get("content") or "What is in this image? Describe it and answer any questions."
        vision_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime']};base64,{image_data['b64']}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": last_text
                }
            ]
        }
        messages: list[dict] = [
            {"role": "system", "content": system},
            *history[:-1],
            vision_message,
        ]
        vision_model = "gpt-4o"
        logger.info("Vision mode activated — using %s", vision_model)
    else:
        messages = [
            {"role": "system", "content": system},
            *history,
        ]
        vision_model = "gpt-4o-mini"
    # ─────────────────────────────────────────────────────────────────────────

    # Chatbot mode — no tools
    if mode == "chatbot":
        response = await openai_client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=0.7,
        )
        final = response.choices[0].message.content or "Done."
        await audit_log(user_slack_id, user_email, [], final)
        return final, None

    if mode == "mixpanel":
        tools = mcp.openai_tools(server="mixpanel")
        no_tools_msg = (
            "The Mixpanel backend isn't connected right now. "
            "Try again in a moment, or ping someone to re-run OAuth bootstrap."
        )
    else:
        plane_tools = mcp.openai_tools(server="plane")
        mixpanel_tools = [
            t for t in mcp.openai_tools(server="mixpanel")
            if t["function"]["name"] in PLANE_MODE_MIXPANEL_SUBSET
        ]
        tools = plane_tools + mixpanel_tools + LOCAL_TOOL_DEFS
        no_tools_msg = "I'm not connected to any backends right now. Try again in a moment."

    if len(tools) > OPENAI_TOOLS_MAX:
        logger.error(
            "Tools list exceeds OpenAI limit: have %d, max %d. Truncating.",
            len(tools), OPENAI_TOOLS_MAX,
        )
        tools = tools[:OPENAI_TOOLS_MAX]

    if not tools:
        return no_tools_msg, None

    tool_call_log: list[dict] = []
    created_issue: dict | None = None
    max_iterations = 8

    for _ in range(max_iterations):
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            final = msg.content or "Done."
            await audit_log(user_slack_id, user_email, tool_call_log, final)
            return final, created_issue

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            logger.info("Tool call: %s args=%s", tc.function.name, args)
            if tc.function.name in LOCAL_TOOL_HANDLERS:
                try:
                    result = await LOCAL_TOOL_HANDLERS[tc.function.name](args)
                except Exception as e:
                    logger.error("Local tool %s crashed: %s", tc.function.name, e, exc_info=True)
                    result = json.dumps({"error": str(e)})
            else:
                result = await mcp.call(tc.function.name, args)
            tool_call_log.append({"name": tc.function.name, "args": args, "result_preview": result[:200]})

            if tc.function.name == "plane__create_work_item":
                try:
                    data = json.loads(result)
                    if isinstance(data, dict) and data.get("id") and args.get("project_id"):
                        created_issue = {"project_id": args["project_id"], "issue_id": data["id"]}
                except (json.JSONDecodeError, TypeError):
                    pass
            elif tc.function.name == "plane__update_work_item":
                proj = args.get("project_id")
                wid = args.get("work_item_id") or args.get("issue_id")
                if proj and wid:
                    created_issue = {"project_id": proj, "issue_id": wid}
            elif tc.function.name == "plane__delete_work_item":
                wid = args.get("work_item_id") or args.get("issue_id")
                if created_issue and created_issue.get("issue_id") == wid:
                    logger.info("Cleared tracked issue %s because LLM deleted it", wid)
                    created_issue = None

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    fallback = "I tried but couldn't complete the request in a reasonable number of steps. Try rephrasing?"
    await audit_log(user_slack_id, user_email, tool_call_log, fallback)
    return fallback, created_issue

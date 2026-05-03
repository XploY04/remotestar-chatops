"""LLM tool-calling loop. Mode-aware:
- "plane": full MCP toolset + LOCAL_TOOL_DEFS, multi-turn tool loop.
- "chatbot": no tools, single completion call.
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


async def agent_loop(
    history: list[dict],
    user_email: str,
    user_slack_id: str,
    channel_id: str | None,
    mode: str,
) -> tuple[str, dict | None]:
    """history is a list of {role, content} dicts. Last item is the current user request.

    Returns (final_text, created_issue) where created_issue is
    {project_id, issue_id} from the most recent successful plane__create/update
    call (plane mode only). In chatbot mode, created_issue is always None.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    system = (
        build_system_prompt(channel_id, mode)
        + f"\n\n## Current request\n- User email: {user_email}"
        + f"\n- User Slack ID: {user_slack_id}\n- Timestamp: {now_iso}\n"
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        *history,
    ]

    # In chatbot mode, no tools at all — single completion call, return the reply.
    if mode == "chatbot":
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
        )
        final = response.choices[0].message.content or "Done."
        await audit_log(user_slack_id, user_email, [], final)
        return final, None

    # Plane mode: full tool loop.
    tools = mcp.openai_tools() + LOCAL_TOOL_DEFS
    if not tools:
        return "I'm not connected to any backends right now. Try again in a moment.", None

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

        # Append assistant's tool-calling message
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

        # Execute each tool call
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

            # Track the issue id so we can attach files after the loop. We watch
            # both create_work_item (new issue) and update_work_item (so files
            # can be attached to an existing ticket via the bot). delete clears it.
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

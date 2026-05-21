# RemoteStar ChatOps Bot

Slack bot for the RemoteStar team. In engineering channels it manages Plane tickets through natural language (built on the official Plane MCP server). In non-engineering channels it runs as a context-aware assistant with no tools. Designed to extend to GitHub and other services by registering more MCP servers.

## Examples

In a Plane-mode channel (e.g. engineering):

```
@chatops create a ticket: API returning 500s during signup, assign rudy@remotestar.io
@chatops list my open candidate tickets
@chatops mark RECRUITER-109 in progress
@chatops add a comment to CANDIDATE-45: looks good, ship it
```

In a chatbot-mode channel (e.g. PMM, BD, tech):

```
@chatops draft three subject lines for our launch email
@chatops summarize what's been said in this thread
@chatops who owns FDE outreach?  (answered from the BD-playbook channel brief)
```

Slash command works in either mode:

```
/cs create a ticket for the recruiter dashboard rendering bug
/cs help
```

## Architecture

```
Slack (@-mention, DM, /cs, reaction)
   │
   ▼  HTTPS via Cloudflare Tunnel
chatops.remotestar.io  →  cloudflared
   │
   ▼  POST /slack/events on localhost:9001
FastAPI + Slack Bolt (chatops.service, systemd)
   │
   ├─ OpenAI gpt-4o-mini   (tool calling; up to 8 iterations per request)
   │
   ├─ Plane MCP server     (subprocess via stdio: `uvx plane-mcp-server stdio`)
   │      │
   │      └─ Self-hosted Plane API (plane.remotestar.io)
   │
   ├─ Plane REST direct    (only for the 3-step S3 attachment upload;
   │                        the Plane MCP exposes no attachment tools)
   │
   └─ MongoDB              (Motor; chatops_audit collection, best-effort)
```

One Slack endpoint. The LLM picks tools from any registered MCP server. Adding GitHub later = adding `github-mcp-server` to `MCP_SERVERS` in `app/plane.py`. No new endpoints, no new code paths.

## Setup

### 1. Plane

- Open Plane → avatar → **Profile Settings → Personal Access Tokens**
- Create token named `chatops`, no expiry, copy it (shown once)
- Note the workspace slug from your Plane URL: `https://plane.your-domain.com/<slug>/projects/`
- Note both project UUIDs from the project URLs

### 2. Slack app

- Go to https://api.slack.com/apps → **Create New App** → **From scratch**
- App name: `Chatops`, workspace: RemoteStar
- After creation:
  - **Basic Information** → copy **Signing Secret**
  - **OAuth & Permissions** → add Bot Token Scopes:
    - `commands` (slash commands)
    - `chat:write` (post replies)
    - `users:read`, `users:read.email` (resolve @-mentions to Plane emails)
    - `app_mentions:read`, `channels:history` (mentions in public channels + thread history)
    - `im:history` (read direct messages so the bot can reply to DMs)
    - `reactions:read` (reaction-driven state changes on bot messages)
  - **Slash Commands** → Create:
    - Command: `/cs`
    - Request URL: `https://chatops.remotestar.io/slack/events` (we'll fill after Tunnel)
    - Description: `RemoteStar ChatOps`
    - Usage hint: `<natural language instruction>`
  - **Event Subscriptions** → Enable Events:
    - Request URL: `https://chatops.remotestar.io/slack/events`
    - Subscribe to bot events:
      - `app_mention` (@chatops in channels)
      - `message.im` (direct messages to the bot)
      - `reaction_added` (so :white_check_mark: etc. can update tickets)
  - **Install to Workspace** → copy **Bot Token** (`xoxb-...`)
- Get the channel ID for any channel you want the bot active in:
  - In Slack, right-click the channel name → **Copy link**. The ID is the last segment, like `C012ABCDE`.
  - Or run `conversations.list` via the Slack API once.
- Invite the bot: in each channel type `/invite @chatops`. The bot won't respond there until you also drop an instructions file (see below).

### 3. Cloudflare Tunnel

You need a public HTTPS URL for Slack to call. Cloudflare Tunnel handles this without opening firewall ports or managing TLS certs.

**Prerequisites:** a Cloudflare account with `remotestar.io` (or whichever domain) added.

```bash
# Install cloudflared on the VPS
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

# One-time browser auth
cloudflared tunnel login

# Create the tunnel
cloudflared tunnel create chatops

# Configure routing. Create /root/.cloudflared/config.yml:
cat > /root/.cloudflared/config.yml <<EOF
tunnel: chatops
credentials-file: /root/.cloudflared/<TUNNEL_UUID>.json

ingress:
  - hostname: chatops.remotestar.io
    service: http://localhost:9001
  - service: http_status:404
EOF

# Route DNS to the tunnel
cloudflared tunnel route dns chatops chatops.remotestar.io
```

After this, `chatops.remotestar.io` resolves to your tunnel and forwards to `localhost:9001` on the VPS. Update the Slack app URLs to use `https://chatops.remotestar.io/slack/events`.

### 4. Install + run on the VPS

```bash
ssh rs

# Install uv (provides uvx, used to launch the Plane MCP server)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart shell

# Clone repo
cd /root
git clone https://github.com/XploY04/remotestar-chatops
cd remotestar-chatops

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with all secrets (see .env.example)
nano .env

# Test locally first
python -m app
# Should print: "MCP server 'plane' ready with NN tools"
# Ctrl+C to stop

# Install systemd services
cp deploy/chatops.service /etc/systemd/system/
cp deploy/cloudflared.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable chatops cloudflared
systemctl start chatops cloudflared

# Check status
systemctl status chatops --no-pager
journalctl -u chatops -f  # tail logs
```

### 5. Verify

- `curl https://chatops.remotestar.io/health` → `{"status":"ok","mcp_servers":["plane"]}`
- Add an instructions file for one channel (see next section), then `systemctl restart chatops`.
- In that channel: `@chatops help`. The bot replies in-thread within 5 to 15 seconds.

## Channel modes and the `instructions/` directory

The bot runs in one of two modes per channel:

- **plane**: full Plane MCP toolset, attachment uploads, reaction-driven status. For engineering teams using Plane.
- **chatbot**: no tools, no attachments, no reactions. A general-purpose assistant with channel-specific context. For teams that don't use Plane (marketing, BD, sales, tech-discussion, etc.).

The bot only responds in channels it knows about. Two ways to make it know about a channel:

```
instructions/
├── plane/
│   ├── <channel_id>.md     # one file per Plane-mode channel
│   └── dm.md               # if present, DMs run in plane mode
└── chatbot/
    ├── <channel_id>.md     # one file per chatbot-mode channel
    └── dm.md               # if present, DMs run in chatbot mode
```

1. **Per-channel file (highest priority).** Drop a markdown file under `instructions/plane/` or `instructions/chatbot/`. Filename is the Slack channel ID with `.md` extension (e.g. `C0B0E9R0PE0.md`). The file's contents are appended verbatim to the system prompt as that channel's custom context. Mode comes from the parent directory.
2. **`DEFAULT_CHANNEL_MODE` env var (fallback).** Set it to `chatbot` (or `plane`) in `.env` and the bot uses that mode for any channel it's invited to that doesn't have a specific file. Per-channel files still win.
3. **Neither set.** The bot stays silent. It logs `Mention in unconfigured channel C... — ignoring` so you can see who tried.

`dm.md` is a special filename. Whichever subdirectory it sits in defines DM behavior. If both `plane/dm.md` and `chatbot/dm.md` exist, the bot logs a warning and uses `plane/dm.md`. If neither exists, DMs fall through to `DEFAULT_CHANNEL_MODE` (if set), otherwise are silently ignored.

The same channel ID under both `plane/` and `chatbot/` is rejected at load with a warning; one wins, the other is dropped.

After editing the directory or `.env`: `systemctl restart chatops`. Hot reload is a future enhancement; for now the bot reads `instructions/` and `settings.default_channel_mode` once at startup.

To find a channel's ID, right-click it in Slack → **Copy link**. The ID is the last segment of the URL.

## Plane project routing

In Plane mode, the bot routes tickets between two projects:

- **CANDIDATE**: candidate-facing app, profiles, jobs, interviews, matching, signup, resume.
- **RECRUITER**: recruiter dashboard, hiring flows, ATS integration, talent, scrapers.

The LLM picks based on keywords. If ambiguous, it asks. You can also explicitly say "in CANDIDATE" or "in RECRUITER" to override.

The system prompt is built fresh per request and injects the current per-project state UUID map, so the LLM never has to ask Plane for state IDs mid-conversation. Caches refresh at startup; restart the service if you add a new state or workspace member.

## Reactions (Plane mode only)

React to one of the bot's own messages with these emojis to change the linked issue's state. The bot picks up the issue ID from the message text.

| Emoji | Result |
|---|---|
| `:white_check_mark:` / `:heavy_check_mark:` | Move to Done (completed group) |
| `:construction:` / `:hammer_and_wrench:` | Move to In Progress (started group) |
| `:back:` | Move to Backlog |
| `:x:` / `:no_entry_sign:` | Move to Cancelled |

The bot replies in-thread confirming the new state, named for whichever state it landed in. Reactions on user messages, or on bot messages that don't reference an issue URL, are ignored.

## Standup cron

Optional. Set `STANDUP_HOUR_UTC=4` in `.env` (or any hour 0 to 23) to have the bot DM every Plane workspace member their pending tickets (`Todo` + `In Progress`) at that hour every day. `-1` disables.

The DM uses the same `chatops__list_assigned_tickets` tool described below, so the per-user list is fast and accurate. Members without a matching Slack account by email are skipped silently.

## Local tools

Beyond what the Plane MCP exposes, the bot ships one local tool:

- **`chatops__list_assigned_tickets`**: returns tickets assigned to a given email, filtered server-side by project and state group. The LLM is instructed to prefer this over `plane__list_work_items` for any "list X's tickets" request, because passing `assignee_ids` to the Plane MCP routes through Plane's `/work-items/advanced-search/` endpoint, which is forbidden for our API key. Defaults to `state_groups=["unstarted"]` (Todo).

Add more local tools by:

1. Defining the OpenAI function schema in `LOCAL_TOOL_DEFS` in `app/plane.py`.
2. Implementing an async handler returning a JSON string.
3. Registering it in `LOCAL_TOOL_HANDLERS`.

## Attachments (Plane mode only)

Drop a file in the same message OR earlier in the thread. After the LLM finishes its tool calls and lands on a `plane__create_work_item` or `plane__update_work_item`, the bot:

1. Downloads each Slack file using the bot token.
2. Walks Plane's 3-step S3 presigned upload (`POST issue-attachments/` → `POST` to S3 → `PATCH is_uploaded=true`).
3. Patches the issue's `description_html` to embed images inline (and file links for non-images), so attachments render in the issue body, not just under the Attachments tab.

The Plane MCP server has no attachment tools (verified across all 109), so this path bypasses MCP and hits Plane's REST API directly with the same `PLANE_API_KEY`.

Chatbot mode silently ignores files.

## Help shortcut

In Plane-mode channels, the bot responds to "help" (and a few variants like "what can you do", "commands") deterministically without burning an LLM call. The reply lists available actions. In chatbot mode, "help" goes through the LLM so the channel's persona answers in-character.

## Audit log

Every tool call is logged to MongoDB collection `chatops_audit` with:

- `slack_user` (Slack user ID)
- `slack_email` (resolved via `users.info`)
- `tool_calls` (list of `{name, args, result_preview}`; preview truncated to 200 chars)
- `result_preview` (first 500 chars of the final assistant reply)
- `created_at` (UTC timestamp)

Best-effort: write failures are logged and swallowed; the bot does not surface them to the user. If `MONGODB_URI` is unset or the ping at startup fails, audit logging is disabled and the bot keeps running.

## Adding a new service (e.g. GitHub)

1. Get a GitHub PAT or set up a GitHub App.
2. Add to `MCP_SERVERS` in `app/plane.py`:

```python
MCP_SERVERS["github"] = StdioServerParameters(
    command="docker",
    args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_PAT"]},
)
```

3. Add `GITHUB_PAT` to `.env`.
4. Update the system prompt in `app/prompts.py` to mention GitHub (so the LLM knows it can route there).
5. Restart: `systemctl restart chatops`.

The LLM auto-discovers GitHub tools and starts using them in any Plane-mode channel. Tools are prefixed `github__<tool>` to keep the namespace flat.

## Tech stack

- **Python 3.12** (runs on 3.10+ in theory; production runs 3.12)
- **Slack Bolt** (slack_bolt): Slack signature verification, slash commands, mentions, lazy listeners
- **FastAPI + uvicorn**: HTTP server on port `9001`
- **OpenAI** (gpt-4o-mini): natural language to tool calls; up to 8 iterations per request
- **MCP Python SDK**: talks to MCP servers via stdio
- **Plane MCP Server** (`uvx plane-mcp-server`): official, 100+ Plane tools
- **MongoDB** (Motor): audit log (optional)
- **aiohttp**: direct Plane REST calls for the attachment bridge
- **Cloudflare Tunnel**: public HTTPS without reverse proxy
- **systemd**: process management on VPS

## File map

```
app/
├── __main__.py         # `python -m app` entrypoint
├── main.py             # FastAPI startup/shutdown, mounts /slack/events and /health
├── slack_app.py        # Shared Slack Bolt AsyncApp instance
├── config.py           # Settings (pydantic-settings) + logger
├── handlers.py         # @-mention, DM, /cs, reaction listeners + shared agent flow
├── agent.py            # LLM tool-call loop (plane mode multi-turn; chatbot single completion)
├── prompts.py          # System prompt builder, help text, Slack mrkdwn coercion
├── instructions.py     # Per-channel instruction loader and mode resolver
├── plane.py            # MCP manager, member/state caches, chatops__* local tools,
│                       #   Slack-file → Plane S3 attachment bridge
├── audit.py            # MongoDB audit log writer (best-effort)
└── standup.py          # Optional daily DM cron
deploy/
├── chatops.service     # systemd unit for the bot
└── cloudflared.service # systemd unit for the tunnel
instructions/
├── plane/              # one .md per Plane-mode channel + optional dm.md
└── chatbot/            # one .md per chatbot-mode channel + optional dm.md
.env.example            # all settable env vars
requirements.txt
```

## License

Proprietary. RemoteStar internal use.

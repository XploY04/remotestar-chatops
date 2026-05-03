# RemoteStar ChatOps Bot

Slack bot for the RemoteStar team to manage Plane tickets through natural language. Built on the official Plane MCP server. Designed to extend to GitHub and other services later by registering more MCP servers.

## Examples

In any allowed Slack channel:

```
@chatops create a ticket: API returning 500s during signup, assign rudy@remotestar.io
@chatops list my open candidate tickets
@chatops mark PROJ-123 as in progress
@chatops add a comment to PROJ-45: looks good, ship it
```

Or via slash command:

```
/cs create a ticket for the recruiter dashboard rendering bug
/cs list high-priority tickets
```

## Architecture

```
Slack → Cloudflare Tunnel → ChatOps bot (FastAPI + Slack Bolt)
                                 ↓
                        OpenAI gpt-4o-mini (tool calling)
                                 ↓
                     Plane MCP server (subprocess via stdio)
                                 ↓
                       Self-hosted Plane API
```

The bot has one Slack endpoint. The LLM picks tools from any registered MCP server. Adding GitHub later = adding `github-mcp-server` to `MCP_SERVERS` in `app.py`. No new endpoints, no new code paths.

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
    - `commands`
    - `app_mentions:read`
    - `chat:write`
    - `users:read`
    - `users:read.email`
    - `channels:history`
  - **Slash Commands** → Create:
    - Command: `/cs`
    - Request URL: `https://chatops.remotestar.io/slack/events` (we'll fill after Tunnel)
    - Description: `RemoteStar ChatOps`
    - Usage hint: `<natural language instruction>`
  - **Event Subscriptions** → Enable Events:
    - Request URL: `https://chatops.remotestar.io/slack/events`
    - Subscribe to bot events: `app_mention`
  - **Install to Workspace** → copy **Bot Token** (`xoxb-...`)
- Get the `#product` channel ID:
  - In Slack, right-click `#product` → **Copy link**
  - The ID is the last segment of the URL (e.g., `C012ABCDE`)
- Invite the bot to the channel: in `#product` type `/invite @chatops`

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

# Configure routing — create /root/.cloudflared/config.yml:
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
- In `#product`: `/cs create a test ticket: Hello from ChatOps`
- Bot replies in channel within 5-15 seconds with a link to the new Plane issue

## Channel modes and the `instructions/` directory

The bot runs in one of two modes per channel:

- **plane** — full Plane MCP toolset, attachment uploads, reaction-driven status. For engineering teams using Plane.
- **chatbot** — no tools, no attachments, no reactions. A general-purpose assistant with channel-specific context. For teams that don't use Plane (marketing, BD, sales, etc.).

There is no default mode. The bot only responds in channels that have an instructions file:

```
instructions/
├── plane/
│   ├── <channel_id>.md     # one file per Plane-mode channel
│   └── dm.md               # if present, DMs run in plane mode
└── chatbot/
    ├── <channel_id>.md     # one file per chatbot-mode channel
    └── dm.md               # if present, DMs run in chatbot mode
```

- Filename = the Slack channel ID (e.g. `C0B0E9R0PE0.md`); the file's contents are appended verbatim to the system prompt as that channel's custom context.
- Mode comes from the parent directory.
- `dm.md` is a special filename. Whichever subdirectory it sits in defines DM behavior. If both `plane/dm.md` and `chatbot/dm.md` exist, the bot logs a warning and uses `plane/dm.md`. If neither exists, DMs are ignored.
- A channel without a file gets no response from the bot — silent, no fallback.
- After editing the directory: `systemctl restart chatops`. Hot reload is a future enhancement.

Find the channel IDs with the throwaway script described in `.claude/plans/luminous-conjuring-dongarra.md` (uses `conversations.list`).

## Plane project routing

In Plane mode, the bot routes tickets between two projects:

- **CANDIDATE** — candidate-facing app, profiles, jobs, interviews
- **RECRUITER** — recruiter dashboard, hiring flows, ATS

The LLM picks based on keywords. If ambiguous, it'll ask. You can also explicitly say "in CANDIDATE" or "in RECRUITER" to override.

## Audit log

Every tool call is logged to MongoDB collection `chatops_audit` with:
- `slack_user`, `slack_email`
- `tool_calls` (name, args)
- `result_preview`
- `created_at`

Useful for debugging and seeing who did what.

## Adding a new service (e.g. GitHub)

1. Get a GitHub PAT or set up a GitHub App
2. Add to `MCP_SERVERS` in `app/plane.py`:

```python
MCP_SERVERS["github"] = StdioServerParameters(
    command="docker",
    args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_PAT"]},
)
```

3. Add `GITHUB_PAT` to `.env`
4. Update the system prompt in `app/prompts.py` to mention GitHub
5. Restart: `systemctl restart chatops`

The LLM auto-discovers GitHub tools and starts using them in any Plane-mode channel.

## Tech stack

- **Python 3.10+**
- **Slack Bolt** (slack_bolt) — Slack signature verification, slash commands, mentions, lazy listeners
- **FastAPI + uvicorn** — HTTP server
- **OpenAI** (gpt-4o-mini) — natural language to tool calls
- **MCP Python SDK** — talks to MCP servers via stdio
- **Plane MCP Server** (`uvx plane-mcp-server`) — official, 100+ Plane tools
- **MongoDB** (Motor) — audit log
- **Cloudflare Tunnel** — public HTTPS without reverse proxy
- **systemd** — process management on VPS

## License

Proprietary — RemoteStar internal use.

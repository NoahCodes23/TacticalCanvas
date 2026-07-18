"""
Voice-driven assistant coach for TacticalCanvas.

The split, and why it matters:

  * ElevenLabs owns the *conversation* — speech-to-text, text-to-speech, voice
    activity detection, turn-taking and barge-in. Those are genuinely hard and
    not worth rebuilding.
  * OpenRouter (tencent/hy3 by default) owns the *thinking*. ElevenLabs' own LLM
    is not used at all: the agent is configured with `llm="custom-llm"` pointing
    at OpenRouter's OpenAI-compatible endpoint, so every turn is reasoned by the
    Tencent model.
  * MCP owns the *doing*. Every tool in mcp_server.py is mirrored onto the agent
    as a client tool, so the Tencent model can pause the replay, move players,
    read board state, and ask for coach advice.

Speaking a request therefore runs: mic -> ElevenLabs STT -> OpenRouter/Tencent
-> tool_calls -> MCP (in-process) -> result back to Tencent -> ElevenLabs TTS.

Setup:
    pip install "elevenlabs[pyaudio]" python-dotenv fastmcp httpx
    # On Windows, pyaudio usually installs fine via pip. If it fails:
    #   pip install pipwin && pipwin install pyaudio

Config (put these in a .env file next to this script):
    ELEVENLABS_API_KEY=your-key-here     (required)
    ELEVENLABS_AGENT_ID=                  (optional — a throwaway agent is
                                           created for you if left blank)
    OPENROUTER_API_KEY=your-key-here     (required — this is the brain)
    OPENROUTER_MODEL=tencent/hy3:free    (optional)

Prerequisite — start the board in another terminal, or the tools have nothing
to talk to:
    python tc.py start

Run:
    python voice_agent.py

Then just talk: "what should I do here?", "move home nine to thirty, twenty",
"reset the scenario". Ctrl+C to stop.
"""

import asyncio
import os
import sys
import threading
import time

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import ClientTools, Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from elevenlabs.types.tool_request_model import ToolRequestModel
from elevenlabs.types.tool_request_model_tool_config import (
    ToolRequestModelToolConfig_Client,
)
from fastmcp import Client as MCPClient

# The MCP server instance — reused in-process so the voice agent and any other
# MCP client call the exact same tools. Importing it is cheap (no network, no
# camera): the tools only open a WebSocket when they are actually invoked.
from mcp_server import mcp as mcp_app

# Line-buffer stdout so the transcript appears live when this runs as a child of
# tc.py --voice. Python block-buffers a non-tty stdout, which silently swallows
# every "You:" / "Agent:" line until the buffer fills or the process exits.
sys.stdout.reconfigure(line_buffering=True)

# Load .env sitting next to this script (works regardless of cwd).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
# Treat an empty/blank agent id the same as unset.
AGENT_ID = (os.environ.get("ELEVENLABS_AGENT_ID") or "").strip() or None

OPENROUTER_API_KEY = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
OPENROUTER_MODEL = (os.environ.get("OPENROUTER_MODEL") or "").strip() or "tencent/hy3:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ElevenLabs stores third-party credentials as workspace secrets and references
# them by id; a raw key is not accepted in the agent config. This is the name we
# store ours under, so repeat runs reuse the same secret instead of piling up.
OPENROUTER_SECRET_NAME = "TACTICALCANVAS_OPENROUTER_KEY"

SYSTEM_PROMPT = (
    "You are an assistant coach standing on the touchline beside a football "
    "(soccer) coach during a stoppage, talking out loud. You are wired into "
    "TacticalCanvas: a tactical board, projected on the wall, replaying real "
    "tracking data from a real match at 25 frames per second. The coach can "
    "drag players around on it by hand. You can see the same board through "
    "tools, and act on it through them.\n"
    "\n"
    "THE BOARD\n"
    "The pitch is 105 metres long (x, 0 is the left goal line) and 68 metres "
    "wide (y). Players are identified as H9 (home shirt 9) or A10 (away shirt "
    "10). The tracking data is anonymised — there are NO player names anywhere, "
    "so refer to players as 'home nine' or 'away ten' and never invent a name, "
    "a club, or a fixture.\n"
    "\n"
    "YOUR MOST IMPORTANT TOOL: get_coach_advice\n"
    "This is far richer than a single number, and it is where almost every "
    "tactical question should go. It freezes the moment, takes five snapshots "
    "spanning the last 1.6 seconds, and runs the full analysis model over each: "
    "who has possession and which player is carrying, how much space the "
    "carrier has and how hard they are being pressed, every available pass "
    "ranked by completion probability and expected value, which passes actually "
    "progress the ball, where each team's defensive and attacking lines are "
    "sitting, team width, depth and shape, who is sprinting, channel overloads, "
    "offside risk, and the recent match events leading in. It returns that as a "
    "few sentences of concrete sideline advice.\n"
    "Call it whenever the coach asks what to do, what is going wrong, what they "
    "are looking at, whether a pass is on, who is free, where the danger is, or "
    "simply says they want advice. Deliver what it gives you in your own voice, "
    "naturally, as if the read were yours. Do not add tactical claims it did not "
    "make — the numbers behind it are experimental estimates, not facts, so "
    "never present them as certainties or predictions.\n"
    "\n"
    "THE OTHER TOOLS\n"
    "get_board_state and list_players give exact positions, shirt numbers and "
    "who has the ball. Use them for factual lookups ('where is home nine?', "
    "'who is nearest the ball?') and any time you need a player id. Never guess "
    "a position.\n"
    "move_player repositions someone, in metres. If the coach gives a number "
    "without a team, ask which team before moving anyone.\n"
    "set_playing, enter_edit_mode, exit_edit_mode and reset_scenario control "
    "the replay. get_coach_advice pauses the match by itself, so you never need "
    "to ask permission to pause before giving advice.\n"
    "toggle_calibration is for aligning the projector, not for tactics.\n"
    "\n"
    "HOW TO SPEAK\n"
    "You are heard, not read. A few short sentences, no lists, no headings, no "
    "reading out strings of numbers unless the coach asked for one specific "
    "figure. Say what you are doing while you do it — 'let me look at the "
    "board' — because the tools take a moment. If a tool fails, say so plainly "
    "and carry on; never invent what it would have said."
)
FIRST_MESSAGE = "Board's live and I'm watching. What do you want to look at?"

# Every tool in mcp_server.py is exposed to the agent, including the tool_1
# debug tool — saying "run tool 1" is the quickest way to prove the whole chain
# (ElevenLabs -> OpenRouter -> tool_call -> MCP -> spoken result) is alive
# without touching the board. Add a name here to hide a tool from voice.
EXCLUDED_TOOLS: set[str] = set()


# ---------------------------------------------------------------------------
# MCP -> ElevenLabs client tools
# ---------------------------------------------------------------------------
async def _list_mcp_tools() -> list:
    """Fetch the MCP tool list (name, description, JSON schema)."""
    async with MCPClient(mcp_app) as mcp:
        return [t for t in await mcp.list_tools() if t.name not in EXCLUDED_TOOLS]


async def _call_mcp_tool(name: str, parameters: dict) -> str:
    """Run one MCP tool in-process and flatten the result to speakable text.

    Errors are returned rather than raised: a tool failure should make the agent
    say "the board isn't running" out loud, not kill the conversation.
    """
    try:
        async with MCPClient(mcp_app) as mcp:
            result = await mcp.call_tool(name, parameters or {})
    except Exception as e:  # noqa: BLE001 - surfaced to the model as text
        return f"Tool {name} failed: {e}"

    data = getattr(result, "data", None)
    if data is not None:
        return data if isinstance(data, str) else repr(data)
    if getattr(result, "content", None):
        return result.content[0].text
    return f"{name} completed."


def _schema_to_elevenlabs(schema: dict) -> dict | None:
    """Convert an MCP inputSchema to ElevenLabs' tool-parameter schema.

    Both are JSON Schema objects, but ElevenLabs rejects the extra keys FastMCP
    emits (notably `additionalProperties`), and wants no parameter block at all
    for a no-argument tool.
    """
    properties = schema.get("properties") or {}
    if not properties:
        return None
    return {
        "type": "object",
        "properties": {
            name: {
                "type": prop.get("type", "string"),
                "description": prop.get("description", name),
            }
            for name, prop in properties.items()
        },
        "required": schema.get("required", []),
    }


def build_client_tools(tools: list) -> ClientTools:
    """The local side: one handler per MCP tool, dispatching by name.

    The default-argument binding on `name` is load-bearing — a closure over the
    loop variable would make every handler call the last tool.
    """
    registry = ClientTools()
    for tool in tools:

        async def handler(parameters: dict, name: str = tool.name) -> str:
            print(f"  [tool] {name}({parameters or {}})")
            output = await _call_mcp_tool(name, parameters)
            print(f"  [tool] -> {output[:160]}")
            return output

        registry.register(tool.name, handler, is_async=True)
    return registry


def sync_agent_tools(client: ElevenLabs, tools: list) -> list[str]:
    """The server side: make sure every MCP tool exists as a client tool in the
    workspace, and return the ids to attach to the agent. Idempotent — safe to
    run on every startup."""
    existing = {t.tool_config.name: t.id for t in client.conversational_ai.tools.list().tools}
    tool_ids = []
    for tool in tools:
        if tool.name in existing:
            tool_ids.append(existing[tool.name])
            continue
        config = {
            "name": tool.name,
            "description": (tool.description or tool.name).strip(),
        }
        params = _schema_to_elevenlabs(tool.inputSchema or {})
        if params:
            config["parameters"] = params
        created = client.conversational_ai.tools.create(
            request=ToolRequestModel(
                tool_config=ToolRequestModelToolConfig_Client(**config)
            )
        )
        tool_ids.append(created.id)
        print(f"  registered client tool: {tool.name}")
    return tool_ids


# ---------------------------------------------------------------------------
# Custom LLM wiring
# ---------------------------------------------------------------------------
def ensure_openrouter_secret(client: ElevenLabs) -> str:
    """Return the workspace-secret id holding the OpenRouter key, creating it on
    first run. ElevenLabs will not echo a stored secret's value back, so we
    cannot compare — an existing secret by this name is trusted as-is."""
    for secret in client.conversational_ai.secrets.list().secrets:
        if secret.name == OPENROUTER_SECRET_NAME:
            return secret.secret_id
    print(f"  storing OpenRouter key as workspace secret {OPENROUTER_SECRET_NAME}")
    created = client.conversational_ai.secrets.create(
        name=OPENROUTER_SECRET_NAME, value=OPENROUTER_API_KEY
    )
    return created.secret_id


def agent_config(secret_id: str, tool_ids: list[str]) -> dict:
    """The conversation config that swaps ElevenLabs' LLM for OpenRouter.

    `llm: "custom-llm"` is what takes their model out of the loop; `custom_llm`
    points at OpenRouter's OpenAI-compatible endpoint. ElevenLabs sends the tool
    definitions along in standard OpenAI `tools` format and executes whatever
    `tool_calls` come back, which is why the MCP bridge keeps working unchanged.
    """
    return {
        "agent": {
            "prompt": {
                "prompt": SYSTEM_PROMPT,
                "llm": "custom-llm",
                "custom_llm": {
                    "url": OPENROUTER_BASE_URL,
                    "model_id": OPENROUTER_MODEL,
                    "api_key": {"secret_id": secret_id},
                    "api_type": "chat_completions",
                    # OpenRouter attributes usage to an app via these headers.
                    "request_headers": {
                        "HTTP-Referer": os.environ.get(
                            "OPENROUTER_SITE_URL", "http://localhost:8000"
                        ),
                        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "TacticalCanvas"),
                    },
                },
                "tool_ids": sorted(set(tool_ids)),
            },
            "first_message": FIRST_MESSAGE,
            "language": "en",
        },
    }


def create_agent(client: ElevenLabs, secret_id: str, tool_ids: list[str]) -> str:
    """Create the coaching agent, already pointed at OpenRouter."""
    print("No ELEVENLABS_AGENT_ID set — creating an agent...")
    agent = client.conversational_ai.agents.create(
        name="TacticalCanvas Assistant Coach",
        conversation_config=agent_config(secret_id, tool_ids),
    )
    print(f"Created agent: {agent.agent_id}")
    print("Tip: put ELEVENLABS_AGENT_ID in .env to reuse it next time.")
    return agent.agent_id


def main() -> None:
    if not API_KEY:
        sys.exit("ELEVENLABS_API_KEY is not set. Set it and try again.")
    if not OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set — it is the agent's LLM. Set it and try again.")

    client = ElevenLabs(api_key=API_KEY)

    print("Wiring up...")
    mcp_tools = asyncio.run(_list_mcp_tools())
    print(f"  {len(mcp_tools)} MCP tools: {', '.join(t.name for t in mcp_tools)}")
    secret_id = ensure_openrouter_secret(client)

    if AGENT_ID:
        agent_id = AGENT_ID
        # Re-apply the whole config on every start rather than diffing it. The
        # agent is ours, the config is small, and a half-updated agent silently
        # falling back to ElevenLabs' own LLM is the exact failure we're
        # avoiding. Non-fatal: a transient API hiccup shouldn't block talking.
        try:
            tool_ids = sync_agent_tools(client, mcp_tools)
            client.conversational_ai.agents.update(
                agent_id, conversation_config=agent_config(secret_id, tool_ids)
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not update the agent config: {e}")
    else:
        # Tools must exist before an agent can reference them.
        agent_id = create_agent(client, secret_id, [])
        tool_ids = sync_agent_tools(client, mcp_tools)
        client.conversational_ai.agents.update(
            agent_id, conversation_config=agent_config(secret_id, tool_ids)
        )

    print(f"  LLM: {OPENROUTER_MODEL} via OpenRouter (ElevenLabs handles voice only)")

    conversation = Conversation(
        client,
        agent_id,
        # Auth: agents can be public (no key) or private (key required). Passing
        # the key is safe either way.
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=build_client_tools(mcp_tools),
        callback_agent_response=lambda text: print(f"\nAgent: {text}"),
        callback_user_transcript=lambda text: print(f"You: {text}"),
    )

    print("\nStarting conversation. Speak into your mic. Ctrl+C to quit.\n")
    conversation.start_session()

    # Ctrl+C handling on Windows: wait_for_session_end() blocks on an event that
    # SIGINT can't interrupt, so a plain Ctrl+C is swallowed. Instead we run that
    # blocking wait on a background thread and keep the main thread in a
    # time.sleep() loop, which *is* interruptible by Ctrl+C on Windows.
    result: dict[str, object] = {}
    done = threading.Event()

    def _wait() -> None:
        try:
            result["id"] = conversation.wait_for_session_end()
        finally:
            done.set()

    threading.Thread(target=_wait, daemon=True).start()

    try:
        while not done.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping — ending session...")
        conversation.end_session()
        done.wait(timeout=5)  # let the session tear down cleanly

    print(f"\nConversation ended. id={result.get('id')}")


if __name__ == "__main__":
    main()

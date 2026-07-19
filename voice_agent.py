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
    "(soccer) coach, talking out loud. You are wired into TacticalCanvas: a "
    "tactical board, projected on the wall, replaying real tracking data from a "
    "real match at 25 frames per second. The coach can drag players around on it "
    "by hand. You can see the same board through tools, and act on it through "
    "them.\n"
    "\n"
    "You are a general-purpose assistant, not an analysis engine. Most of what "
    "the coach says is an instruction to operate the board — show something, "
    "hide something, pause, move a player, switch match. Do that, confirm it in "
    "a few words, and stop. Be conversational and helpful about anything else "
    "they ask, including things unrelated to the board.\n"
    "\n"
    "THE BOARD\n"
    "The pitch is 105 metres long (x, 0 is the left goal line) and 68 metres "
    "wide (y). Players are identified as H9 (home shirt 9) or A10 (away shirt "
    "10). The tracking data is anonymised — there are NO player names anywhere, "
    "so refer to players as 'home nine' or 'away ten' and never invent a name, "
    "a club, or a fixture.\n"
    "\n"
    "DO NOT VOLUNTEER ANALYSIS\n"
    "Never offer a tactical read, a critique, or a suggestion off your own bat, "
    "and never answer a tactical question from what you can infer. You have one "
    "tool that produces analysis — get_coach_advice — and its output is the ONLY "
    "source of tactical opinion you may speak. If the coach has not asked for "
    "advice, do not produce any.\n"
    "Call get_coach_advice only when the coach explicitly asks for advice, "
    "feedback, an opinion, or a read of the situation: 'what should I do here', "
    "'what's going wrong', 'give me advice', 'what do you think'. It pauses the "
    "match itself. Deliver what it returns in your own voice, and add nothing to "
    "it — the numbers behind it are experimental estimates, not facts.\n"
    "If a request is an instruction to show something, it is NOT a request for "
    "advice, even when it sounds tactical. Operate the board instead.\n"
    "\n"
    "PICKING THE RIGHT TOOL\n"
    "Anything phrased as show / display / turn on / hide / toggle is a board "
    "view, and goes to set_overlay. In particular:\n"
    "  'suggest positions', 'suggested positions', 'where should players be', "
    "'show me better positions' -> set_overlay(overlay='suggested'). This draws "
    "ghost circles on the board. It is NOT get_coach_advice.\n"
    "  offside line -> set_overlay(overlay='offside')\n"
    "  compactness, team width, shape -> set_overlay(overlay='compactness')\n"
    "  defender shadows, defensive reach -> set_overlay(overlay='shadows')\n"
    "  pitch control, space, who owns which area -> set_overlay(overlay='pitch_control')\n"
    "  formation -> set_overlay(overlay='formation')\n"
    "Pass enabled=false to switch one off. set_experiment controls the three AI "
    "experiment switches; run_demo_preset applies a canned demo view.\n"
    "get_board_state and list_players give exact positions, shirt numbers and "
    "who has the ball — use them for factual lookups ('where is home nine?', "
    "'who is nearest the ball?') and any time you need a player id. Reporting a "
    "position is a fact, not analysis, so that is always fine. Never guess one.\n"
    "move_player repositions someone, in metres. If the coach gives a number "
    "without a team, ask which team first.\n"
    "set_playing, enter_edit_mode, exit_edit_mode and reset_scenario control the "
    "replay. load_match and list_matches switch the match. set_coaching_team "
    "picks the side you are advising. start_calibration and cancel_calibration "
    "are for aligning the projector, not for tactics.\n"
    "If you are unsure which tool a request means, ask a short clarifying "
    "question rather than guessing or falling back on advice.\n"
    "\n"
    "HOW TO SPEAK\n"
    "You are heard, not read. A few short sentences, no lists, no headings, no "
    "reading out strings of numbers unless the coach asked for one specific "
    "figure. Confirmations should be very short — 'suggested positions are up'. "
    "Say what you are doing while you do it — 'let me look at the board' — "
    "because the tools take a moment. If a tool fails, say so plainly and carry "
    "on; never invent what it would have said."
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

    def convert(name: str, prop: dict) -> dict:
        # Literal[...] params arrive as an enum, sometimes wrapped in anyOf when
        # the parameter is optional. Forwarding it is what stops the model
        # inventing an overlay name: the valid set becomes part of the schema
        # instead of prose buried in the description.
        options = prop.get("enum")
        kind = prop.get("type")
        if not options or not kind:
            for variant in prop.get("anyOf") or []:
                options = options or variant.get("enum")
                kind = kind or variant.get("type")
        converted = {"type": kind or "string", "description": prop.get("description", name)}
        if options:
            converted["enum"] = list(options)
        return converted

    return {
        "type": "object",
        "properties": {name: convert(name, prop) for name, prop in properties.items()},
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
        # Turn-taking, tuned for a room with background chatter rather than a
        # quiet desk. "patient" makes the agent wait through pauses instead of
        # treating the first gap -- or a stray voice behind the coach -- as its
        # cue to start talking, and the longer timeout gives a coach who is
        # thinking mid-sentence room to finish. The dashboard's push-to-talk is
        # the real defence; this stops the agent interrupting on the turns that
        # do get through.
        "turn": {
            "turn_timeout": 10,
            "turn_eagerness": "patient",
        },
        # The actual background-noise filter, and it lives here rather than in
        # any SDK: ElevenLabs decides whether incoming speech is the person
        # talking to the agent or a voice behind them, and drops the latter
        # before it becomes a turn. Off by default, which is why a busy room
        # used to put other people's conversations into the transcript.
        "vad": {"background_voice_detection": True},
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

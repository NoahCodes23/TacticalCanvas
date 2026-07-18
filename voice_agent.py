"""
Simple ElevenLabs conversational agent you can talk back and forth to.

Setup:
    pip install "elevenlabs[pyaudio]" python-dotenv
    # On Windows, pyaudio usually installs fine via pip. If it fails:
    #   pip install pipwin && pipwin install pyaudio

Config (put these in a .env file next to this script):
    ELEVENLABS_API_KEY=your-key-here     (required)
    ELEVENLABS_AGENT_ID=                  (optional — a throwaway agent is
                                           created for you if left blank)

Run:
    python voice_agent.py

Then just talk. Ctrl+C to stop.

Tool calling: the agent has one client tool, "run_tool_1". When you say "run tool 1",
the ElevenLabs LLM emits the tool call, and the handler below runs the MCP `tool_1`
in-process (via a FastMCP in-memory client) and speaks the result back. The tool is
attached to the agent idempotently on startup, so it self-heals.
"""

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

# The MCP server instance — reused in-process so the voice tool and any MCP agent
# call the exact same tool_1. Importing it is cheap (no network, no camera).
from mcp_server import mcp as mcp_app

# Load .env sitting next to this script (works regardless of cwd).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
# Treat an empty/blank agent id the same as unset.
AGENT_ID = (os.environ.get("ELEVENLABS_AGENT_ID") or "").strip() or None

# What the agent is + how it opens the conversation. Tweak freely.
SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant helping a developer at a "
    "hackathon. Keep replies short and conversational since they're spoken aloud."
)
FIRST_MESSAGE = "Hey! I'm listening — what do you want to talk about?"

# Client tool exposed to the ElevenLabs agent. The name must match on both sides:
# the tool declared on the agent (server-side, so the LLM knows it exists) and the
# handler registered below (client-side, so we actually run it).
CLIENT_TOOL_NAME = "run_tool_1"
CLIENT_TOOL_DESCRIPTION = (
    "Runs the local debug tool called 'tool 1'. Call this immediately, with no "
    "parameters, whenever the user asks to run tool 1 or 'the debug tool'."
)


async def _handle_run_tool_1(parameters: dict) -> str:
    """Client-tool handler: bridge the voice agent to MCP tool_1, in-process."""
    async with MCPClient(mcp_app) as mcp:
        result = await mcp.call_tool("tool_1", {})
    text = getattr(result, "data", None)
    if not text and getattr(result, "content", None):
        text = result.content[0].text
    return text or "tool 1 ran"


def build_client_tools() -> ClientTools:
    """The local side: map the agent's tool call to our async handler."""
    tools = ClientTools()
    tools.register(CLIENT_TOOL_NAME, _handle_run_tool_1, is_async=True)
    return tools


def ensure_client_tool(client: ElevenLabs, agent_id: str) -> None:
    """The server side: make sure the agent has run_tool_1 attached, without
    disturbing its system prompt. Idempotent — safe to call on every startup."""
    tool_id = None
    for t in client.conversational_ai.tools.list().tools:
        if t.tool_config.name == CLIENT_TOOL_NAME:
            tool_id = t.id
            break
    if tool_id is None:
        created = client.conversational_ai.tools.create(
            request=ToolRequestModel(
                tool_config=ToolRequestModelToolConfig_Client(
                    name=CLIENT_TOOL_NAME, description=CLIENT_TOOL_DESCRIPTION
                )
            )
        )
        tool_id = created.id

    prompt = client.conversational_ai.agents.get(agent_id).conversation_config.agent.prompt
    current = prompt.tool_ids or []
    if tool_id not in current:
        # Re-send the system prompt in the same payload so it is never clobbered.
        client.conversational_ai.agents.update(
            agent_id,
            conversation_config={
                "agent": {
                    "prompt": {
                        "prompt": prompt.prompt,
                        "tool_ids": sorted(set(current) | {tool_id}),
                    }
                }
            },
        )


def create_agent(client: ElevenLabs) -> str:
    """Create a minimal conversational agent and return its id."""
    print("No ELEVENLABS_AGENT_ID set — creating a temporary agent...")
    agent = client.conversational_ai.agents.create(
        name="Hackathon Voice Agent",
        conversation_config={
            "agent": {
                "prompt": {"prompt": SYSTEM_PROMPT},
                "first_message": FIRST_MESSAGE,
                "language": "en",
            },
        },
    )
    print(f"Created agent: {agent.agent_id}")
    print("Tip: export ELEVENLABS_AGENT_ID to reuse it next time.")
    return agent.agent_id


def main() -> None:
    if not API_KEY:
        sys.exit("ELEVENLABS_API_KEY is not set. Set it and try again.")

    client = ElevenLabs(api_key=API_KEY)
    agent_id = AGENT_ID or create_agent(client)

    # Make sure the agent knows about run_tool_1. Non-fatal: a transient API hiccup
    # shouldn't stop you from having a normal conversation.
    try:
        ensure_client_tool(client, agent_id)
    except Exception as e:
        print(f"[warn] could not verify the run_tool_1 client tool: {e}")

    conversation = Conversation(
        client,
        agent_id,
        # Auth: agents can be public (no key) or private (key required). Passing
        # the key is safe either way.
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=build_client_tools(),
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

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
"""

import os
import signal
import sys

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

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

    conversation = Conversation(
        client,
        agent_id,
        # Auth: agents can be public (no key) or private (key required). Passing
        # the key is safe either way.
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=lambda text: print(f"\nAgent: {text}"),
        callback_user_transcript=lambda text: print(f"You: {text}"),
    )

    # Clean shutdown on Ctrl+C.
    signal.signal(signal.SIGINT, lambda *_: conversation.end_session())

    print("\nStarting conversation. Speak into your mic. Ctrl+C to quit.\n")
    conversation.start_session()

    conversation_id = conversation.wait_for_session_end()
    print(f"\nConversation ended. id={conversation_id}")


if __name__ == "__main__":
    main()

"""
General MCP server for TacticalCanvas.

Exposes the tactical board's controls as MCP tools so *any* LLM agent — Claude
Desktop, a custom agent loop, an ElevenLabs voice agent, etc. — can drive the
board on its own. It does not change the FastAPI core: each tool is just another
WebSocket client speaking the same versioned command protocol the dashboard uses
(server/protocol.py). The TacticalCanvas server must be running.

Setup:
    pip install fastmcp          # websockets is already a dependency of the core

Prerequisite — start the board (in another terminal):
    python tc.py start
    # or headless, no camera:  TC_NO_VISION=1 python run.py

Run this server:
    python mcp_server.py                     # stdio transport (Claude Desktop, agent loops)
    TC_MCP_TRANSPORT=http python mcp_server.py   # streamable-HTTP (remote/voice agents)

Point it at a non-default board:
    TC_WS_URL=ws://localhost:9000/ws python mcp_server.py
    # or just TC_PORT=9000

Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "tacticalcanvas": {
          "command": "python",
          "args": ["c:/Users/Noah/Desktop/hack the 6ix 2026/TacticalCanvas/mcp_server.py"]
        }
      }
    }

Then just ask the model to "pause and move the right-back to (30, 10)"; it will
pick and call the tools itself.
"""

import logging
import os
import sys
import time

import httpx
import websockets
from fastmcp import Context, FastMCP

# Keep third-party logging quiet. Importing this module (voice_agent.py does) must
# not unleash websockets/elevenlabs DEBUG spam, so the root stays at WARNING. Log to
# stderr, never stdout: in stdio transport, stdout is the MCP protocol channel and
# any stray print there corrupts the stream.
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Our own debug-tool logger: one red line on stderr (its own stream). propagate=False
# keeps it off the root handler so it isn't also printed as a plain line.
logger = logging.getLogger("tacticalcanvas.mcp")
logger.setLevel(logging.DEBUG)
logger.propagate = False
_debug_handler = logging.StreamHandler(sys.stderr)
_debug_handler.setFormatter(logging.Formatter("\033[91m%(levelname)s: %(message)s\033[0m"))
logger.addHandler(_debug_handler)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = 1  # must match server/protocol.py
PITCH_LENGTH = 105.0  # metres; must match server/match_data.py
PITCH_WIDTH = 68.0


def _ws_url() -> str:
    if url := os.environ.get("TC_WS_URL"):
        return url
    port = os.environ.get("TC_PORT", "8000")
    return f"ws://localhost:{port}/ws"


def _http_url() -> str:
    """Base URL of the same board, for the REST endpoints (coach advice)."""
    if base := os.environ.get("TC_HTTP_URL"):
        return base.rstrip("/")
    port = os.environ.get("TC_PORT", "8000")
    return f"http://localhost:{port}"


mcp = FastMCP("tacticalcanvas")

_seq = 0


def _envelope(msg_type: str, payload: dict | None = None) -> dict:
    """Build a command envelope matching server/protocol.py:Envelope."""
    global _seq
    _seq += 1
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "scenarioId": "demo",
        "clientId": "mcp",
        "sequenceNumber": _seq,
        "timestamp": time.time() * 1000.0,
        "type": msg_type,
        "payload": payload or {},
    }


async def _recv_snapshot(ws) -> dict:
    """Read messages until a STATE_SNAPSHOT arrives; return its payload."""
    while True:
        msg = await ws.recv()
        import json

        data = json.loads(msg)
        if data.get("type") == "STATE_SNAPSHOT":
            return data["payload"]
        if data.get("type") == "ERROR":
            raise RuntimeError(f"server error: {data['payload'].get('reason')}")


async def _get_state() -> dict:
    """Connect, read the authoritative snapshot the server sends on connect."""
    async with websockets.connect(_ws_url()) as ws:
        return await _recv_snapshot(ws)


async def _send_commands(envelopes: list[dict]) -> dict:
    """
    Open one connection, send the given commands in order, return the final
    snapshot. A single connection keeps drag ownership stable across the
    START/END pair (the server scopes ownership per-connection).
    """
    async with websockets.connect(_ws_url()) as ws:
        await _recv_snapshot(ws)  # drain the on-connect snapshot
        last = None
        for env in envelopes:
            await ws.send(_dumps(env))
            last = await _recv_snapshot(ws)  # each command triggers a broadcast
        return last if last is not None else await _recv_snapshot(ws)


def _dumps(obj: dict) -> str:
    import json

    return json.dumps(obj)


def _trim_player(p: dict) -> dict:
    return {
        "id": p["id"],
        "team": p["team"],
        "number": p["number"],
        "x": p["x"],
        "y": p["y"],
        "edited": p.get("edited", False),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool
async def tool_1(ctx: Context) -> str:
    """Tool 1 — a debug test tool. Run this whenever the user asks to "run tool 1".
    It logs a debug message and returns confirmation; it does not touch the board."""
    message = "tool 1 ran"
    logger.error(message)  # -> single red line on stderr (its own stream)
    await ctx.debug(message)  # -> also sent to the MCP client / agent as a log message
    return message


@mcp.tool
async def get_board_state() -> dict:
    """Get the current state of the tactical board: whether the replay is
    playing, edit mode, the current frame, the ball position, and every player's
    id/team/number/position (in pitch metres). Call this first to see what's on
    the board and to learn valid player ids."""
    s = await _get_state()
    return {
        "playing": s["playing"],
        "editMode": s["editMode"],
        "frameIndex": s["frameIndex"],
        "ball": s["ball"],
        "pitch": s["pitch"],
        "players": [_trim_player(p) for p in s["players"]],
    }


@mcp.tool
async def list_players() -> list[dict]:
    """List every player with id, team ('home'/'away'), shirt number, and current
    position in pitch metres. Player ids look like 'H9' (home #9) or 'A10'
    (away #10). Use this to find the id to pass to move_player."""
    s = await _get_state()
    return [_trim_player(p) for p in s["players"]]


@mcp.tool
async def set_playing(playing: bool) -> dict:
    """Play or pause the match replay. Pass true to resume playback, false to
    pause. Pausing is required before editing the board."""
    s = await _send_commands([_envelope("SET_PLAYING", {"playing": playing})])
    return {"playing": s["playing"], "editMode": s["editMode"]}


@mcp.tool
async def enter_edit_mode() -> dict:
    """Enter edit mode: pauses the replay and freezes the current frame so
    players can be repositioned. move_player does this automatically, but call it
    explicitly to stage several edits."""
    s = await _send_commands([_envelope("ENTER_EDIT_MODE")])
    return {"editMode": s["editMode"], "playing": s["playing"]}


@mcp.tool
async def exit_edit_mode() -> dict:
    """Leave edit mode and drop any manual player edits' grab state. The replay
    stays paused until you call set_playing(true)."""
    s = await _send_commands([_envelope("EXIT_EDIT_MODE")])
    return {"editMode": s["editMode"], "playing": s["playing"]}


@mcp.tool
async def reset_scenario() -> dict:
    """Reset the board to the start of the scenario: clears all manual edits,
    exits edit mode, rewinds to frame 0, and resumes playback."""
    s = await _send_commands([_envelope("RESET_SCENARIO")])
    return {"playing": s["playing"], "frameIndex": s["frameIndex"]}


@mcp.tool
async def toggle_calibration() -> dict:
    """Toggle the projector calibration-marker overlay (the four magenta targets)
    on or off. Used when aligning the projector, not during tactical editing."""
    s = await _send_commands([_envelope("TOGGLE_CALIBRATION")])
    return {"calibrationOverlay": s["calibrationOverlay"]}


@mcp.tool
async def get_coach_advice() -> dict:
    """Get tactical coaching advice about the current moment of the match. Call
    this whenever the user asks what they should do, what is going wrong, how to
    fix the shape, or for any read of the current situation. It analyses the last
    five tracking snapshots plus recent match events and returns a few sentences
    of sideline advice. The replay must be paused, so this pauses it for you."""
    # /api/coach-advice rejects a playing match (409), and the coach cannot ask
    # for the pause themselves mid-sentence. Pausing here makes "what should I do"
    # a single utterance instead of two.
    await _send_commands([_envelope("SET_PLAYING", {"playing": False})])

    url = f"{_http_url()}/api/coach-advice"
    timeout = httpx.Timeout(90.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url)
    if response.is_error:
        detail = response.text[:200]
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"coach advice failed ({response.status_code}): {detail}")
    data = response.json()
    return {"advice": data.get("advice", ""), "model": data.get("model", "")}


@mcp.tool
async def move_player(player_id: str, x_m: float, y_m: float) -> dict:
    """Move a player to a position on the pitch, in metres. The pitch is 105m
    long (x, 0=own goal-line side) by 68m wide (y). This enters edit mode and
    pauses the replay automatically. player_id is like 'H9' (home #9) or 'A10';
    use list_players to find ids. Returns the player's resulting position."""
    if not (0.0 <= x_m <= PITCH_LENGTH):
        raise ValueError(f"x_m must be within 0..{PITCH_LENGTH} metres")
    if not (0.0 <= y_m <= PITCH_WIDTH):
        raise ValueError(f"y_m must be within 0..{PITCH_WIDTH} metres")

    # Confirm the id exists so we can give a clear error instead of a silent no-op.
    state = await _get_state()
    if not any(p["id"] == player_id for p in state["players"]):
        valid = ", ".join(p["id"] for p in state["players"])
        raise ValueError(f"unknown player_id {player_id!r}. Valid ids: {valid}")

    board_x = x_m / PITCH_LENGTH
    board_y = y_m / PITCH_WIDTH
    s = await _send_commands(
        [
            _envelope(
                "DRAG_PLAYER_START",
                {"playerId": player_id, "boardX": board_x, "boardY": board_y},
            ),
            _envelope("DRAG_PLAYER_END", {"playerId": player_id}),
        ]
    )
    moved = next((p for p in s["players"] if p["id"] == player_id), None)
    return {
        "player": _trim_player(moved) if moved else None,
        "editMode": s["editMode"],
    }


if __name__ == "__main__":
    transport = os.environ.get("TC_MCP_TRANSPORT", "stdio")
    if transport == "http":
        port = int(os.environ.get("TC_MCP_PORT", "8765"))
        mcp.run(transport="http", host="127.0.0.1", port=port)
    else:
        mcp.run()

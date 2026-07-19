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
from pydantic import Field
from typing import Annotated
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


# Analysis-overlay buttons in the dashboard sidebar: name -> (command, snapshot
# field). The server side of each is a pure toggle, so the snapshot field is what
# lets set_overlay honour an explicit enabled=true/false.
OVERLAYS = {
    "offside": ("TOGGLE_OFFSIDE", "offsideOverlay"),
    "compactness": ("TOGGLE_COMPACTNESS", "compactnessOverlay"),
    "shadows": ("TOGGLE_SHADOWS", "shadowOverlay"),
    "pitch_control": ("TOGGLE_PITCH_CONTROL", "pitchControlOverlay"),
    "formation": ("TOGGLE_FORMATION", "formationOverlay"),
    "suggested": ("TOGGLE_SUGGESTED", "suggestedOverlay"),
}

# The three AI-experiment switches. Keys are the server's experiment names
# (server/state.py:EXPERIMENT_DEFAULTS); the aliases are what a person says.
EXPERIMENTS = {
    "passRecommendations": ("pass_recommendations", "passes"),
    "technicalIndicators": ("technical_indicators", "indicators"),
    "receiverTargets": ("receiver_targets", "targets"),
}
# Lookup is on a lowercased key, so the camelCase server name is folded too --
# a model that read the raw state and echoes "passRecommendations" still lands.
_EXPERIMENT_ALIASES = {
    alias.lower(): name
    for name, aliases in EXPERIMENTS.items()
    for alias in (name, *aliases)
}

# Speed notches for step_playback_speed. Matches the dashboard's own steps and
# stays inside the server's 0.1-4.0 clamp.
SPEED_LADDER = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

# The dashboard's one-click demo buttons: experiment settings, plus whether to
# play (True), pause (False), or leave playback alone (None).
DEMO_PRESETS = {
    "decision": ({"passRecommendations": True, "technicalIndicators": True,
                  "receiverTargets": False}, False),
    "shape": ({"passRecommendations": False, "technicalIndicators": True,
               "receiverTargets": False}, True),
    "movement": ({"passRecommendations": True, "technicalIndicators": True,
                  "receiverTargets": True}, False),
    "freeze": ({"technicalIndicators": True}, False),
    "clear": ({"passRecommendations": False, "technicalIndicators": False,
               "receiverTargets": False}, None),
}


# Enum-in-schema, str-at-runtime.
#
# The enum is what stops the model guessing a value that does not exist -- it is
# the difference between "suggest positions" reaching set_overlay(suggested) and
# falling through to whatever tool it does understand. The annotation stays `str`
# rather than Literal on purpose: speech transcription produces "pitch control"
# and "receiver targets", and the normalisation in each tool absorbs that. A
# Literal would reject those before the tool ever ran.
def _enum(values, description: str):
    return Annotated[str, Field(description=description, json_schema_extra={"enum": list(values)})]


OverlayName = _enum(
    OVERLAYS,
    "Which overlay to switch. 'suggested' draws ghost circles showing where "
    "players should be standing -- use it for any request to suggest, recommend "
    "or show better positions.",
)
ExperimentName = _enum(
    (aliases[0] for aliases in EXPERIMENTS.values()),
    "Which AI experiment switch to change.",
)
PresetName = _enum(DEMO_PRESETS, "Which canned demo view to apply.")
TeamName = _enum(("home", "away"), "Which side to coach.")
SpeedStep = _enum(("faster", "slower"), "Which way to step the replay speed.")


def _clock(media_time_ms: float) -> str:
    """m:ss, so a spoken confirmation can name the moment."""
    total = max(0, int(media_time_ms // 1000))
    return f"{total // 60}:{total % 60:02d}"

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


async def _drain_to_pong(ws) -> dict | None:
    """
    Send a PING and read until the PONG, returning the last STATE_SNAPSHOT seen.

    The barrier matters. handle_command() broadcasts a snapshot at the end of
    every command, but the playback loop *also* broadcasts on its own tick, so a
    snapshot already in flight can arrive first -- reading "the next snapshot"
    after a command reports pre-command state. PING is answered inline and in
    order, so once PONG lands, every command we sent before it has been handled
    and its snapshot already queued ahead of the PONG. The last snapshot before
    the PONG is therefore the post-command one.
    """
    await ws.send(_dumps(_envelope("PING", {"t": time.time() * 1000.0})))
    latest = None
    while True:
        import json

        data = json.loads(await ws.recv())
        kind = data.get("type")
        if kind == "STATE_SNAPSHOT":
            latest = data["payload"]
        elif kind == "PONG":
            return latest
        elif kind == "ERROR":
            raise RuntimeError(f"server error: {data['payload'].get('reason')}")


async def _run_commands(build) -> dict:
    """
    Open one connection, send the commands `build` derives from the current
    snapshot, and return the state that resulted from them.

    One connection per call, for two reasons: it keeps drag ownership stable
    across a START/END pair (the server scopes ownership per-connection), and it
    lets a command depend on freshly-read state without another client's toggle
    slipping in between the read and the write.
    """
    async with websockets.connect(_ws_url()) as ws:
        snapshot = await _recv_snapshot(ws)  # the on-connect snapshot
        envelopes = build(snapshot)
        if not envelopes:
            return snapshot
        for env in envelopes:
            await ws.send(_dumps(env))
        return await _drain_to_pong(ws) or await _recv_snapshot(ws)


async def _send_commands(envelopes: list[dict]) -> dict:
    """Send a fixed list of commands and return the resulting state."""
    return await _run_commands(lambda _snapshot: envelopes)


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
async def set_playback_speed(rate: float) -> dict:
    """Set how fast the replay plays. 1.0 is normal speed, 0.5 is half speed,
    2.0 is double. Clamped to 0.1-4.0. Use this when the coach names a speed
    ('half speed', 'double speed', 'back to normal'); use step_playback_speed
    for a bare 'faster' or 'slower'."""
    s = await _send_commands([_envelope("SET_PLAYBACK_RATE", {"rate": float(rate)})])
    return {"playbackRate": s["playbackRate"], "playing": s["playing"]}


@mcp.tool
async def step_playback_speed(direction: SpeedStep) -> dict:
    """Make the replay one step faster or slower. Use this for a bare 'faster',
    'slower', 'speed it up', 'slow it down' -- it reads the current speed and
    moves one notch along 0.25, 0.5, 1, 1.5, 2, 3, 4, so it works without
    knowing what the speed is now."""
    want_faster = direction.strip().lower() in ("faster", "up", "quicker", "speed up")

    def build(snapshot: dict) -> list[dict]:
        current = float(snapshot.get("playbackRate") or 1.0)
        if want_faster:
            nxt = next((s for s in SPEED_LADDER if s > current + 1e-6), SPEED_LADDER[-1])
        else:
            nxt = next((s for s in reversed(SPEED_LADDER) if s < current - 1e-6), SPEED_LADDER[0])
        if abs(nxt - current) < 1e-6:
            return []  # already at the end of the ladder
        return [_envelope("SET_PLAYBACK_RATE", {"rate": nxt})]

    s = await _run_commands(build)
    return {"playbackRate": s["playbackRate"]}


@mcp.tool
async def skip_time(seconds: float = 10.0) -> dict:
    """Jump the replay forward or backward. Positive seconds skip forward,
    negative skip back -- so 'skip ahead' is 10 and 'go back ten seconds' is
    -10. Defaults to 10 seconds. Clamped at the start of the match."""
    delta_ms = float(seconds) * 1000.0

    def build(snapshot: dict) -> list[dict]:
        target = max(0.0, float(snapshot.get("mediaTimeMs") or 0.0) + delta_ms)
        return [_envelope("SEEK_TO", {"mediaTimeMs": target})]

    s = await _run_commands(build)
    return {
        "mediaTimeMs": s["mediaTimeMs"],
        "frameIndex": s["frameIndex"],
        "clock": _clock(s["mediaTimeMs"]),
    }


@mcp.tool
async def start_calibration() -> dict:
    """Start field calibration — the dashboard's 'Calibrate field' button. Shows
    the four magenta targets and asks the vision worker to capture the corner
    clicks. Fails if no camera/vision worker is running."""
    s = await _send_commands([_envelope("START_CALIBRATION")])
    return {"calibration": s.get("calibration"), "calibrationOverlay": s["calibrationOverlay"]}


@mcp.tool
async def cancel_calibration() -> dict:
    """Cancel an in-progress field calibration and hide the marker overlay."""
    s = await _send_commands([_envelope("CANCEL_CALIBRATION")])
    return {"calibration": s.get("calibration"), "calibrationOverlay": s["calibrationOverlay"]}


@mcp.tool
async def set_overlay(overlay: OverlayName, enabled: bool | None = None) -> dict:
    """Turn a tactical analysis overlay on or off on the projected pitch. These
    are the dashboard's Analysis buttons. Valid overlay names:
      'offside'       — the offside line
      'compactness'   — line-to-line distances and team width
      'shadows'       — defender reach shadows
      'pitch_control' — the pitch-control / space-ownership map
      'formation'     — the inferred formation for each team
      'suggested'     — ghost circles for suggested player positions
    Pass enabled=true/false to force a state, or omit it to toggle. Returns the
    resulting on/off state of every overlay."""
    key = overlay.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in OVERLAYS:
        raise ValueError(f"unknown overlay {overlay!r}. Valid: {', '.join(OVERLAYS)}")
    command, field = OVERLAYS[key]

    def build(snapshot: dict) -> list[dict]:
        if enabled is not None and bool(snapshot.get(field)) == enabled:
            return []  # already in the requested state; toggling would undo it
        return [_envelope(command)]

    s = await _run_commands(build)
    return {name: bool(s.get(f)) for name, (_, f) in OVERLAYS.items()}


@mcp.tool
async def set_experiment(experiment: ExperimentName, enabled: bool = True) -> dict:
    """Turn one of the three AI experiment switches on or off. These are the
    dashboard's AI experiments panel. Valid names:
      'pass_recommendations' — rank the available passes for the player on the ball
      'technical_indicators' — the numeric indicator panel (line height, xT, space)
      'receiver_targets'     — movement targets for potential receivers
    Returns the resulting state of all three."""
    key = _EXPERIMENT_ALIASES.get(experiment.strip().lower().replace("-", "_").replace(" ", "_"))
    if key is None:
        valid = ", ".join(aliases[0] for aliases in EXPERIMENTS.values())
        raise ValueError(f"unknown experiment {experiment!r}. Valid: {valid}")
    s = await _send_commands(
        [_envelope("SET_EXPERIMENT", {"name": key, "enabled": bool(enabled)})]
    )
    return s["experiments"]


@mcp.tool
async def run_demo_preset(preset: PresetName) -> dict:
    """Set up a canned demo view in one call — the dashboard's demo buttons.
      'decision' — Demo 1: pause and rank the available passes
      'shape'    — Demo 2: live team shape while the replay plays
      'movement' — Demo 3: pause with pass ranking plus receiver movement targets
      'freeze'   — freeze the frame with the indicator panel on
      'clear'    — switch every experiment off, leaving playback as it is
    Use this instead of several set_experiment calls when the user asks for one
    of the demos."""
    key = preset.strip().lower()
    if key not in DEMO_PRESETS:
        raise ValueError(f"unknown preset {preset!r}. Valid: {', '.join(DEMO_PRESETS)}")
    settings, playing = DEMO_PRESETS[key]
    commands = [
        _envelope("SET_EXPERIMENT", {"name": name, "enabled": value})
        for name, value in settings.items()
    ]
    if playing is not None:
        commands.append(_envelope("SET_PLAYING", {"playing": playing}))
    s = await _send_commands(commands)
    return {"experiments": s["experiments"], "playing": s["playing"]}


@mcp.tool
async def list_matches() -> dict:
    """List the test matches that can be loaded, and which one is active. Use
    this to find the matchId for load_match."""
    s = await _get_state()
    return {"active": s["matchId"], "matches": s["availableMatches"]}


@mcp.tool
async def load_match(match_id: str) -> dict:
    """Switch the board to a different test match. This rewinds to kickoff and
    clears any manual player edits. Call list_matches first for valid ids."""
    state = await _get_state()
    available = [m["id"] for m in state["availableMatches"]]
    if match_id not in available:
        raise ValueError(f"unknown match_id {match_id!r}. Valid ids: {', '.join(available)}")
    s = await _send_commands([_envelope("LOAD_MATCH", {"matchId": match_id})])
    return {"matchId": s["matchId"], "matchLabel": s["matchLabel"], "frameIndex": s["frameIndex"]}


@mcp.tool
async def set_coaching_team(team: TeamName) -> dict:
    """Choose which side the coaching advice and suggested positions are written
    for — the dashboard's 'Switch side' button. Pass 'home' or 'away'. Changing
    sides re-frames every subsequent get_coach_advice answer."""
    key = team.strip().lower()
    if key not in ("home", "away"):
        raise ValueError(f"team must be 'home' or 'away', got {team!r}")
    s = await _send_commands([_envelope("SET_COACHING_TEAM", {"team": key})])
    return {"coachingTeam": s["coachingTeam"], "possession": s["possession"]}


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

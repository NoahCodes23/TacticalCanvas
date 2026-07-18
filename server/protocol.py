from typing import Any, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1
CommandType = Literal[
    "SET_PLAYING",              # {playing: bool}
    "SET_PLAYBACK_TIME",        # {mediaTimeMs, playing?}  -- video is the clock
    "ENTER_EDIT_MODE",          # {}
    "EXIT_EDIT_MODE",           # {}
    "DRAG_PLAYER_START",        # {playerId, boardX, boardY}
    "DRAG_PLAYER_MOVE",         # {playerId, boardX, boardY}
    "DRAG_PLAYER_END",          # {playerId}
    "RESET_SCENARIO",           # {}
    "TOGGLE_CALIBRATION",       # {}
    "START_CALIBRATION",        # {}
    "CANCEL_CALIBRATION",       # {}
    "TOGGLE_OFFSIDE",           # {}
    "TOGGLE_COMPACTNESS",       # {}  -- line-to-line + width readout
    "TOGGLE_SHADOWS",           # {}  -- defender reach shadows
    "TOGGLE_PITCH_CONTROL",     # {}  -- arrival/nearest-player control map
    "TOGGLE_FORMATION",         # {}  -- inferred team formations
    "TOGGLE_SUGGESTED",         # {}  -- ghost circles for suggested positions
    "SET_COACHING_TEAM",        # {team: 'home'|'away'}
    "TOGGLE_COACHING_TEAM",     # {}
    "SET_SHADOW_SECONDS",       # {seconds}  -- reach horizon, clamped 0.5-4.0
    "SET_EXPERIMENT",           # {name, enabled?} -- opt-in analytics feature
    "LOAD_MATCH",               # {matchId}  -- switch the active test match
    "PING",                     # {t}  -- client clock, echoed back for RTT
]

ServerMessageType = Literal["STATE_SNAPSHOT", "VISION_UPDATE", "ERROR", "PONG"]

class Envelope(BaseModel):
    protocolVersion: int = PROTOCOL_VERSION
    scenarioId: str = "demo"
    clientId: str = "unknown"
    sequenceNumber: int = 0
    timestamp: float = 0.0
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)

def server_message(
    msg_type: ServerMessageType,
    payload: dict[str, Any],
    scenario_id: str,
    sequence: int,
    timestamp_ms: float,
) -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "scenarioId": scenario_id,
        "clientId": "server",
        "sequenceNumber": sequence,
        "timestamp": timestamp_ms,
        "type": msg_type,
        "payload": payload,
    }

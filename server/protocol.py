from typing import Any, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1
CommandType = Literal[
    "SET_PLAYING",              # {playing: bool}
    "SEEK_TO",                  # {mediaTimeMs}  -- absolute seek, server clock
    "SET_PLAYBACK_RATE",        # {rate}  -- replay speed, clamped 0.1-4.0
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
    "TOGGLE_SUGGESTED",         # {}  -- suggested off-ball ghost positions
    "SET_COACHING_TEAM",        # {team}  -- pick the side the coach is on
    "TOGGLE_COACHING_TEAM",     # {}  -- flip the coached side
    "SELECT_COACH",             # {coachId}  -- pick the coach persona/playstyle
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

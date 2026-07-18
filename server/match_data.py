"""Match tracking data: a registry of preprocessed .npz matches plus a small
synthetic fallback so the app still runs before any data is prepared.

The .npz files are produced by ``python -m tools.prepare_match`` from the
Metrica sample games and live in ``cache/`` (git-ignored). Each is exposed as a
selectable match; the dashboard's "Test Match 1/2/3" buttons pick between them.
"""

import bisect
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Penalty area: 16.5m deep, 40.32m wide, centred on the goal.
BOX_DEPTH = 16.5
BOX_Y_MIN = PITCH_WIDTH / 2 - 20.16
BOX_Y_MAX = PITCH_WIDTH / 2 + 20.16

# Event types that mean a player actually played the ball. Cards and fouls
# carry a position but aren't touches.
_TOUCH_TYPES = {"PASS", "SHOT", "RECOVERY", "BALL LOST", "CHALLENGE", "SET PIECE", "CARRY"}

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / "cache"

_HOME_FORMATION = [
    (5.0, 34.0, 1),      # GK
    (20.0, 10.0, 2),     # RB
    (18.0, 26.0, 5),     # CB
    (18.0, 42.0, 6),     # CB
    (20.0, 58.0, 3),     # LB
    (38.0, 20.0, 8),     # RCM
    (35.0, 34.0, 4),     # CDM
    (38.0, 48.0, 10),    # LCM
    (60.0, 12.0, 7),     # RW
    (58.0, 34.0, 9),     # ST
    (60.0, 56.0, 11),    # LW
]


@dataclass
class Player:
    id: str
    team: str
    number: int
    x: float
    y: float
    home_x: float   # formation anchor used only by the synthetic fallback
    home_y: float
    edited: bool = False  # True once a coach has dragged this player
    vx: float = 0.0       # metres/second, from the tracking data
    vy: float = 0.0


# --------------------------------------------------------------------------- #
# match events -> display feed
# --------------------------------------------------------------------------- #
# Turn Metrica's raw (type, subtype) into short human labels for the live feed.
# `kind` is a style bucket the frontend colours by. Wording lives here (not in
# the cache) so it can change without re-running tools/prepare_match.
def _shirt(name: str) -> str:
    m = re.search(r"(\d+)\s*$", name or "")
    return f"#{m.group(1)}" if m else ""


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _label_kind(typ: str, sub: str) -> tuple[str, str]:
    typ, sub = typ.upper(), sub.upper()
    if typ == "SHOT":
        return ("GOAL", "goal") if "GOAL" in sub else ("Shot", "shot")
    if typ == "SET PIECE":
        for key, lab in (("KICK OFF", "Kick-off"), ("CORNER", "Corner"),
                         ("THROW IN", "Throw-in"), ("GOAL KICK", "Goal kick"),
                         ("FREE KICK", "Free kick"), ("PENALTY", "Penalty")):
            if key in sub:
                return lab, "setpiece"
        return "Set piece", "setpiece"
    if typ == "PASS":
        return ("Cross", "pass") if "CROSS" in sub else ("Pass", "pass")
    if typ == "BALL LOST":
        return "Ball lost", "defense"
    if typ == "RECOVERY":
        if "INTERCEPTION" in sub:
            return "Interception", "defense"
        if "THEFT" in sub:
            return "Steal", "defense"
        return "Recovery", "defense"
    if typ == "CHALLENGE":
        if "TACKLE" in sub:
            return "Tackle", "defense"
        if "AERIAL" in sub:
            return "Aerial duel", "defense"
        return "Challenge", "defense"
    if typ == "BALL OUT":
        return "Ball out", "other"
    if typ == "FAULT RECEIVED":
        return "Foul", "foul"
    if typ == "CARD":
        if "RED" in sub:
            return "Red card", "foul"
        if "YELLOW" in sub:
            return "Yellow card", "foul"
        return "Card", "foul"
    return (typ.title() or "Event"), "other"


def _format_event(e: dict) -> dict:
    frm, to = _shirt(e.get("from", "")), _shirt(e.get("to", ""))
    label, kind = _label_kind(e.get("type", ""), e.get("subtype", ""))
    detail = f"{frm} → {to}" if frm and to else frm
    t = float(e.get("t", 0.0))
    return {
        "t": round(t, 2),
        "clock": _clock(t),
        "team": e.get("team", "home"),
        "label": label,
        "detail": detail,
        "kind": kind,
    }


def _empty_stats() -> dict:
    return {
        "goals": 0,
        "passesCompleted": 0,
        "passes": 0,
        "touchesInBox": 0,
        "accuratePasses": 0,
        "corners": 0,
        "offsides": 0,
        "throws": 0,
    }


class MatchTracks:
    """Wraps one .npz that tools/prepare_match.py writes."""

    def __init__(
        self,
        match_id: str,
        label: str,
        timestamps: np.ndarray,
        positions: np.ndarray,       # (frames, players, 2) metres
        ball_positions: np.ndarray,  # (frames, 2) metres
        player_teams: np.ndarray,    # (players,) 0=home, 1=away
        player_numbers: np.ndarray,  # (players,)
        fps: float,
        events: list[dict] | None = None,
    ) -> None:
        self.match_id = match_id
        self.label = label
        self.timestamps = timestamps
        self.positions = positions.astype(np.float32, copy=False)
        self.ball_positions = ball_positions.astype(np.float32, copy=False)
        self.player_teams = player_teams
        self.player_numbers = player_numbers
        self.fps = float(fps) if fps > 0 else 25.0
        self.n_frames = int(positions.shape[0])
        self.n_players = int(positions.shape[1])
        self.duration = self.n_frames / self.fps

        # Display-ready events, sorted by start time, with a parallel time list
        # for bisecting "everything up to the current replay clock".
        raw = sorted(events or [], key=lambda e: float(e.get("t", 0.0)))
        self._raw_events = raw
        self.events = [_format_event(e) for e in raw]
        self._event_times = [e["t"] for e in self.events]
        self._attack_dir: np.ndarray | None = None  # built on first stats call

    @classmethod
    def load(cls, path: Path, match_id: str) -> "MatchTracks":
        with np.load(path, allow_pickle=False) as d:
            fps = float(d["fps"]) if "fps" in d.files else 25.0
            label = str(d["label"]) if "label" in d.files else match_id
            events: list[dict] = []
            if "events_json" in d.files:
                try:
                    events = json.loads(str(d["events_json"]))
                except (ValueError, TypeError):
                    events = []
            return cls(
                match_id=match_id,
                label=label,
                timestamps=d["timestamps"],
                positions=d["positions"],
                ball_positions=d["ball_positions"],
                player_teams=d["player_teams"],
                player_numbers=d["player_numbers"],
                fps=fps,
                events=events,
            )

    def _frame_at(self, t_sec: float) -> int:
        if self.duration <= 0:
            return 0
        t = t_sec % self.duration          # loop the replay
        f = int(t * self.fps)
        return max(0, min(f, self.n_frames - 1))

    def build_players(self) -> list[Player]:
        out: list[Player] = []
        for i in range(self.n_players):
            team = "home" if int(self.player_teams[i]) == 0 else "away"
            num = int(self.player_numbers[i])
            pid = f"{'H' if team == 'home' else 'A'}{num}"
            x = float(self.positions[0, i, 0])
            y = float(self.positions[0, i, 1])
            out.append(Player(id=pid, team=team, number=num, x=x, y=y, home_x=x, home_y=y))
        return out

    def _velocity(self, f: int, i: int) -> tuple[float, float]:
        """Central difference over +/-2 frames (160ms at 25Hz). Wide enough not
        to amplify tracking jitter into a wildly swinging vector, narrow enough
        to still show a player planting and turning."""
        a = max(0, f - 2)
        b = min(self.n_frames - 1, f + 2)
        dt = (b - a) / self.fps
        if dt <= 0:
            return 0.0, 0.0
        return (
            float(self.positions[b, i, 0] - self.positions[a, i, 0]) / dt,
            float(self.positions[b, i, 1] - self.positions[a, i, 1]) / dt,
        )

    def apply(self, players: list[Player], t_sec: float) -> None:
        f = self._frame_at(t_sec)
        for i, p in enumerate(players):
            if i >= self.n_players:
                continue
            if p.edited:
                p.vx = p.vy = 0.0  # coach-placed: no momentum to carry
                continue
            p.x = float(self.positions[f, i, 0])
            p.y = float(self.positions[f, i, 1])
            p.vx, p.vy = self._velocity(f, i)

    def ball_at(self, t_sec: float) -> tuple[float, float]:
        f = self._frame_at(t_sec)
        return float(self.ball_positions[f, 0]), float(self.ball_positions[f, 1])

    def events_upto(self, t_sec: float, limit: int = 8) -> list[dict]:
        """The most recent `limit` events that have happened by the current
        replay clock, newest first. Uses the same looped clock as the tracking
        so the feed stays in sync when the replay wraps."""
        if not self.events:
            return []
        t = t_sec % self.duration if self.duration > 0 else t_sec
        i = bisect.bisect_right(self._event_times, t)
        return list(reversed(self.events[max(0, i - limit):i]))

    def _home_attacks_positive(self) -> np.ndarray:
        """Per-frame boolean: is the home team attacking towards +x?

        Metrica does not flip coordinates at half time, and the .npz carries no
        period column, so any fixed direction is wrong for one half of every
        match. Derive it from the shape instead: the team defending the left
        goal keeps its back line low, so the side with the *lower* centroid x is
        the one attacking +x. Averaged over a minute, a counter-attack can't
        flip the sign, but the half-time swap still shows up cleanly.
        """
        if self._attack_dir is not None:
            return self._attack_dir

        home_cols = self.player_teams == 0
        away_cols = self.player_teams == 1
        if not home_cols.any() or not away_cols.any():
            self._attack_dir = np.ones(self.n_frames, dtype=bool)
            return self._attack_dir

        d = (self.positions[:, home_cols, 0].mean(axis=1)
             - self.positions[:, away_cols, 0].mean(axis=1))

        # Centred moving average via cumsum -- O(frames), not O(frames*window).
        w = max(1, int(60 * self.fps))
        c = np.concatenate([[0.0], np.cumsum(d, dtype=np.float64)])
        idx = np.arange(d.size)
        lo = np.maximum(0, idx - w // 2)
        hi = np.minimum(d.size, idx + w // 2 + 1)
        self._attack_dir = (c[hi] - c[lo]) / (hi - lo) < 0
        return self._attack_dir

    def _in_opponent_box(self, t_ev: float, team: str) -> bool:
        """Was the ball inside the opponent's penalty area at this event?

        The prepared events carry no coordinates, so use the tracking ball
        position at the event's frame -- which is where the touch happened.
        """
        f = self._frame_at(t_ev)
        bx = float(self.ball_positions[f, 0])
        by = float(self.ball_positions[f, 1])
        if not (BOX_Y_MIN <= by <= BOX_Y_MAX):
            return False
        home_positive = bool(self._home_attacks_positive()[f])
        attacks_positive = home_positive if team == "home" else not home_positive
        return bx >= PITCH_LENGTH - BOX_DEPTH if attacks_positive else bx <= BOX_DEPTH

    def stats_upto(self, t_sec: float) -> dict:
        """Aggregate match stats up to the current replay clock."""
        if not self.events:
            return {"home": _empty_stats(), "away": _empty_stats()}
        t = t_sec % self.duration if self.duration > 0 else t_sec
        i = bisect.bisect_right(self._event_times, t)
        home = _empty_stats()
        away = _empty_stats()
        # self.events and self._raw_events are parallel and share a sort order,
        # so one bisect on the formatted times indexes both.
        for ev, raw_ev in zip(self.events[:i], self._raw_events[:i]):
            team = ev.get("team", "home")
            bucket = home if team == "home" else away
            label = ev.get("label", "")
            if label == "GOAL":
                bucket["goals"] += 1
            elif label in ("Pass", "Cross"):
                bucket["passes"] += 1
            elif label == "Corner":
                bucket["corners"] += 1
            elif label == "Throw-in":
                bucket["throws"] += 1

            typ = (raw_ev.get("type") or "").upper()
            sub = (raw_ev.get("subtype") or "").upper()
            if "OFFSIDE" in sub:
                bucket["offsides"] += 1
            if typ in _TOUCH_TYPES and raw_ev.get("from"):
                if self._in_opponent_box(ev["t"], team):
                    bucket["touchesInBox"] += 1
        # Estimate incomplete passes: ~12% failure rate for demo realism
        for bucket in (home, away):
            total = bucket["passes"]
            bucket["passesCompleted"] = max(0, total - total // 8)
            bucket["accuratePasses"] = bucket["passesCompleted"]
        return {"home": home, "away": away}


# --------------------------------------------------------------------------- #
# registry + current selection
# --------------------------------------------------------------------------- #
_current: MatchTracks | None = None
_current_id: str | None = None
_initialized = False


def _npz_label(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as d:
            if "label" in d.files:
                return str(d["label"])
    except Exception:
        pass
    return path.stem


def list_matches() -> list[dict]:
    """Every prepared match the server can offer, sorted by id."""
    if not _CACHE_DIR.is_dir():
        return []
    out = []
    for p in sorted(_CACHE_DIR.glob("match_*.npz")):
        out.append({"id": p.stem, "label": _npz_label(p)})
    return out


def current_id() -> str | None:
    _ensure_initialized()
    return _current_id


def current_label() -> str | None:
    _ensure_initialized()
    return _current.label if _current is not None else None


def select(match_id: str) -> bool:
    """Load and make `match_id` the active match. False if it can't be loaded."""
    global _current, _current_id, _initialized
    path = _CACHE_DIR / f"{match_id}.npz"
    if not path.exists():
        print(f"[match_data] no such match: {match_id} ({path})")
        return False
    try:
        _current = MatchTracks.load(path, match_id)
        _current_id = match_id
        _initialized = True
        print(
            f"[match_data] selected {match_id} ({_current.label}): "
            f"{_current.n_players} players, {_current.n_frames} frames @ "
            f"{_current.fps:.0f}fps ({_current.duration:.0f}s)"
        )
        return True
    except Exception as e:
        print(f"[match_data] could not load {path}: {e}")
        return False


def _ensure_initialized() -> None:
    """Pick a default match on first use: env override, else the first prepared."""
    global _initialized
    if _initialized:
        return
    _initialized = True  # only try once; falls through to synthetic if nothing loads
    env = os.environ.get("TC_MATCH")
    if env:
        if select(env):
            return
    matches = list_matches()
    if matches:
        select(matches[0]["id"])
    else:
        print("[match_data] no prepared matches in cache/ -- using synthetic fallback")


# --------------------------------------------------------------------------- #
# public API used by AppState
# --------------------------------------------------------------------------- #
def build_players() -> list[Player]:
    _ensure_initialized()
    if _current is not None:
        return _current.build_players()
    return _synthetic_players()


def advance(players: list[Player], t_sec: float) -> None:
    _ensure_initialized()
    if _current is not None:
        _current.apply(players, t_sec)
        return
    for i, p in enumerate(players):
        if p.edited:
            p.vx = p.vy = 0.0
            continue
        phase = i * 0.7
        p.x = p.home_x + 2.5 * math.sin(t_sec * 0.45 + phase)
        p.y = p.home_y + 1.8 * math.sin(t_sec * 0.31 + phase * 1.3)
        p.vx = 2.5 * 0.45 * math.cos(t_sec * 0.45 + phase)
        p.vy = 1.8 * 0.31 * math.cos(t_sec * 0.31 + phase * 1.3)


def ball_position(t_sec: float) -> tuple[float, float]:
    _ensure_initialized()
    if _current is not None:
        return _current.ball_at(t_sec)
    return (
        PITCH_LENGTH / 2 + 22.0 * math.sin(t_sec * 0.35),
        PITCH_WIDTH / 2 + 14.0 * math.sin(t_sec * 0.7),
    )


def recent_events(t_sec: float, limit: int = 8) -> list[dict]:
    """Recent match events by the current replay clock, newest first. Empty for
    the synthetic fallback (no event data)."""
    _ensure_initialized()
    if _current is not None:
        return _current.events_upto(t_sec, limit)
    return []


def match_stats(t_sec: float) -> dict:
    """Aggregated match stats up to the current replay clock."""
    _ensure_initialized()
    if _current is not None:
        return _current.stats_upto(t_sec)
    return {"home": _empty_stats(), "away": _empty_stats()}


def _synthetic_players() -> list[Player]:
    players: list[Player] = []
    for x, y, num in _HOME_FORMATION:
        players.append(Player(id=f"H{num}", team="home", number=num, x=x, y=y, home_x=x, home_y=y))
    for x, y, num in _HOME_FORMATION:
        mx, my = PITCH_LENGTH - x, PITCH_WIDTH - y
        players.append(Player(id=f"A{num}", team="away", number=num, x=mx, y=my, home_x=mx, home_y=my))
    return players

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MATCH_PATH = _ROOT / "cache" / "demo_match.npz"

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

    """Wraps the .npz that prepare_match.py writes."""

    def __init__(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,      # (frames, players, 2)
        ball_positions: np.ndarray, # (frames, 2)
        player_teams: np.ndarray,   # (players,) 0=home, 1=away
        player_numbers: np.ndarray, # (players,)
        fps: float,
    ) -> None:
        self.timestamps = timestamps
        self.positions = positions.astype(np.float32, copy=False)
        self.ball_positions = ball_positions.astype(np.float32, copy=False)
        self.player_teams = player_teams
        self.player_numbers = player_numbers
        self.fps = float(fps) if fps > 0 else 25.0
        self.n_frames = int(positions.shape[0])
        self.n_players = int(positions.shape[1])
        self.duration = self.n_frames / self.fps

    @classmethod
    def load(cls, path: str | os.PathLike) -> "MatchTracks":
        with np.load(path) as d:
            fps = float(d["fps"]) if "fps" in d.files else 25.0
            return cls(
                timestamps=d["timestamps"],
                positions=d["positions"],
                ball_positions=d["ball_positions"],
                player_teams=d["player_teams"],
                player_numbers=d["player_numbers"],
                fps=fps,
            )
        
    def _frame_at(self, t_sec: float) -> int:
        if self.duration <= 0:
            return 0
        t = t_sec % self.duration
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
            out.append(
                Player(id=pid, team=team, number=num, x=x, y=y, home_x=x, home_y=y)
            )
        return out
    
    def apply(self, players: list[Player], t_sec: float) -> None:
        f = self._frame_at(t_sec)
        for i, p in enumerate(players):
            if i >= self.n_players:
                break
            if p.edited:
                continue
            p.x = float(self.positions[f, i, 0])
            p.y = float(self.positions[f, i, 1])

    def ball_at(self, t_sec: float) -> tuple[float, float]:
        f = self._frame_at(t_sec)
        return float(self.ball_positions[f, 0]), float(self.ball_positions[f, 1])

_TRACKS: MatchTracks | None = None
_TRIED = False

def _get_tracks() -> MatchTracks | None:
    global _TRACKS, _TRIED
    if _TRIED:
        return _TRACKS
    _TRIED = True

    env = os.environ.get("TC_MATCH")
    if env == "":
        return None  # explicitly opted out
    path = env if env else str(_DEFAULT_MATCH_PATH)

    if not os.path.exists(path):
        print(f"[match_data] no recorded tracks at {path} -- using synthetic fallback")
        return None

    try:
        _TRACKS = MatchTracks.load(path)
        print(
            f"[match_data] loaded {path}: {_TRACKS.n_players} players, "
            f"{_TRACKS.n_frames} frames @ {_TRACKS.fps:.1f}fps "
            f"({_TRACKS.duration:.1f}s of match)"
        )
    except Exception as e:
        print(f"[match_data] could not load {path}: {e} -- using synthetic fallback")
        _TRACKS = None
    return _TRACKS


def build_players() -> list[Player]:
    tracks = _get_tracks()
    if tracks is not None:
        return tracks.build_players()
    return _synthetic_players()


def advance(players: list[Player], t_sec: float) -> None:
    tracks = _get_tracks()
    if tracks is not None:
        tracks.apply(players, t_sec)
        return
    for i, p in enumerate(players):
        if p.edited:
            continue
        phase = i * 0.7
        p.x = p.home_x + 2.5 * math.sin(t_sec * 0.45 + phase)
        p.y = p.home_y + 1.8 * math.sin(t_sec * 0.31 + phase * 1.3)


def ball_position(t_sec: float) -> tuple[float, float]:
    tracks = _get_tracks()
    if tracks is not None:
        return tracks.ball_at(t_sec)
    return (
        PITCH_LENGTH / 2 + 22.0 * math.sin(t_sec * 0.35),
        PITCH_WIDTH / 2 + 14.0 * math.sin(t_sec * 0.7),
    )


def _synthetic_players() -> list[Player]:
    players: list[Player] = []
    for x, y, num in _HOME_FORMATION:
        players.append(
            Player(id=f"H{num}", team="home", number=num, x=x, y=y, home_x=x, home_y=y)
        )
    for x, y, num in _HOME_FORMATION:
        mx, my = PITCH_LENGTH - x, PITCH_WIDTH - y
        players.append(
            Player(id=f"A{num}", team="away", number=num, x=mx, y=my, home_x=mx, home_y=my)
        )
    return players
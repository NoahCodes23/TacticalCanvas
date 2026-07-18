"""Suggested off-ball positions.

For every player on the possession team that isn't currently the ball carrier,
propose a nearby spot that they would credibly own more than an opponent. The
ghost circles compare "where you are" against "where a defensible support
position is within your movement budget".

Not a trained model. Bounded gradient-ascent search over the same
_arrival_time / pitch_control_at heuristic the rest of the tactical overlays
already use. Notes on the objective:

* **Score = pitch control at the destination itself.**  An earlier version
  scored control-at-the-ball, but that quantity is dominated by whoever is
  already nearest the ball -- a defender moving 5m barely changed it, and no
  ghost ever cleared the improvement threshold. Scoring at the destination
  directly answers the coach's question: "if I go there, do I own it or does
  a defender?" The gain vs. staying put is what shows on the label.

* **Movement budget is 5m.**  A player can only cover ~5m in the time a pass
  takes to arrive. Suggesting they teleport 30m away is a screenshot, not
  advice. Their current velocity still counts through _arrival_time, so a
  player already running toward a spot pays less to reach it than a standing
  one.
"""

from __future__ import annotations

import math
from typing import Any

from .experimental import _team_players, pitch_control_at

# Search grid -- see module docstring for why these values.
MOVE_BUDGET_M = 5.0
RADIAL_SAMPLES = 4          # 1.25m, 2.5m, 3.75m, 5m from current position
ANGULAR_SAMPLES = 12        # every 30 degrees
REFINE_SAMPLES = 8          # local refine around the best coarse hit
REFINE_RADIUS_M = 1.25

# Ghost circle thresholds. Sub-threshold gains just add visual noise.
MIN_GAIN = 0.05             # ~5 percentage points of local pitch control
MIN_MOVE_M = 1.0            # too-tiny moves feel like a jitter, not a call


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _mutated(base: Any, x: float, y: float) -> Any:
    """Shallow-copy a player-like object with (x, y) overridden.

    ``pitch_control_at`` inspects ``.x``, ``.y``, ``.team``, and ``.vx``/``.vy``
    via ``_arrival_time``. We only shift position -- keeping the velocity means
    the arrival-time model still accounts for a player running toward the spot,
    which is the whole point of budgeting the move.
    """
    # SimpleNamespace / dataclass / plain object -- copy attributes rather than
    # relying on ``copy.replace``, which requires dataclasses.
    class _P:
        __slots__ = ("id", "team", "number", "x", "y", "vx", "vy")
    p = _P()
    p.id = getattr(base, "id", "")
    p.team = getattr(base, "team", "")
    p.number = getattr(base, "number", 0)
    p.x = x
    p.y = y
    p.vx = float(getattr(base, "vx", 0.0))
    p.vy = float(getattr(base, "vy", 0.0))
    return p


def _score_at_destination(
    players: list[Any],
    swap_id: str,
    x: float,
    y: float,
    team: str,
) -> float:
    """Team pitch control AT (x, y) if the swap_id player is moved to (x, y).
    That player's arrival time collapses to 0, so this is essentially "how
    dominant is my ownership of this spot vs. the nearest opponent"."""
    swapped = [
        _mutated(p, x, y) if p.id == swap_id else p
        for p in players
    ]
    return pitch_control_at(swapped, team, x, y)


def _search_one(
    players: list[Any],
    player: Any,
    team: str,
    pitch_length: float,
    pitch_width: float,
) -> tuple[float, float, float] | None:
    """Return (x, y, gain) or None if the best candidate isn't worth drawing.

    Baseline is control at the candidate spot *without* moving anyone; score
    is control at that same spot *with* this player standing on it. Gain > 0
    means "I would create a defensible spot the team doesn't currently own."
    """
    cx, cy = float(player.x), float(player.y)

    best_x, best_y, best_gain = cx, cy, 0.0
    for i in range(1, RADIAL_SAMPLES + 1):
        r = MOVE_BUDGET_M * i / RADIAL_SAMPLES
        for j in range(ANGULAR_SAMPLES):
            theta = 2.0 * math.pi * j / ANGULAR_SAMPLES
            x = _clamp(cx + r * math.cos(theta), 0.5, pitch_length - 0.5)
            y = _clamp(cy + r * math.sin(theta), 0.5, pitch_width - 0.5)
            baseline = pitch_control_at(players, team, x, y)
            if baseline > 0.95:
                # Already essentially owned by us -- no ghost needed here.
                continue
            score = _score_at_destination(players, player.id, x, y, team)
            gain = score - baseline
            if gain > best_gain:
                best_x, best_y, best_gain = x, y, gain

    # Local refine around the coarse best (only if the coarse pass found one).
    if best_gain > 0.0:
        for j in range(REFINE_SAMPLES):
            theta = 2.0 * math.pi * j / REFINE_SAMPLES
            x = _clamp(best_x + REFINE_RADIUS_M * math.cos(theta), 0.5, pitch_length - 0.5)
            y = _clamp(best_y + REFINE_RADIUS_M * math.sin(theta), 0.5, pitch_width - 0.5)
            baseline = pitch_control_at(players, team, x, y)
            score = _score_at_destination(players, player.id, x, y, team)
            gain = score - baseline
            if gain > best_gain:
                best_x, best_y, best_gain = x, y, gain

    move = math.hypot(best_x - cx, best_y - cy)
    if best_gain < MIN_GAIN or move < MIN_MOVE_M:
        return None
    return best_x, best_y, best_gain


def suggested_positions(
    players: list[Any],
    ball: tuple[float, float],
    possession: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> list[dict]:
    """One entry per off-ball player worth moving. Empty list if nothing helps."""
    own = _team_players(players, possession)
    if len(own) < 2:
        return []

    bx, by = ball
    # Ball carrier = closest own player. Skip them: their positioning is the
    # ball's positioning, and asking them to move is asking to redefine "ball".
    carrier = min(own, key=lambda p: (p.x - bx) ** 2 + (p.y - by) ** 2)

    out: list[dict] = []
    for player in own:
        if player.id == carrier.id:
            continue
        result = _search_one(
            players, player, possession, pitch_length, pitch_width,
        )
        if result is None:
            continue
        x, y, gain = result
        out.append({
            "playerId": player.id,
            "playerNumber": int(getattr(player, "number", 0)),
            "team": player.team,
            "current": {"x": round(float(player.x), 2), "y": round(float(player.y), 2)},
            "suggested": {"x": round(x, 2), "y": round(y, 2)},
            "gain": round(gain, 3),
        })
    # Best-suggested-move first, so if the client ever caps the count it drops
    # the least-useful ghosts.
    out.sort(key=lambda item: item["gain"], reverse=True)
    return out

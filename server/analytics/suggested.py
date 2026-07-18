"""Suggested off-ball positions.

For every player on the possession team that isn't currently the ball carrier,
propose a nearby spot that raises the team's pitch control at the ball. Rendered
as ghost circles on the projected board so the coach can compare their own
intuition against the model.

This is NOT a trained model. It is a bounded gradient-ascent search over the
same _arrival_time / pitch_control_at heuristic the rest of the tactical
overlays already use. Two design choices worth calling out:

* **Objective = pitch control at ball, not xT.**  xT rewards being closer to
  goal, so it collapses every suggestion to "run forward" -- interesting
  once, boring always. Team control at the ball rewards support angles and
  passing outlets, which is what a coach actually reads a board for.

* **Movement budget is 5m.**  A player can only cover ~5m in the time a pass
  takes to arrive. Suggesting they teleport 30m away is a screenshot, not
  advice. The budget is per-player, along their current velocity (so a run
  in motion counts toward it) with a small angular search.

The search is a coarse polar sweep followed by a local refine -- ~64 objective
evaluations per player, ~700 per frame for a 22-player team. That is small
enough to run every state broadcast without caching. It is still gated behind
the overlay flag so it doesn't burn cycles when nothing is looking at it.
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
MIN_GAIN = 0.015            # ~1.5 percentage points of team pitch control
MIN_MOVE_M = 0.75           # too-tiny moves feel like a jitter, not a call


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


def _score(
    players: list[Any],
    swap_id: str,
    x: float,
    y: float,
    team: str,
    ball_x: float,
    ball_y: float,
) -> float:
    """Team pitch control at the ball if one player is moved to (x, y)."""
    swapped = [
        _mutated(p, x, y) if p.id == swap_id else p
        for p in players
    ]
    return pitch_control_at(swapped, team, ball_x, ball_y)


def _search_one(
    players: list[Any],
    player: Any,
    team: str,
    ball_x: float,
    ball_y: float,
    pitch_length: float,
    pitch_width: float,
    baseline: float,
) -> tuple[float, float, float] | None:
    """Return (x, y, gain) or None if the best candidate isn't worth drawing."""
    cx, cy = float(player.x), float(player.y)

    best_x, best_y, best_score = cx, cy, baseline
    for i in range(1, RADIAL_SAMPLES + 1):
        r = MOVE_BUDGET_M * i / RADIAL_SAMPLES
        for j in range(ANGULAR_SAMPLES):
            theta = 2.0 * math.pi * j / ANGULAR_SAMPLES
            x = _clamp(cx + r * math.cos(theta), 0.5, pitch_length - 0.5)
            y = _clamp(cy + r * math.sin(theta), 0.5, pitch_width - 0.5)
            s = _score(players, player.id, x, y, team, ball_x, ball_y)
            if s > best_score:
                best_x, best_y, best_score = x, y, s

    # Local refine around the coarse best (only if coarse actually improved).
    if best_score > baseline:
        for j in range(REFINE_SAMPLES):
            theta = 2.0 * math.pi * j / REFINE_SAMPLES
            x = _clamp(best_x + REFINE_RADIUS_M * math.cos(theta), 0.5, pitch_length - 0.5)
            y = _clamp(best_y + REFINE_RADIUS_M * math.sin(theta), 0.5, pitch_width - 0.5)
            s = _score(players, player.id, x, y, team, ball_x, ball_y)
            if s > best_score:
                best_x, best_y, best_score = x, y, s

    gain = best_score - baseline
    move = math.hypot(best_x - cx, best_y - cy)
    if gain < MIN_GAIN or move < MIN_MOVE_M:
        return None
    return best_x, best_y, gain


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
    baseline = pitch_control_at(players, possession, bx, by)

    out: list[dict] = []
    for player in own:
        if player.id == carrier.id:
            continue
        result = _search_one(
            players, player, possession, bx, by,
            pitch_length, pitch_width, baseline,
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

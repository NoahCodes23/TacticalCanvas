"""Expected Goals (xG) — per-shot scoring probability.

A logistic on distance to goal and shot angle, using the coefficients from
David Sumpter's Soccermatics tutorial (fit on Wyscout open-play shots). It's
the two-feature baseline every intro xG post reproduces — plenty for a demo
sidebar, and swappable for a trained model without touching callers.

Coordinates are raw pitch metres; ``direction`` matches
``experimental.attacking_direction`` for the shooting team so the goal-line
lookup is symmetric across halves.
"""

from __future__ import annotations

import math

_GOAL_WIDTH = 7.32  # metres, laws of the game
_HALF_GOAL = _GOAL_WIDTH / 2.0

# Sumpter open-play logistic on (angle_rad, distance_m).
_INTERCEPT = -3.19
_ANGLE_COEF = 1.88
_DIST_COEF = -0.10


def xg_value(
    x: float,
    y: float,
    direction: int,
    pitch_length: float,
    pitch_width: float,
) -> float:
    """P(goal) from a shot taken at (x, y). Returns 0..1."""
    goal_x = pitch_length if direction > 0 else 0.0
    goal_y = pitch_width / 2.0
    dx = abs(goal_x - x)
    dy = y - goal_y

    # Angle the goalmouth subtends from the shot location. The atan2 form
    # stays well-behaved at every position, including behind the goal-line.
    denom = dx * dx + dy * dy - _HALF_GOAL * _HALF_GOAL
    angle = math.atan2(_GOAL_WIDTH * dx, denom)
    if angle < 0:
        angle += math.pi

    distance = math.hypot(dx, dy)
    logit = _INTERCEPT + _ANGLE_COEF * angle + _DIST_COEF * distance
    return 1.0 / (1.0 + math.exp(-logit))

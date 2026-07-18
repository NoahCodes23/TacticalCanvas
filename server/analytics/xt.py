"""Expected Threat (xT) — grid-based possession value.

xT answers "how much is having the ball at (x, y) worth, in goals?" It's a
12x8 lookup published by Karun Singh in his 2019 blog post, computed by
iterating action-value from ~500k Premier League events. The grid below is a
close approximation of those published values — good enough that pass deltas
tell the same story (long balls into the box add much more xT than square
passes in midfield), while being defensible to drop in place of a trained
table later.

Coordinates: the grid is attacking left-to-right. Column 0 sits on the
attacking team's own goal-line, column 11 on the opponent's. Callers pass
raw pitch (x, y) plus the possessing team's attacking direction; we flip x
when the team is attacking right-to-left so the lookup is direction-agnostic.
"""

from __future__ import annotations

# 8 rows (y, top=0 -> bottom=7) x 12 cols (x, own goal=0 -> opp goal=11).
# Values are goals-added; the opponent-box column dominates.
_GRID: tuple[tuple[float, ...], ...] = (
    (0.006, 0.008, 0.009, 0.011, 0.014, 0.017, 0.021, 0.028, 0.036, 0.049, 0.070, 0.170),
    (0.007, 0.009, 0.011, 0.013, 0.017, 0.021, 0.026, 0.034, 0.045, 0.062, 0.093, 0.240),
    (0.008, 0.010, 0.013, 0.015, 0.019, 0.025, 0.031, 0.041, 0.055, 0.078, 0.121, 0.320),
    (0.008, 0.011, 0.014, 0.017, 0.021, 0.027, 0.035, 0.046, 0.062, 0.088, 0.140, 0.380),
    (0.008, 0.011, 0.014, 0.017, 0.021, 0.027, 0.035, 0.046, 0.062, 0.088, 0.140, 0.380),
    (0.008, 0.010, 0.013, 0.015, 0.019, 0.025, 0.031, 0.041, 0.055, 0.078, 0.121, 0.320),
    (0.007, 0.009, 0.011, 0.013, 0.017, 0.021, 0.026, 0.034, 0.045, 0.062, 0.093, 0.240),
    (0.006, 0.008, 0.009, 0.011, 0.014, 0.017, 0.021, 0.028, 0.036, 0.049, 0.070, 0.170),
)

_COLS = 12
_ROWS = 8


def xt_value(
    x: float,
    y: float,
    direction: int,
    pitch_length: float,
    pitch_width: float,
) -> float:
    """Look up xT at pitch coordinate (x, y).

    ``direction`` is +1 when the possessing team attacks toward x=pitch_length,
    -1 otherwise (matches ``experimental.attacking_direction``).
    """
    attack_x = x if direction > 0 else pitch_length - x
    col = int(attack_x / pitch_length * _COLS)
    row = int(y / pitch_width * _ROWS)
    col = min(max(col, 0), _COLS - 1)
    row = min(max(row, 0), _ROWS - 1)
    return _GRID[row][col]


def xt_delta(
    origin: tuple[float, float],
    destination: tuple[float, float],
    direction: int,
    pitch_length: float,
    pitch_width: float,
) -> float:
    """Value added by a move from origin to destination (can be negative)."""
    a = xt_value(origin[0], origin[1], direction, pitch_length, pitch_width)
    b = xt_value(destination[0], destination[1], direction, pitch_length, pitch_width)
    return b - a

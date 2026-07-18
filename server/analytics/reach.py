"""Defender reach shadows: where a player can physically get to within N seconds.

Pure kinematics -- no model, no training data. For each direction around the
player we integrate a double integrator with a reaction delay and a speed cap:

    reaction    the player keeps drifting at their current velocity
    accelerate  at A_MAX until they reach V_MAX in that direction
    cruise      at V_MAX for whatever time is left

Sampling that displacement around a ring of directions traces a teardrop: long
ahead of a player who is already running, pinched behind them, because turning
round costs them the momentum they have. A standing player gets a circle. The
gaps *between* the teardrops are the unmarked zones the overlay exists to show.

Only the along-direction component of velocity is carried through the reaction
phase; the sideways drift is a second-order correction that isn't worth the
complexity at a 2-second horizon.
"""

import math

# Tuned for an outfield footballer at match intensity. V_MAX is a sustained
# turn-and-run speed rather than a straight-line sprint peak (~9-10 m/s), and
# A_MAX is what a player actually holds while changing direction, not the
# instantaneous peak off a standing start.
V_MAX_MS = 8.0
A_MAX_MS2 = 5.0
REACTION_S = 0.2
DIRECTIONS = 28


def reach_distance(
    v_along: float,
    horizon: float,
    v_max: float = V_MAX_MS,
    a_max: float = A_MAX_MS2,
    reaction: float = REACTION_S,
) -> float:
    """How far the player travels along one direction in `horizon` seconds,
    given `v_along` -- the component of their current velocity along it
    (negative when they are moving the opposite way)."""
    if horizon <= 0.0:
        return 0.0

    t_react = min(reaction, horizon)
    d = v_along * t_react

    t_left = horizon - t_react
    s0 = max(-v_max, min(v_along, v_max))
    t_accel = min(max(0.0, (v_max - s0) / a_max), t_left)
    d += s0 * t_accel + 0.5 * a_max * t_accel * t_accel
    d += v_max * (t_left - t_accel)

    # A player sprinting hard one way has a "negative" reach directly behind
    # them for short horizons -- they are still travelling away from it.
    return max(0.0, d)


def reach_polygon(
    x: float,
    y: float,
    vx: float,
    vy: float,
    horizon: float,
    pitch_length: float,
    pitch_width: float,
    directions: int = DIRECTIONS,
) -> list[list[float]]:
    """The reachable region as a closed polygon in pitch metres.

    Vertices are clamped into the pitch rectangle rather than properly clipped
    against it. For a blob against an axis-aligned rect the two agree except
    within a metre or so of a corner, and this keeps the shape a fixed-length
    ring the renderer can draw without special cases.
    """
    pts: list[list[float]] = []
    for i in range(directions):
        theta = 2.0 * math.pi * i / directions
        ux, uy = math.cos(theta), math.sin(theta)
        r = reach_distance(vx * ux + vy * uy, horizon)
        px = min(max(x + r * ux, 0.0), pitch_length)
        py = min(max(y + r * uy, 0.0), pitch_width)
        pts.append([round(px, 1), round(py, 1)])
    return pts

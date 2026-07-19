"""Explainable, opt-in tactical indicators and next-action recommendations.

This module deliberately contains no learned-model or heavyweight dependency.
It is an experimental baseline that turns live tracking coordinates into useful,
inspectable features.  The output schema is suitable for replacing the scoring
function with a trained logistic-regression/XGBoost model later without changing
the renderer or WebSocket protocol.

The scorer combines:

* pass distance and forward progress;
* pressure on the passer and receiver;
* defenders screening the passing lane;
* arrival-time pitch control at the destination;
* a smooth, location-based expected-threat (xT) prior; and
* the cost of losing possession from the current field position.

It should be described as an explainable heuristic prototype, not as a validated
prediction model.  That distinction is included in every response sent to the
browser.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


MODEL_INFO = {
    "name": "Explainable tracking baseline",
    "version": "0.1-experimental",
    "kind": "heuristic",
    "trained": False,
    "disclaimer": "Experimental estimate, not a trained or validated model.",
}

TOP_RECOMMENDATIONS = 3
TARGET_HORIZON_S = 2.5


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    # Clamping prevents exp overflow when a malformed tracking frame arrives.
    value = _clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-value))


def _distance(a: Any, x: float, y: float) -> float:
    return math.hypot(float(a.x) - x, float(a.y) - y)


def _team_players(players: Iterable[Any], team: str) -> list[Any]:
    return [player for player in players if player.team == team]


def attacking_direction(players: list[Any], team: str) -> int:
    """Return +1 when ``team`` attacks toward x=length, otherwise -1.

    Tracking feeds do not always carry the half/direction explicitly.  Team
    centroids are a stable proxy: the side whose centroid is leftmost defends
    the left goal and attacks right.  The fallback keeps synthetic/partial
    frames deterministic.
    """
    own = _team_players(players, team)
    other = [player for player in players if player.team != team]
    if own and other:
        own_x = sum(float(player.x) for player in own) / len(own)
        other_x = sum(float(player.x) for player in other) / len(other)
        return 1 if own_x < other_x else -1
    return 1 if team == "home" else -1


def _attack_x(x: float, direction: int, pitch_length: float) -> float:
    return x if direction > 0 else pitch_length - x


def expected_threat(
    x: float,
    y: float,
    direction: int,
    pitch_length: float,
    pitch_width: float,
) -> float:
    """Smooth xT-like spatial prior in [0, 1].

    It rises non-linearly toward goal and rewards central destinations in the
    final third.  It is intentionally a prior, not a learned xT grid; exposing
    it through one function makes a data-trained table a drop-in replacement.
    """
    xn = _clip(_attack_x(x, direction, pitch_length) / pitch_length, 0.0, 1.0)
    yn = (y - pitch_width / 2.0) / (pitch_width * 0.32)
    centrality = math.exp(-(yn * yn))
    value = 0.008 + 0.105 * xn**1.7 + 0.62 * xn**4.5 * centrality
    return _clip(value, 0.0, 1.0)


def _arrival_time(player: Any, x: float, y: float) -> float:
    """Approximate time for a player to arrive, including current momentum."""
    dx, dy = x - float(player.x), y - float(player.y)
    distance = math.hypot(dx, dy)
    if distance < 0.15:
        return 0.0

    ux, uy = dx / distance, dy / distance
    toward = max(0.0, float(getattr(player, "vx", 0.0)) * ux + float(getattr(player, "vy", 0.0)) * uy)
    v_max, acceleration, reaction = 8.0, 5.0, 0.2
    toward = min(toward, v_max)
    remaining = max(0.0, distance - toward * reaction)
    time_to_cap = (v_max - toward) / acceleration
    distance_to_cap = toward * time_to_cap + 0.5 * acceleration * time_to_cap**2
    if remaining <= distance_to_cap:
        moving = (-toward + math.sqrt(toward**2 + 2.0 * acceleration * remaining)) / acceleration
    else:
        moving = time_to_cap + (remaining - distance_to_cap) / v_max
    return reaction + moving


def pitch_control_at(players: list[Any], team: str, x: float, y: float) -> float:
    """Arrival-time control probability for ``team`` at one pitch location."""
    own = [_arrival_time(player, x, y) for player in players if player.team == team]
    opposition = [_arrival_time(player, x, y) for player in players if player.team != team]
    if not own:
        return 0.0
    if not opposition:
        return 1.0
    advantage_seconds = min(opposition) - min(own)
    return _sigmoid(1.7 * advantage_seconds)


def _point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    """Return perpendicular distance and segment progress t in [0, 1]."""
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = _clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), t


def _nearest_opponent_distance(players: list[Any], team: str, x: float, y: float) -> float:
    distances = [_distance(player, x, y) for player in players if player.team != team]
    return min(distances, default=20.0)


def _lane_features(
    players: list[Any],
    team: str,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[int, float]:
    pass_distance = math.hypot(bx - ax, by - ay)
    corridor = 1.25 + 0.035 * pass_distance
    blockers = 0
    clearance = 20.0
    for defender in players:
        if defender.team == team:
            continue
        distance, t = _point_segment_distance(
            float(defender.x), float(defender.y), ax, ay, bx, by
        )
        if 0.08 < t < 0.96:
            clearance = min(clearance, distance)
            if distance < corridor:
                blockers += 1
    return blockers, clearance


def _defenders_bypassed(
    players: list[Any],
    team: str,
    origin_x: float,
    destination_x: float,
    destination_y: float,
    direction: int,
    pitch_length: float,
) -> int:
    start = _attack_x(origin_x, direction, pitch_length)
    end = _attack_x(destination_x, direction, pitch_length)
    if end <= start:
        return 0
    return sum(
        1
        for defender in players
        if defender.team != team
        and start < _attack_x(float(defender.x), direction, pitch_length) < end
        and abs(float(defender.y) - destination_y) <= 18.0
    )


def _offside(
    players: list[Any],
    team: str,
    receiver_x: float,
    ball_x: float,
    direction: int,
    pitch_length: float,
) -> bool:
    defenders = sorted(
        (
            _attack_x(float(player.x), direction, pitch_length)
            for player in players
            if player.team != team
        ),
        reverse=True,
    )
    if len(defenders) < 2:
        return False
    receiver_depth = _attack_x(receiver_x, direction, pitch_length)
    ball_depth = _attack_x(ball_x, direction, pitch_length)
    line = max(ball_depth, defenders[1])
    return receiver_depth > pitch_length / 2.0 and receiver_depth > line + 0.1


def _pass_explanation(features: dict[str, Any]) -> list[str]:
    positive: list[tuple[float, str]] = []
    negative: list[tuple[float, str]] = []
    if features["forwardProgressM"] >= 8.0:
        positive.append((features["forwardProgressM"] / 20.0, f"gains {features['forwardProgressM']:.0f} m"))
    if features["xTGain"] >= 0.015:
        positive.append((features["xTGain"] * 10.0, f"adds {features['xTGain']:.3f} xT"))
    if features["receiverSpaceM"] >= 5.0:
        positive.append((features["receiverSpaceM"] / 10.0, f"{features['receiverSpaceM']:.1f} m receiver space"))
    if features["laneDefenders"] == 0:
        positive.append((0.7, "clear passing lane"))
    if features["defendersBypassed"]:
        positive.append((0.5 + features["defendersBypassed"] * 0.1,
                         f"bypasses {features['defendersBypassed']} defender(s)"))
    if features["destinationControl"] >= 0.72:
        positive.append((features["destinationControl"], "strong destination control"))

    if features["laneDefenders"]:
        negative.append((features["laneDefenders"], f"{features['laneDefenders']} lane blocker(s)"))
    if features["receiverSpaceM"] < 3.0:
        negative.append((3.0 - features["receiverSpaceM"], "receiver tightly marked"))
    if features["offside"]:
        negative.append((10.0, "receiver is offside"))

    positive.sort(reverse=True)
    negative.sort(reverse=True)
    reasons = [text for _, text in positive[:2]]
    if negative:
        reasons.append(negative[0][1])
    return reasons or ["balanced retention option"]


def _score_pass(
    players: list[Any],
    carrier: Any,
    receiver: Any,
    ball: tuple[float, float],
    direction: int,
    pitch_length: float,
    pitch_width: float,
    receiver_position: tuple[float, float] | None = None,
    ignore_offside: bool = False,
) -> dict[str, Any]:
    team = carrier.team
    rx, ry = receiver_position or (float(receiver.x), float(receiver.y))
    cx, cy = float(carrier.x), float(carrier.y)
    distance = math.hypot(rx - cx, ry - cy)
    progress = direction * (rx - cx)
    receiver_space = _nearest_opponent_distance(players, team, rx, ry)
    passer_space = _nearest_opponent_distance(players, team, cx, cy)
    lane_defenders, lane_clearance = _lane_features(players, team, cx, cy, rx, ry)
    destination_control = pitch_control_at(players, team, rx, ry)
    origin_xt = expected_threat(cx, cy, direction, pitch_length, pitch_width)
    destination_xt = expected_threat(rx, ry, direction, pitch_length, pitch_width)
    xt_gain = destination_xt - origin_xt
    bypassed = _defenders_bypassed(
        players, team, cx, rx, ry, direction, pitch_length
    )
    is_offside = _offside(
        players, team, rx, ball[0], direction, pitch_length
    )

    # Coefficients are deliberately human-readable baseline weights.  They
    # approximate the monotonic relationships a completion model should learn.
    logit = (
        2.35
        - 0.052 * distance
        + 0.14 * (_clip(receiver_space, 0.0, 10.0) - 4.0)
        + 0.07 * (_clip(passer_space, 0.0, 10.0) - 3.0)
        - 0.92 * lane_defenders
        + 0.08 * (_clip(lane_clearance, 0.0, 8.0) - 3.0)
        + 1.25 * (destination_control - 0.5)
        - 0.012 * abs(ry - cy)
    )
    completion = (
        0.02 if is_offside and not ignore_offside
        else _clip(_sigmoid(logit), 0.03, 0.98)
    )

    turnover_cost = 0.035 + 0.32 * origin_xt + 0.018 * max(progress, 0.0) / pitch_length
    action_reward = (
        0.018
        + max(0.0, xt_gain)
        + 0.045 * max(progress, 0.0) / pitch_length
        + 0.007 * bypassed
    )
    expected_value = completion * action_reward - (1.0 - completion) * turnover_cost

    features: dict[str, Any] = {
        "distanceM": round(distance, 2),
        "forwardProgressM": round(progress, 2),
        "receiverSpaceM": round(receiver_space, 2),
        "passerSpaceM": round(passer_space, 2),
        "laneDefenders": lane_defenders,
        "laneClearanceM": round(lane_clearance, 2),
        "defendersBypassed": bypassed,
        "destinationControl": round(destination_control, 3),
        "originXT": round(origin_xt, 4),
        "destinationXT": round(destination_xt, 4),
        "xTGain": round(xt_gain, 4),
        "turnoverCost": round(turnover_cost, 4),
        "offside": is_offside,
    }
    return {
        "receiverId": receiver.id,
        "receiverNumber": receiver.number,
        "team": team,
        "from": {"x": round(cx, 2), "y": round(cy, 2)},
        "to": {"x": round(rx, 2), "y": round(ry, 2)},
        "completionProbability": round(completion, 3),
        "expectedValue": round(expected_value, 4),
        "score": round(expected_value * 100.0, 2),
        "risk": "high" if completion < 0.55 else "medium" if completion < 0.76 else "low",
        "features": features,
        "explanation": _pass_explanation(features),
    }


def pass_completion_probability(
    players: list[Any],
    carrier: Any,
    receiver: Any,
    ball: tuple[float, float],
    direction: int,
    pitch_length: float,
    pitch_width: float,
) -> float:
    """Completion probability for one pass, from the same scorer ``analyze`` uses.

    Public entry point so other modules (the simulation planner) can price a
    pass without re-deriving the whole analysis snapshot — one formula, one
    number, everywhere.

    The offside override is skipped here on purpose: the simulation's world has
    no offside rule (attackers legitimately drift beyond the line mid-move), so
    pricing a sim pass at the flat offside penalty would contradict the play
    the viewer just watched. The logistic itself is unchanged.
    """
    scored = _score_pass(
        players, carrier, receiver, ball, direction, pitch_length, pitch_width,
        ignore_offside=True,
    )
    return float(scored["completionProbability"])


def _convex_hull_area(points: list[tuple[float, float]]) -> float:
    """Monotonic-chain convex hull area, with no scipy dependency."""
    unique = sorted(set(points))
    if len(unique) < 3:
        return 0.0

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(
        sum(
            hull[i][0] * hull[(i + 1) % len(hull)][1]
            - hull[(i + 1) % len(hull)][0] * hull[i][1]
            for i in range(len(hull))
        )
    ) / 2.0


def _team_metrics(
    players: list[Any],
    team: str,
    ball: tuple[float, float],
    pitch_length: float,
    pitch_width: float,
) -> dict[str, Any]:
    own = _team_players(players, team)
    opponents = [player for player in players if player.team != team]
    if not own:
        return {}
    direction = attacking_direction(players, team)
    ordered = sorted(own, key=lambda player: _attack_x(float(player.x), direction, pitch_length))
    outfield = ordered[1:] if len(ordered) > 6 else ordered
    depths = [_attack_x(float(player.x), direction, pitch_length) for player in outfield]
    ys = [float(player.y) for player in outfield]
    speeds = [math.hypot(float(getattr(player, "vx", 0.0)), float(getattr(player, "vy", 0.0))) for player in own]
    nearest_opponent = [
        min((_distance(opponent, float(player.x), float(player.y)) for opponent in opponents), default=20.0)
        for player in own
    ]
    defender_depths = sorted(depths)[: min(4, len(depths))]
    attacking_depths = sorted(depths, reverse=True)[: min(3, len(depths))]
    xt_values = [
        expected_threat(float(player.x), float(player.y), direction, pitch_length, pitch_width)
        for player in outfield
    ]
    width = max(ys) - min(ys) if ys else 0.0
    depth = max(depths) - min(depths) if depths else 0.0
    area = _convex_hull_area([(float(player.x), float(player.y)) for player in outfield])
    spacing = sum(nearest_opponent) / len(nearest_opponent) if nearest_opponent else 0.0
    avg_speed = sum(speeds) / len(speeds)
    shape_score = 100.0 - 1.15 * abs(width - 44.0) - 0.8 * abs(depth - 42.0)
    return {
        "attackingDirection": "right" if direction > 0 else "left",
        "fieldTiltPct": round(100.0 * sum(depths) / max(1, len(depths)) / pitch_length, 1),
        "lineHeightM": round(sum(defender_depths) / max(1, len(defender_depths)), 1),
        "attackingLineM": round(sum(attacking_depths) / max(1, len(attacking_depths)), 1),
        "teamWidthM": round(width, 1),
        "teamDepthM": round(depth, 1),
        "occupiedAreaM2": round(area, 0),
        "avgSpeedMps": round(avg_speed, 2),
        "highIntensityRuns": sum(speed >= 3.5 for speed in speeds),
        "sprints": sum(speed >= 5.5 for speed in speeds),
        "ballPressureM": round(min((_distance(player, ball[0], ball[1]) for player in own), default=0.0), 1),
        "avgOpponentSpaceM": round(spacing, 1),
        "avgXT": round(sum(xt_values) / max(1, len(xt_values)), 4),
        "shapeScore": round(_clip(shape_score, 0.0, 100.0), 0),
    }


def _receiver_targets(
    players: list[Any],
    carrier: Any,
    passes: list[dict[str, Any]],
    ball: tuple[float, float],
    direction: int,
    pitch_length: float,
    pitch_width: float,
    target_limit: int | None = TOP_RECOMMENDATIONS,
    include_holds: bool = False,
) -> list[dict[str, Any]]:
    by_id = {player.id: player for player in players}
    targets: list[dict[str, Any]] = []
    offsets = [(0.0, 0.0)]
    for radius in (4.0, 8.0):
        for i in range(8):
            angle = 2.0 * math.pi * i / 8.0
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))

    candidates = passes if target_limit is None else passes[:target_limit]
    for original in candidates:
        receiver = by_id.get(original["receiverId"])
        if receiver is None:
            continue
        best = original
        best_adjusted = float(original["expectedValue"])
        best_move = 0.0
        for dx, dy in offsets[1:]:
            x, y = float(receiver.x) + dx, float(receiver.y) + dy
            if not (0.5 <= x <= pitch_length - 0.5 and 0.5 <= y <= pitch_width - 0.5):
                continue
            # Avoid recommendations that collapse two teammates onto the same
            # point.  Opponent proximity remains allowed and is scored as risk.
            if any(
                teammate.team == carrier.team
                and teammate.id not in (receiver.id, carrier.id)
                and _distance(teammate, x, y) < 2.5
                for teammate in players
            ):
                continue
            candidate = _score_pass(
                players, carrier, receiver, ball, direction, pitch_length, pitch_width, (x, y)
            )
            if candidate["features"]["offside"]:
                continue
            move = math.hypot(dx, dy)
            adjusted = float(candidate["expectedValue"]) - 0.0012 * move
            if adjusted > best_adjusted:
                best, best_adjusted, best_move = candidate, adjusted, move

        improvement = float(best["expectedValue"]) - float(original["expectedValue"])
        if best_move > 0.5 and improvement > 0.0015:
            targets.append({
                "playerId": receiver.id,
                "playerNumber": receiver.number,
                "from": {"x": round(float(receiver.x), 2), "y": round(float(receiver.y), 2)},
                "to": best["to"],
                "moveDistanceM": round(best_move, 1),
                "reachableInS": round(best_move / 5.0, 1),
                "currentScore": original["score"],
                "targetScore": best["score"],
                "improvement": round(improvement * 100.0, 2),
                "reason": best["explanation"][0],
            })
        elif include_holds:
            targets.append({
                "playerId": receiver.id,
                "playerNumber": receiver.number,
                "from": {"x": round(float(receiver.x), 2), "y": round(float(receiver.y), 2)},
                "to": {"x": round(float(receiver.x), 2), "y": round(float(receiver.y), 2)},
                "moveDistanceM": 0.0,
                "reachableInS": 0.0,
                "currentScore": original["score"],
                "targetScore": original["score"],
                "improvement": 0.0,
                "reason": "hold position; no nearby move improves the current option",
            })
    return targets


def analyze(
    players: list[Any],
    ball: tuple[float, float],
    possession: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    include_receiver_targets: bool = False,
    receiver_target_limit: int | None = TOP_RECOMMENDATIONS,
    include_hold_targets: bool = False,
) -> dict[str, Any]:
    """Compute one explainable tactical-analysis snapshot."""
    if not players:
        return {"model": MODEL_INFO, "context": {}, "passes": [], "receiverTargets": [], "teams": {}}

    possessing = _team_players(players, possession)
    if not possessing:
        return {"model": MODEL_INFO, "context": {}, "passes": [], "receiverTargets": [], "teams": {}}
    carrier = min(possessing, key=lambda player: _distance(player, ball[0], ball[1]))
    direction = attacking_direction(players, possession)
    passes = [
        _score_pass(players, carrier, receiver, ball, direction, pitch_length, pitch_width)
        for receiver in possessing
        if receiver.id != carrier.id
    ]
    passes.sort(key=lambda item: item["expectedValue"], reverse=True)
    for rank, candidate in enumerate(passes, 1):
        candidate["rank"] = rank
        candidate["recommended"] = rank <= TOP_RECOMMENDATIONS

    ball_depth = _attack_x(ball[0], direction, pitch_length)
    phase = "build-up" if ball_depth < pitch_length / 3.0 else "progression" if ball_depth < 2.0 * pitch_length / 3.0 else "final third"
    channel = "left" if ball[1] < pitch_width / 3.0 else "right" if ball[1] > 2.0 * pitch_width / 3.0 else "central"
    opponents = [player for player in players if player.team != possession]
    carrier_pressure = min((_distance(player, float(carrier.x), float(carrier.y)) for player in opponents), default=20.0)
    speeds = [math.hypot(float(getattr(player, "vx", 0.0)), float(getattr(player, "vy", 0.0))) for player in players]
    viable = [candidate for candidate in passes if candidate["completionProbability"] >= 0.55 and not candidate["features"]["offside"]]
    progressive = [candidate for candidate in viable if candidate["features"]["forwardProgressM"] >= 8.0]

    lanes = []
    for name, low, high in (
        ("left", 0.0, pitch_width / 3.0),
        ("central", pitch_width / 3.0, 2.0 * pitch_width / 3.0),
        ("right", 2.0 * pitch_width / 3.0, pitch_width),
    ):
        own_count = sum(low <= float(player.y) < high for player in possessing)
        opp_count = sum(low <= float(player.y) < high for player in opponents)
        lanes.append({"channel": name, "attackers": own_count, "defenders": opp_count, "overload": own_count - opp_count})

    targets = _receiver_targets(
        players,
        carrier,
        passes,
        ball,
        direction,
        pitch_length,
        pitch_width,
        receiver_target_limit,
        include_hold_targets,
    ) if include_receiver_targets else []

    return {
        "model": dict(MODEL_INFO),
        "context": {
            "possession": possession,
            "ballCarrierId": carrier.id,
            "ballCarrierNumber": carrier.number,
            "attackingDirection": "right" if direction > 0 else "left",
            "phase": phase,
            "channel": channel,
            "pitchControlAtBall": round(pitch_control_at(players, possession, ball[0], ball[1]), 3),
            "carrierPressureM": round(carrier_pressure, 2),
            "pressureIndex": round(100.0 * math.exp(-carrier_pressure / 4.5), 0),
            "tempoMps": round(sum(speeds) / max(1, len(speeds)), 2),
            "transitionIndex": round(_clip((sum(speeds) / max(1, len(speeds)) - 0.8) * 30.0, 0.0, 100.0), 0),
            "viablePasses": len(viable),
            "progressivePasses": len(progressive),
            "bestChannel": max(lanes, key=lambda lane: lane["overload"])["channel"],
            "channelOverloads": lanes,
        },
        "passes": passes,
        "receiverTargets": targets,
        "teams": {
            "home": _team_metrics(players, "home", ball, pitch_length, pitch_width),
            "away": _team_metrics(players, "away", ball, pitch_length, pitch_width),
        },
    }

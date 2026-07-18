import pytest

from server.analytics.reach import (
    A_MAX_MS2,
    REACTION_S,
    V_MAX_MS,
    reach_distance,
    reach_polygon,
)


def test_standing_player_reaches_equally_in_all_directions():
    d = [reach_distance(0.0, 2.0) for _ in range(4)]
    assert len(set(d)) == 1
    # No reaction drift; accelerate to V_MAX, then cruise for the remainder.
    moving_time = 2.0 - REACTION_S
    accel_time = min(moving_time, V_MAX_MS / A_MAX_MS2)
    expected = (
        0.5 * A_MAX_MS2 * accel_time**2
        + V_MAX_MS * (moving_time - accel_time)
    )
    assert d[0] == pytest.approx(expected)


def test_running_player_reaches_further_ahead_than_behind():
    ahead = reach_distance(6.0, 2.0)
    behind = reach_distance(-6.0, 2.0)
    assert ahead > behind
    # Momentum is genuinely expensive to reverse, not a cosmetic asymmetry.
    assert ahead > 2 * behind


def test_reach_never_negative_behind_a_sprinter():
    assert reach_distance(-V_MAX_MS, 0.3) == 0.0


def test_speed_cap_binds_over_a_long_horizon():
    # Already at top speed: the whole horizon is spent cruising.
    assert reach_distance(V_MAX_MS, 3.0) == pytest.approx(V_MAX_MS * 3.0)
    # And nobody outruns v_max over the horizon regardless of direction.
    assert reach_distance(0.0, 5.0) < V_MAX_MS * 5.0


def test_zero_horizon_reaches_nowhere():
    assert reach_distance(5.0, 0.0) == 0.0


def test_polygon_is_a_closed_ring_inside_the_pitch():
    pts = reach_polygon(52.5, 34.0, 3.0, 0.0, 2.0, 105.0, 68.0)
    assert len(pts) == 28
    for x, y in pts:
        assert 0.0 <= x <= 105.0
        assert 0.0 <= y <= 68.0


def test_polygon_is_clamped_into_the_pitch_at_the_touchline():
    # A player on the touchline sprinting at it: nothing may leak off the pitch.
    pts = reach_polygon(2.0, 1.0, 0.0, -8.0, 3.0, 105.0, 68.0)
    assert min(y for _, y in pts) == 0.0
    assert min(x for x, _ in pts) >= 0.0


def test_polygon_leans_towards_the_direction_of_travel():
    x0 = 52.5
    pts = reach_polygon(x0, 34.0, 7.0, 0.0, 2.0, 105.0, 68.0)
    forward = max(x for x, _ in pts) - x0
    backward = x0 - min(x for x, _ in pts)
    assert forward > backward

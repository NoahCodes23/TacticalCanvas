import math
import unittest
from types import SimpleNamespace

from server.analytics.experimental import pass_completion_probability
from server.analytics.xg import xg_value
from server.simulation import SimulationEngine


def player(pid, team, number, x, y):
    return SimpleNamespace(id=pid, team=team, number=number, x=x, y=y)


def build_players():
    """A clean left-to-right home chain: 2 -> 7 -> 8 -> 9 -> shot.

    Every leg is 8-30 m and forward; the away side sits deep and wide of the
    lanes so planning is stable and the tests aren't hostage to defence luck.
    """
    return [
        player("H2", "home", 2, 20, 34),
        player("H7", "home", 7, 40, 24),
        player("H8", "home", 8, 60, 34),
        player("H9", "home", 9, 85, 34),
        player("A1", "away", 1, 98, 8),
        player("A2", "away", 2, 99, 20),
        player("A3", "away", 3, 100, 34),
        player("A4", "away", 4, 101, 48),
        player("A5", "away", 5, 102, 60),
    ]


def start_engine():
    engine = SimulationEngine()
    ok = engine.build_and_start(build_players(), (19.5, 34.0), "home", 105.0, 68.0)
    assert ok, "engine should plan a move from this shape"
    return engine


def run_to_done(engine, max_seconds=60.0):
    dt = 1.0 / 60.0
    for _ in range(int(max_seconds / dt)):
        engine.tick(dt)
        if engine.phase == "done":
            return
    raise AssertionError("simulation never finished")


class SimulationProbabilityTests(unittest.TestCase):
    def test_every_step_carries_success_and_sequence_probability(self):
        engine = start_engine()
        self.assertGreaterEqual(len(engine.steps), 2)
        running = 1.0
        for step in engine.steps:
            p = step["successProbability"]
            self.assertGreater(p, 0.0)
            self.assertLessEqual(p, 1.0)
            running *= p
            self.assertAlmostEqual(step["sequenceProbability"], running, places=3)

    def test_sequence_probability_is_nonincreasing_and_on_snapshot(self):
        engine = start_engine()
        seq = [s["sequenceProbability"] for s in engine.steps]
        self.assertEqual(seq, sorted(seq, reverse=True))
        snap = engine.snapshot()
        self.assertEqual(snap["sequenceProbability"], seq[-1])

    def test_pass_probability_comes_from_the_shared_scorer(self):
        engine = start_engine()
        first = engine.steps[0]
        self.assertEqual(first["type"], "pass")
        frm = engine._by_number(first["fromNumber"])
        to = engine._by_number(first["toNumber"])
        expected = round(
            pass_completion_probability(
                engine.players, frm, to, (frm.x, frm.y), 1, 105.0, 68.0
            ),
            3,
        )
        self.assertEqual(first["successProbability"], expected)

    def test_shot_probability_is_xg_at_the_shooter(self):
        engine = start_engine()
        shot = next(s for s in engine.steps if s["type"] == "shot")
        shooter = engine._by_number(shot["fromNumber"])
        self.assertEqual(
            shot["successProbability"],
            round(xg_value(shooter.x, shooter.y, 1, 105.0, 68.0), 3),
        )


class SimulationSeekTests(unittest.TestCase):
    def test_trajectory_is_recorded_and_cleared_on_stop(self):
        engine = start_engine()
        run_to_done(engine)
        self.assertGreater(len(engine.trajectory), 10)
        frame = engine.trajectory[0]
        self.assertEqual(set(frame), {"t", "step", "ball", "players"})
        self.assertEqual(len(frame["players"]), 9)
        engine.stop()
        self.assertEqual(engine.trajectory, [])
        self.assertEqual(engine._checkpoints, {})

    def test_seek_first_step_restores_the_kickoff_shape(self):
        engine = start_engine()
        run_to_done(engine)
        recorded = len(engine.trajectory)
        self.assertIsNone(engine.seek_step(0))
        self.assertFalse(engine.playing)
        self.assertEqual(engine.step_index, 0)
        self.assertIsNone(engine.outcome)
        self.assertEqual(engine.phase, "settle")
        self.assertEqual(engine.steps[0]["status"], "active")
        self.assertTrue(all(s["status"] == "pending" for s in engine.steps[1:]))
        carrier = engine._by_number(2)
        self.assertAlmostEqual(carrier.x, 20.0)
        self.assertAlmostEqual(carrier.y, 34.0)
        # Scrubbing alone must never destroy the recording.
        self.assertEqual(len(engine.trajectory), recorded)

    def test_seek_last_step_and_out_of_range(self):
        engine = start_engine()
        run_to_done(engine)
        last = len(engine.steps) - 1
        self.assertIsNone(engine.seek_step(last))
        self.assertEqual(engine.step_index, last)
        self.assertFalse(engine.playing)
        self.assertIsInstance(engine.seek_step(len(engine.steps)), str)
        self.assertIsInstance(engine.seek_step(-1), str)

    def test_seeking_an_unplayed_step_is_an_error(self):
        engine = start_engine()
        # Nothing has ticked: only step 0 has a checkpoint.
        self.assertGreaterEqual(len(engine.steps), 2)
        err = engine.seek_step(1)
        self.assertIsInstance(err, str)
        self.assertIn("not played", err)

    def test_seek_on_inactive_engine_is_an_error(self):
        engine = SimulationEngine()
        self.assertIsInstance(engine.seek_step(0), str)

    def test_forward_seek_spans_recording_beyond_restored_forecast(self):
        # A lane screened at ~3.0 m: too tight for the planner (>= 3.2) so the
        # plan is a lone hopeful shot, but fine for execution (>= 2.4) so the
        # move extends itself with new steps while it plays. Seeking back then
        # restores the short one-step forecast — steps the move actually
        # played beyond it must remain reachable through their checkpoints.
        engine = SimulationEngine()
        ok = engine.build_and_start(
            [
                player("H2", "home", 2, 30, 34),
                player("H7", "home", 7, 48, 34),
                player("A9", "away", 9, 39, 31),
                player("A1", "away", 1, 100, 30),
                player("A2", "away", 2, 101, 38),
            ],
            (29.5, 34.0), "home", 105.0, 68.0,
        )
        self.assertTrue(ok)
        self.assertEqual(len(engine.steps), 1)
        run_to_done(engine)
        last = max(engine._checkpoints)
        self.assertGreater(last, 0)
        self.assertIsNone(engine.seek_step(0))
        self.assertEqual(len(engine.steps), 1)   # restored forecast is short
        self.assertGreater(engine.snapshot()["maxReachedStep"], 0)
        self.assertIsNone(engine.seek_step(last))
        self.assertEqual(engine.step_index, last)
        self.assertIsInstance(engine.seek_step(last + 1), str)

    def test_resume_after_seek_replays_deterministically(self):
        engine = start_engine()
        run_to_done(engine)
        outcome = engine.outcome
        labels = [s["label"] for s in engine.steps]
        frames = len(engine.trajectory)
        self.assertIsNone(engine.seek_step(0))
        engine.resume()
        run_to_done(engine)
        self.assertEqual(engine.outcome, outcome)
        self.assertEqual([s["label"] for s in engine.steps], labels)
        self.assertEqual(len(engine.trajectory), frames)


if __name__ == "__main__":
    unittest.main()

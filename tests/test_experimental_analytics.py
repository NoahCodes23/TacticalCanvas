import unittest
from types import SimpleNamespace

from server.analytics.experimental import analyze, expected_threat


def player(pid, team, number, x, y, vx=0.0, vy=0.0):
    return SimpleNamespace(
        id=pid, team=team, number=number, x=x, y=y, vx=vx, vy=vy
    )


class ExperimentalAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.players = [
            player("H1", "home", 1, 5, 34),
            player("H4", "home", 4, 30, 34),
            player("H7", "home", 7, 48, 12),
            player("H8", "home", 8, 45, 45),
            player("H9", "home", 9, 69, 34),
            player("A1", "away", 1, 100, 34),
            player("A2", "away", 2, 78, 10),
            player("A4", "away", 4, 55, 34),
            player("A5", "away", 5, 76, 45),
            player("A6", "away", 6, 82, 58),
        ]

    def test_xthreat_increases_toward_goal_and_flips_with_direction(self):
        left = expected_threat(20, 34, 1, 105, 68)
        right = expected_threat(90, 34, 1, 105, 68)
        self.assertGreater(right, left)
        self.assertAlmostEqual(right, expected_threat(15, 34, -1, 105, 68))

    def test_analysis_ranks_every_teammate_and_bounds_probabilities(self):
        result = analyze(self.players, (30, 34), "home")
        self.assertEqual(result["context"]["ballCarrierId"], "H4")
        self.assertEqual(len(result["passes"]), 4)
        self.assertEqual([p["rank"] for p in result["passes"]], [1, 2, 3, 4])
        self.assertTrue(all(0.0 <= p["completionProbability"] <= 1.0 for p in result["passes"]))
        self.assertEqual(sum(p["recommended"] for p in result["passes"]), 3)

    def test_blocked_central_lane_is_explicitly_reported(self):
        result = analyze(self.players, (30, 34), "home")
        central = next(p for p in result["passes"] if p["receiverId"] == "H9")
        wide = next(p for p in result["passes"] if p["receiverId"] == "H7")
        self.assertGreaterEqual(central["features"]["laneDefenders"], 1)
        self.assertLess(central["completionProbability"], wide["completionProbability"])

    def test_technical_indicators_cover_both_teams(self):
        result = analyze(self.players, (30, 34), "home")
        self.assertEqual(set(result["teams"]), {"home", "away"})
        for metrics in result["teams"].values():
            self.assertIn("fieldTiltPct", metrics)
            self.assertIn("lineHeightM", metrics)
            self.assertIn("occupiedAreaM2", metrics)
            self.assertIn("sprints", metrics)
            self.assertIn("avgXT", metrics)

    def test_receiver_targets_are_optional_and_inside_pitch(self):
        without = analyze(self.players, (30, 34), "home", include_receiver_targets=False)
        with_targets = analyze(self.players, (30, 34), "home", include_receiver_targets=True)
        self.assertEqual(without["receiverTargets"], [])
        for target in with_targets["receiverTargets"]:
            self.assertGreaterEqual(target["to"]["x"], 0.0)
            self.assertLessEqual(target["to"]["x"], 105.0)
            self.assertGreaterEqual(target["to"]["y"], 0.0)
            self.assertLessEqual(target["to"]["y"], 68.0)
            self.assertGreater(target["improvement"], 0.0)


if __name__ == "__main__":
    unittest.main()

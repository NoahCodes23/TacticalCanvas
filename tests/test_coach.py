import json
import unittest

from server import coaches
from server.coach import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    build_messages,
    build_request_body,
    compose_system_prompt,
)
from server.state import (
    AppState,
    COACH_RAW_HISTORY_FRAMES,
    COACH_SNAPSHOT_COUNT,
    COACH_SNAPSHOT_SPACING_FRAMES,
)


class CoachAdviceTests(unittest.TestCase):
    def test_prompt_contains_ordered_frame_window_and_safety_language(self):
        frames = [{"frameIndex": 10}, {"frameIndex": 11}]
        messages = build_messages(frames, "Test Match")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("experimental heuristic", SYSTEM_PROMPT)
        self.assertIn("five meaningfully spaced snapshots", SYSTEM_PROMPT)
        self.assertIn("4 to 6 short sentences", SYSTEM_PROMPT)
        self.assertIn("under 100 words", SYSTEM_PROMPT)
        payload = json.loads(messages[1]["content"].split("\n\n", 1)[1])
        self.assertEqual(payload["window"]["order"], "oldest_to_newest")
        self.assertEqual(payload["window"]["frameCount"], 2)
        self.assertEqual(payload["frames"], frames)
        self.assertEqual(payload["window"]["snapshotSpacingMs"], 400)

    def test_default_model_is_openai_gpt_4_1_mini(self):
        self.assertEqual(DEFAULT_MODEL, "gpt-4.1-mini")

    def test_state_returns_five_evenly_spaced_snapshots(self):
        state = AppState()
        for frame in range(COACH_RAW_HISTORY_FRAMES + 25):
            state.frame_index = frame
            state.media_time_ms = frame * 40.0
            state._record_coach_frame()
            state._record_coach_frame()

        window = state.coach_frame_inputs()
        self.assertEqual(len(window), COACH_SNAPSHOT_COUNT)
        self.assertEqual(window[-1]["frameIndex"], COACH_RAW_HISTORY_FRAMES + 24)
        self.assertEqual(
            [b["frameIndex"] - a["frameIndex"] for a, b in zip(window, window[1:])],
            [COACH_SNAPSHOT_SPACING_FRAMES] * (COACH_SNAPSHOT_COUNT - 1),
        )
        self.assertEqual(len({frame["frameIndex"] for frame in window}), len(window))

    def test_seek_backfills_five_spaced_tracking_snapshots(self):
        state = AppState()
        state.set_playback_time(1000 * 40.0, False)
        window = state.coach_frame_inputs()

        self.assertEqual(len(window), COACH_SNAPSHOT_COUNT)
        self.assertEqual(
            [frame["frameIndex"] for frame in window],
            [960, 970, 980, 990, 1000],
        )

    def test_recent_events_are_included_as_short_context(self):
        event = {"label": "Pass", "clock": "12:03", "detail": "#8"}
        messages = build_messages([{"frameIndex": 1}], "Test Match", [event])
        payload = json.loads(messages[1]["content"].split("\n\n", 1)[1])
        self.assertEqual(payload["recentEvents"], [event])

    def test_openai_request_has_no_provider_flag(self):
        # OpenAI (the default) rejects OpenRouter's provider field, so it must
        # be absent for the default base URL.
        body = build_request_body([], "Test Match", DEFAULT_MODEL)
        self.assertNotIn("provider", body)
        self.assertEqual(body["max_tokens"], 250)

    def test_openrouter_base_url_still_enforces_zdr(self):
        body = build_request_body(
            [], "Test Match", DEFAULT_MODEL, base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(body["provider"], {"zdr": True})


class CoachPersonaTests(unittest.TestCase):
    def test_default_prompt_is_unchanged_without_a_persona(self):
        self.assertEqual(compose_system_prompt(), SYSTEM_PROMPT)

    def test_persona_prompt_appends_style_but_keeps_safety_rules(self):
        coach = coaches.get_coach("aggressive")
        prompt = compose_system_prompt(coach.style_prompt, coach.name)
        # Base prompt (with its facts-whitelist rule) is preserved verbatim...
        self.assertTrue(prompt.startswith(SYSTEM_PROMPT))
        self.assertIn(coach.name, prompt)
        self.assertIn("Hunt the ball high", prompt)
        # ...and the numbers guardrail is re-asserted on top of the persona.
        self.assertIn('"facts"', prompt)

    def test_persona_flows_through_build_messages(self):
        coach = coaches.get_coach("defensive")
        messages = build_messages(
            [{"frameIndex": 1}], "Test Match",
            style_prompt=coach.style_prompt, persona_name=coach.name,
        )
        self.assertIn(coach.name, messages[0]["content"])
        self.assertIn("defence-first", messages[0]["content"])

    def test_public_coach_list_hides_style_prompt(self):
        public = coaches.list_coaches()
        self.assertEqual(public[0]["id"], coaches.DEFAULT_COACH_ID)
        for entry in public:
            self.assertNotIn("style_prompt", entry)
            self.assertEqual(set(entry), {"id", "name", "emoji", "tagline", "accent"})

    def test_state_tracks_and_validates_the_active_coach(self):
        state = AppState()
        self.assertEqual(state.coach_id, coaches.DEFAULT_COACH_ID)
        snap = state.snapshot()
        self.assertEqual(snap["activeCoach"], coaches.DEFAULT_COACH_ID)
        self.assertEqual(snap["availableCoaches"], coaches.list_coaches())

        before = state.revision
        self.assertTrue(state.set_coach("counter"))
        self.assertEqual(state.coach_id, "counter")
        self.assertGreater(state.revision, before)   # invalidates cached advice

        self.assertFalse(state.set_coach("nonexistent"))
        self.assertEqual(state.coach_id, "counter")


if __name__ == "__main__":
    unittest.main()

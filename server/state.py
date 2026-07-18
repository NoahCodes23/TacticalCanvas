import math
import time
from collections import deque
from types import SimpleNamespace

from . import match_data
from .analytics.experimental import analyze as analyze_experimental
from .analytics.formation import detect_formation
from .analytics.reach import reach_polygon
from .analytics.suggested import suggested_positions
from .match_data import PITCH_LENGTH, PITCH_WIDTH, Player

GRAB_RADIUS_M = 3.0

# A challenger must get this much closer to the ball than the team currently
# credited with it before possession flips, so a 50/50 doesn't strobe the
# shadow overlay between the two teams every other frame.
POSSESSION_MARGIN_M = 1.5

SHADOW_SECONDS_MIN = 0.5
SHADOW_SECONDS_MAX = 4.0

# When the dashboard has a video attached it drives the clock via
# SET_PLAYBACK_TIME on every displayed frame. If we've heard one within this
# window we treat the video as the master and stop auto-advancing in tick();
# after the window we assume the video is gone and take back the clock.
EXTERNAL_CLOCK_STALE_MS = 500.0

EXPERIMENT_DEFAULTS = {
    "passRecommendations": False,
    "technicalIndicators": False,
    "receiverTargets": False,
}

COACH_RAW_HISTORY_FRAMES = 20
COACH_SNAPSHOT_COUNT = 5
COACH_SNAPSHOT_SPACING_FRAMES = 10  # 400 ms at the 25 Hz tracking rate


def now_ms() -> float:
    return time.time() * 1000.0


class AppState:
    def __init__(self) -> None:
        self.scenario_id = "demo"
        self.revision = 0
        self.sequence = 0

        self.playing = True
        self.playback_rate = 1.0
        self.frame_index = 0
        self.media_time_ms = 0.0
        # Wall-clock ms of the last SET_PLAYBACK_TIME. 0 means never; the
        # server owns the clock until a video says otherwise.
        self.external_clock_last_ms = 0.0

        self.edit_mode = False
        self.calibration_overlay = False
        self.offside_overlay = False
        self.compactness_overlay = False
        self.shadow_overlay = False
        self.pitch_control_overlay = False
        self.formation_overlay = False
        self.suggested_overlay = False
        self._suggested_cache_key = None
        self._suggested_cache = []
        self.experiments = dict(EXPERIMENT_DEFAULTS)
        # Sized by the shape it draws rather than the number: 2s puts a standing
        # player's reach at ~8m, half of what the old 3s default drew. Reach is
        # not linear in the horizon -- the acceleration phase dominates early --
        # so halving the radius means 3.0 -> 2.0, not 3.0 -> 1.5.
        self.shadow_seconds = 2.0
        self.possession = "home"

        self.players: list[Player] = match_data.build_players()
        self.ball = (PITCH_LENGTH / 2, PITCH_WIDTH / 2)
        self.match_id = match_data.current_id()
        self.match_label = match_data.current_label()

        self.grabbed: dict[str, str] = {}
        self.cursors: dict[str, dict] = {}
        self._experimental_cache_key: tuple | None = None
        self._experimental_cache: dict | None = None
        # Keep a cheap, raw tracking history at the native 25 Hz frame rate.
        # The expensive tactical indicators are calculated only when the coach
        # explicitly asks for LLM advice.
        self._coach_history: deque[dict] = deque(maxlen=COACH_RAW_HISTORY_FRAMES)

        self.vision_stats = {
            "fps": 0.0,
            "captureFps": 0.0,
            "hands": 0,
            "calibrated": False,
            "handPx": 0,
            "inferenceMs": 0.0,
            "captureToInferenceMs": 0.0,
            "captureToServerMs": 0.0,
            "captureDrops": 0,
        }
        self._record_coach_frame()

    def _bump(self) -> None:
        self.revision += 1

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def player_by_id(self, player_id: str) -> Player | None:
        for p in self.players:
            if p.id == player_id:
                return p
        return None

    @staticmethod
    def board_to_pitch(bx: float, by: float) -> tuple[float, float]:
        return bx * PITCH_LENGTH, by * PITCH_WIDTH

    def nearest_player(self, x_m: float, y_m: float) -> Player | None:
        """Nearest player to a pitch-metre point, within GRAB_RADIUS_M."""
        best, best_d2 = None, GRAB_RADIUS_M**2
        for p in self.players:
            if p.id in self.grabbed.values():
                continue  
            d2 = (p.x - x_m) ** 2 + (p.y - y_m) ** 2
            if d2 < best_d2:
                best, best_d2 = p, d2
        return best

    def tick(self, dt: float) -> None:
        video_driven = (
            self.external_clock_last_ms > 0
            and now_ms() - self.external_clock_last_ms < EXTERNAL_CLOCK_STALE_MS
        )
        if not self.edit_mode:
            if video_driven:
                # Time already came in on the last SET_PLAYBACK_TIME; just
                # refresh the world at that pinned instant so pitch control,
                # shadows etc. reflect the video's *current* frame.
                t = self.media_time_ms / 1000.0
                match_data.advance(self.players, t)
                self.ball = match_data.ball_position(t)
            elif self.playing:
                self.media_time_ms += dt * 1000.0 * self.playback_rate
                self.frame_index = int(self.media_time_ms / 40.0)  # 25Hz tracking data
                t = self.media_time_ms / 1000.0
                match_data.advance(self.players, t)
                self.ball = match_data.ball_position(t)
        # Runs while paused too, so dragging a player in edit mode can hand
        # possession over and flip which team the shadows are drawn for.
        self._update_possession()
        self._record_coach_frame()

    def _update_possession(self) -> None:
        """Whoever is nearest the ball has it, with hysteresis (see
        POSSESSION_MARGIN_M). Metrica's tracking has no possession column, so
        this is inferred rather than read."""
        bx, by = self.ball
        nearest = {"home": math.inf, "away": math.inf}
        for p in self.players:
            d = math.hypot(p.x - bx, p.y - by)
            if d < nearest.get(p.team, math.inf):
                nearest[p.team] = d

        holder = self.possession
        rival = "away" if holder == "home" else "home"
        if math.isinf(nearest[rival]):
            return
        if math.isinf(nearest[holder]) or nearest[rival] + POSSESSION_MARGIN_M < nearest[holder]:
            self.possession = rival
            self._bump()

    def defending_team(self) -> str:
        return "away" if self.possession == "home" else "home"

    def enter_edit_mode(self) -> None:
        if not self.edit_mode:
            self.edit_mode = True
            self.playing = False
            self._bump()

    def exit_edit_mode(self) -> None:
        if self.edit_mode:
            self.edit_mode = False
            self.grabbed.clear()
            self._bump()

    def set_playing(self, playing: bool) -> None:
        self.playing = playing
        if playing:
            self.edit_mode = False
            self.grabbed.clear()
        self._bump()

    def set_playback_time(self, media_time_ms: float, playing: bool | None = None) -> None:
        """Video says 'we're now at this displayed frame'. Pin the tracking
        clock to it and, if the caller told us, mirror its play/pause state
        (so a coach hitting space in the video also freezes the pitch)."""
        next_media_time_ms = max(0.0, float(media_time_ms))
        next_frame_index = int(next_media_time_ms / 40.0)
        # A seek should start a fresh temporal window; mixing frames from two
        # distant moments would produce misleading coaching advice.
        if abs(next_frame_index - self.frame_index) > 2:
            self._coach_history.clear()
        self.media_time_ms = next_media_time_ms
        self.frame_index = next_frame_index
        self.external_clock_last_ms = now_ms()
        if not self.edit_mode:
            t = self.media_time_ms / 1000.0
            match_data.advance(self.players, t)
            self.ball = match_data.ball_position(t)
            self._update_possession()
            self._record_coach_frame()
        if playing is not None:
            was_playing = self.playing
            self.playing = bool(playing)
            # Coming out of edit mode by pressing play in the video is the same
            # gesture as pressing Resume in the sidebar -- drop the freeze.
            if self.playing and self.edit_mode:
                self.edit_mode = False
                self.grabbed.clear()
            if was_playing != self.playing:
                self._bump()
        self._bump()

    def reset_scenario(self) -> None:
        self.players = match_data.build_players()
        self.grabbed.clear()
        self.edit_mode = False
        self.playing = True
        self.media_time_ms = 0.0
        self.frame_index = 0
        self.ball = match_data.ball_position(0.0)
        self._coach_history.clear()
        self._record_coach_frame()
        self._bump()

    def load_match(self, match_id: str) -> bool:
        """Switch the active test match; rewinds and clears any edits. Returns
        False (state untouched) if the match couldn't be loaded."""
        if not match_data.select(match_id):
            return False
        self.match_id = match_data.current_id()
        self.match_label = match_data.current_label()
        self.players = match_data.build_players()
        self.grabbed.clear()
        self.edit_mode = False
        self.calibration_overlay = False
        self.offside_overlay = False
        self.compactness_overlay = False
        self.shadow_overlay = False
        self.pitch_control_overlay = False
        self.formation_overlay = False
        self.suggested_overlay = False
        self._suggested_cache_key = None
        self._suggested_cache = []
        self.experiments = dict(EXPERIMENT_DEFAULTS)
        self._experimental_cache_key = None
        self._experimental_cache = None
        self.playing = True
        self.media_time_ms = 0.0
        self.frame_index = 0
        self.ball = match_data.ball_position(0.0)
        self._coach_history.clear()
        self._record_coach_frame()
        self._bump()
        return True

    def _record_coach_frame(self) -> None:
        """Store one raw sample per tracking frame, replacing same-frame edits."""
        sample = {
            "frameIndex": self.frame_index,
            "mediaTimeMs": round(self.media_time_ms, 1),
            "possession": self.possession,
            "ball": {"x": round(self.ball[0], 2), "y": round(self.ball[1], 2)},
            "players": self._players_snapshot(),
        }
        if self._coach_history and self._coach_history[-1]["frameIndex"] == self.frame_index:
            self._coach_history[-1] = sample
        else:
            self._coach_history.append(sample)

    def coach_frame_inputs(self) -> list[dict]:
        """Return five meaningful snapshots over the latest 1.6 seconds."""
        self._record_coach_frame()
        by_index = {frame["frameIndex"]: frame for frame in self._coach_history}
        desired_indices = sorted({
            max(0, self.frame_index - offset * COACH_SNAPSHOT_SPACING_FRAMES)
            for offset in range(COACH_SNAPSHOT_COUNT - 1, -1, -1)
        })
        ordered: list[dict] = []
        fallback_possession = self.possession
        for frame_index in desired_indices:
            frame = by_index.get(frame_index)
            if frame is None:
                frame = self._sample_match_frame(frame_index, fallback_possession)
            fallback_possession = frame["possession"]
            ordered.append(frame)
        return [
            {
                **frame,
                "ball": dict(frame["ball"]),
                "players": [dict(player) for player in frame["players"]],
            }
            for frame in ordered
        ]

    @staticmethod
    def _sample_match_frame(frame_index: int, fallback_possession: str) -> dict:
        """Reconstruct a raw tracking frame when the browser jumped to it."""
        players = match_data.build_players()
        media_time_ms = frame_index * 40.0
        match_data.advance(players, media_time_ms / 1000.0)
        ball = match_data.ball_position(media_time_ms / 1000.0)
        nearest = {"home": math.inf, "away": math.inf}
        for player in players:
            nearest[player.team] = min(
                nearest.get(player.team, math.inf),
                math.hypot(player.x - ball[0], player.y - ball[1]),
            )
        possession = min(nearest, key=nearest.get) if players else fallback_possession
        return {
            "frameIndex": frame_index,
            "mediaTimeMs": media_time_ms,
            "possession": possession,
            "ball": {"x": round(ball[0], 2), "y": round(ball[1], 2)},
            "players": [
                {
                    "id": player.id,
                    "team": player.team,
                    "number": player.number,
                    "x": round(player.x, 2),
                    "y": round(player.y, 2),
                    "vx": round(player.vx, 2),
                    "vy": round(player.vy, 2),
                    "edited": player.edited,
                    "grabbed": False,
                }
                for player in players
            ],
        }

    @staticmethod
    def analyze_coach_frames(frames: list[dict]) -> list[dict]:
        """Calculate the complete tactical model for a captured frame window."""
        analyzed: list[dict] = []
        newest_time = frames[-1]["mediaTimeMs"] if frames else 0.0
        for frame in frames:
            players = [SimpleNamespace(**player) for player in frame["players"]]
            ball = (float(frame["ball"]["x"]), float(frame["ball"]["y"]))
            indicators = analyze_experimental(
                players,
                ball,
                frame["possession"],
                PITCH_LENGTH,
                PITCH_WIDTH,
                include_receiver_targets=True,
                receiver_target_limit=None,
                include_hold_targets=True,
            )
            analyzed.append({
                **frame,
                "relativeTimeMs": round(frame["mediaTimeMs"] - newest_time, 1),
                "analysis": indicators,
            })
        return analyzed

    def toggle_calibration(self) -> None:
        self.calibration_overlay = not self.calibration_overlay
        self._bump()

    def toggle_offside(self) -> None:
        self.offside_overlay = not self.offside_overlay
        self._bump()

    def toggle_compactness(self) -> None:
        self.compactness_overlay = not self.compactness_overlay
        self._bump()

    def toggle_shadows(self) -> None:
        self.shadow_overlay = not self.shadow_overlay
        self._bump()

    def toggle_pitch_control(self) -> None:
        self.pitch_control_overlay = not self.pitch_control_overlay
        self._bump()

    def toggle_formation(self) -> None:
        self.formation_overlay = not self.formation_overlay
        self._bump()

    def toggle_suggested(self) -> None:
        self.suggested_overlay = not self.suggested_overlay
        self._suggested_cache_key = None
        self._bump()

    def suggested_positions_snapshot(self) -> list[dict]:
        """Ghost-position suggestions for the possession team. Empty when off.
        Cached per (frame, revision, ball) so a paused edit that shifts a
        defender re-runs it, but idle playback doesn't."""
        if not self.suggested_overlay:
            return []
        cache_key = (
            self.frame_index,
            self.revision,
            self.possession,
            round(self.ball[0], 2),
            round(self.ball[1], 2),
        )
        if cache_key != self._suggested_cache_key:
            self._suggested_cache = suggested_positions(
                self.players,
                self.ball,
                self.possession,
                PITCH_LENGTH,
                PITCH_WIDTH,
            )
            self._suggested_cache_key = cache_key
        return self._suggested_cache

    def set_experiment(self, name: str, enabled: bool | None = None) -> bool:
        """Enable/toggle one allow-listed experiment; return False if unknown."""
        if name not in self.experiments:
            return False
        current = self.experiments[name]
        self.experiments[name] = (not current) if enabled is None else bool(enabled)
        self._experimental_cache_key = None
        self._bump()
        return True

    def experimental_analysis(self) -> dict | None:
        """Compute analytics only while an experiment is explicitly enabled.

        Tracking data advances at 25 Hz while state snapshots are broadcast at
        30 Hz.  Keying the cache by frame avoids doing identical work twice and
        keeps the default (all flags off) path at effectively zero cost.
        """
        if not any(self.experiments.values()):
            return None
        cache_key = (
            self.frame_index,
            self.revision,
            self.possession,
            round(self.ball[0], 2),
            round(self.ball[1], 2),
            self.experiments["receiverTargets"],
        )
        if cache_key != self._experimental_cache_key:
            self._experimental_cache = analyze_experimental(
                self.players,
                self.ball,
                self.possession,
                PITCH_LENGTH,
                PITCH_WIDTH,
                include_receiver_targets=self.experiments["receiverTargets"],
            )
            self._experimental_cache_key = cache_key
        return self._experimental_cache

    def team_formations(self) -> dict:
        """{home: "4-3-3", away: "4-4-2"} or empty strings when unknown. Runs
        only when the overlay is on; each side inferred from that team's own
        centroid so it stays correct after half-time flips direction."""
        if not self.formation_overlay:
            return {"home": "", "away": ""}
        home = [p for p in self.players if p.team == "home"]
        away = [p for p in self.players if p.team == "away"]
        if not home or not away:
            return {"home": "", "away": ""}
        home_defends_left = (
            sum(p.x for p in home) / len(home) < sum(p.x for p in away) / len(away)
        )

        def formation_of(team_players: list, defends_left: bool) -> str:
            if len(team_players) < 7:
                return ""
            depth = (lambda p: p.x) if defends_left else (lambda p: PITCH_LENGTH - p.x)
            outfield = sorted(team_players, key=depth)[1:]   # drop GK (deepest)
            return detect_formation([depth(p) for p in outfield])

        return {
            "home": formation_of(home, home_defends_left),
            "away": formation_of(away, not home_defends_left),
        }

    def set_shadow_seconds(self, seconds: float) -> None:
        self.shadow_seconds = min(max(float(seconds), SHADOW_SECONDS_MIN), SHADOW_SECONDS_MAX)
        self._bump()

    def defender_shadows(self) -> list[dict]:
        """Reach polygons for the team without the ball. Empty unless the
        overlay is on -- this is the only per-frame cost it adds."""
        if not self.shadow_overlay:
            return []
        defending = self.defending_team()
        return [
            {
                "playerId": p.id,
                "team": p.team,
                "points": reach_polygon(
                    p.x, p.y, p.vx, p.vy, self.shadow_seconds, PITCH_LENGTH, PITCH_WIDTH
                ),
            }
            for p in self.players
            if p.team == defending
        ]

    def drag_start(self, player_id: str, bx: float, by: float, owner: str) -> None:
        p = self.player_by_id(player_id)
        if p is None:
            return
        self.enter_edit_mode()
        self.grabbed[owner] = player_id
        p.edited = True
        p.x, p.y = self.board_to_pitch(bx, by)
        # A coach-placed player has no momentum: their reach shadow is a circle
        # from a standing start, not a teardrop leaning wherever they happened
        # to be running before the drag. Zeroed here rather than in
        # match_data.advance() because edit mode freezes the replay clock, so
        # advance() never runs while a drag is in progress.
        p.vx = p.vy = 0.0
        self._bump()

    def drag_move(self, player_id: str, bx: float, by: float, owner: str) -> None:
        if self.grabbed.get(owner) != player_id:
            return  
        p = self.player_by_id(player_id)
        if p is None:
            return
        x, y = self.board_to_pitch(bx, by)
        p.x = min(max(x, 0.0), PITCH_LENGTH)
        p.y = min(max(y, 0.0), PITCH_WIDTH)
        p.vx = p.vy = 0.0
        self._bump()

    def drag_end(self, owner: str) -> None:
        if self.grabbed.pop(owner, None) is not None:
            self._bump()

    def handle_vision_event(self, evt: dict) -> None:
        etype = evt.get("type")

        if etype == "vision_stats":
            received_at_ms = now_ms()
            captured_at_ms = evt.get("capturedAtMs", received_at_ms)
            self.vision_stats = {
                "fps": evt.get("fps", 0.0),
                "captureFps": evt.get("captureFps", 0.0),
                "hands": evt.get("hands", 0),
                "calibrated": evt.get("calibrated", False),
                "handPx": evt.get("handPx", 0),
                "inferenceMs": evt.get("inferenceMs", 0.0),
                "captureToInferenceMs": evt.get("captureToInferenceMs", 0.0),
                "captureToServerMs": round(
                    max(0.0, received_at_ms - captured_at_ms), 1
                ),
                "captureDrops": evt.get("captureDrops", 0),
                "cameraFps": evt.get("cameraFps", 0.0),
                "submittedFrames": evt.get("submittedFrames", 0),
                "completedFrames": evt.get("completedFrames", 0),
                "capturedAtMs": captured_at_ms,
            }
            return

        if etype == "hand_lost":
            hand = evt.get("handId", "?")
            self.cursors.pop(hand, None)
            self.drag_end(hand)
            return

        hand = evt.get("handId", "?")
        bx, by = evt.get("boardX", 0.0), evt.get("boardY", 0.0)
        received_at_ms = now_ms()
        captured_at_ms = evt.get("capturedAtMs", received_at_ms)
        capture_to_server_ms = max(0.0, received_at_ms - captured_at_ms)
        self.vision_stats["captureToServerMs"] = round(capture_to_server_ms, 1)

        self.cursors[hand] = {
            "boardX": bx,
            "boardY": by,
            "grabbing": etype in ("grab_start", "grab_move"),
            "confidence": evt.get("confidence", 0.0),
            "lastSeenMs": received_at_ms,
            "capturedAtMs": captured_at_ms,
            "inferenceMs": evt.get("inferenceMs", 0.0),
            "captureToServerMs": capture_to_server_ms,
        }

        if etype in ("grab_start", "grab_move"):
            player_id = self.grabbed.get(hand)
            if player_id:
                self.drag_move(player_id, bx, by, owner=hand)
            else:
                x_m, y_m = self.board_to_pitch(bx, by)
                target = self.nearest_player(x_m, y_m)
                if target is not None:
                    self.drag_start(target.id, bx, by, owner=hand)

        elif etype == "grab_end":
            self.drag_end(hand)

    def prune_cursors(self, stale_ms: float = 1200.0) -> None:
        cutoff = now_ms() - stale_ms
        for hand in [h for h, c in self.cursors.items() if c["lastSeenMs"] < cutoff]:
            self.cursors.pop(hand, None)
            self.drag_end(hand)

    def _players_snapshot(self) -> list[dict]:
        return [
            {
                "id": player.id,
                "team": player.team,
                "number": player.number,
                "x": round(player.x, 2),
                "y": round(player.y, 2),
                "vx": round(player.vx, 2),
                "vy": round(player.vy, 2),
                "edited": player.edited,
                "grabbed": player.id in self.grabbed.values(),
            }
            for player in self.players
        ]

    def _cursors_snapshot(self) -> list[dict]:
        return [
            {
                "handId": hand,
                "boardX": round(cursor["boardX"], 4),
                "boardY": round(cursor["boardY"], 4),
                "grabbing": cursor["grabbing"],
                "confidence": round(cursor["confidence"], 2),
                "capturedAtMs": round(cursor.get("capturedAtMs", 0.0), 2),
                "inferenceMs": round(cursor.get("inferenceMs", 0.0), 2),
                "captureToServerMs": round(
                    cursor.get("captureToServerMs", 0.0), 2
                ),
            }
            for hand, cursor in self.cursors.items()
        ]

    def vision_snapshot(self) -> dict:
        """Small state delta sent immediately for latency-sensitive hand input."""
        return {
            "revision": self.revision,
            "playing": self.playing,
            "editMode": self.edit_mode,
            "players": self._players_snapshot(),
            "cursors": self._cursors_snapshot(),
            "vision": self.vision_stats,
            "serverTimestampMs": now_ms(),
        }

    def snapshot(self) -> dict:
        return {
            "revision": self.revision,
            "playing": self.playing,
            "playbackRate": self.playback_rate,
            "frameIndex": self.frame_index,
            "mediaTimeMs": round(self.media_time_ms, 1),
            "editMode": self.edit_mode,
            "calibrationOverlay": self.calibration_overlay,
            "offsideOverlay": self.offside_overlay,
            "compactnessOverlay": self.compactness_overlay,
            "shadowOverlay": self.shadow_overlay,
            "pitchControlOverlay": self.pitch_control_overlay,
            "formationOverlay": self.formation_overlay,
            "suggestedOverlay": self.suggested_overlay,
            "suggestedPositions": self.suggested_positions_snapshot(),
            "experiments": dict(self.experiments),
            "experimentalAnalysis": self.experimental_analysis(),
            "formations": self.team_formations(),
            "shadowSeconds": self.shadow_seconds,
            "possession": self.possession,
            "shadows": self.defender_shadows(),
            "matchId": self.match_id,
            "matchLabel": self.match_label,
            "availableMatches": match_data.list_matches(),
            "events": match_data.recent_events(self.media_time_ms / 1000.0, 8),
            "matchStats": match_data.match_stats(self.media_time_ms / 1000.0),
            "serverTimestampMs": now_ms(),
            "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
            "players": self._players_snapshot(),
            "ball": {"x": round(self.ball[0], 2), "y": round(self.ball[1], 2)},
            "cursors": self._cursors_snapshot(),
            "vision": self.vision_stats,
        }

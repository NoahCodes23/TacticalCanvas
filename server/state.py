import math
import time

from . import match_data
from .analytics.reach import reach_polygon
from .match_data import PITCH_LENGTH, PITCH_WIDTH, Player

GRAB_RADIUS_M = 3.0

# A challenger must get this much closer to the ball than the team currently
# credited with it before possession flips, so a 50/50 doesn't strobe the
# shadow overlay between the two teams every other frame.
POSSESSION_MARGIN_M = 1.5

SHADOW_SECONDS_MIN = 0.5
SHADOW_SECONDS_MAX = 4.0


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

        self.edit_mode = False
        self.calibration_overlay = False
        self.offside_overlay = False
        self.compactness_overlay = False
        self.shadow_overlay = False
        self.shadow_seconds = 2.0
        self.possession = "home"

        self.players: list[Player] = match_data.build_players()
        self.ball = (PITCH_LENGTH / 2, PITCH_WIDTH / 2)
        self.match_id = match_data.current_id()
        self.match_label = match_data.current_label()

        self.grabbed: dict[str, str] = {}
        self.cursors: dict[str, dict] = {}

        self.vision_stats = {"fps": 0.0, "hands": 0, "calibrated": False, "handPx": 0}

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
        if self.playing and not self.edit_mode:
            self.media_time_ms += dt * 1000.0 * self.playback_rate
            self.frame_index = int(self.media_time_ms / 40.0)  # 25Hz tracking data
            t = self.media_time_ms / 1000.0
            match_data.advance(self.players, t)
            self.ball = match_data.ball_position(t)
        # Runs while paused too, so dragging a player in edit mode can hand
        # possession over and flip which team the shadows are drawn for.
        self._update_possession()

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

    def reset_scenario(self) -> None:
        self.players = match_data.build_players()
        self.grabbed.clear()
        self.edit_mode = False
        self.playing = True
        self.media_time_ms = 0.0
        self.frame_index = 0
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
        self.playing = True
        self.media_time_ms = 0.0
        self.frame_index = 0
        self.ball = match_data.ball_position(0.0)
        self._bump()
        return True

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
            self.vision_stats = {
                "fps": evt.get("fps", 0.0),
                "hands": evt.get("hands", 0),
                "calibrated": evt.get("calibrated", False),
                "handPx": evt.get("handPx", 0),
            }
            return

        if etype == "hand_lost":
            hand = evt.get("handId", "?")
            self.cursors.pop(hand, None)
            self.drag_end(hand)
            return

        hand = evt.get("handId", "?")
        bx, by = evt.get("boardX", 0.0), evt.get("boardY", 0.0)

        self.cursors[hand] = {
            "boardX": bx,
            "boardY": by,
            "grabbing": etype in ("grab_start", "grab_move"),
            "confidence": evt.get("confidence", 0.0),
            "lastSeenMs": now_ms(),
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
            "shadowSeconds": self.shadow_seconds,
            "possession": self.possession,
            "shadows": self.defender_shadows(),
            "matchId": self.match_id,
            "matchLabel": self.match_label,
            "availableMatches": match_data.list_matches(),
            "events": match_data.recent_events(self.media_time_ms / 1000.0, 8),
            "serverTimestampMs": now_ms(),
            "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
            "players": [
                {
                    "id": p.id,
                    "team": p.team,
                    "number": p.number,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "vx": round(p.vx, 2),
                    "vy": round(p.vy, 2),
                    "edited": p.edited,
                    "grabbed": p.id in self.grabbed.values(),
                }
                for p in self.players
            ],
            "ball": {"x": round(self.ball[0], 2), "y": round(self.ball[1], 2)},
            "cursors": [
                {
                    "handId": h,
                    "boardX": round(c["boardX"], 4),
                    "boardY": round(c["boardY"], 4),
                    "grabbing": c["grabbing"],
                    "confidence": round(c["confidence"], 2),
                }
                for h, c in self.cursors.items()
            ],
            "vision": self.vision_stats,
        }
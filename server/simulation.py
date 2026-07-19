"""Tactical simulation engine — plays out a coached attacking move.

The coach freezes a moment, and the simulation answers "if we tried to build
through here, what happens?" It pre-plans a forward passing chain from the
possessing team's shape (that's the step list the side panel shows), then
executes it in real time against a reactive defence: defenders and midfielders
hold zones and shift toward the ball, the nearest man presses the carrier, and
a pass caught in a defender's path is intercepted.

Design rules that keep it consistent with the rest of the app:

* **The engine is authoritative and deterministic.** It owns positions while a
  sim is running; ``write_back`` copies them onto the shared ``Player`` objects
  so every existing overlay, snapshot field, and the 2D renderer keep working
  untouched. The 3D view and the 2D pitch are just two readers of the same
  state.
* **dt-based, frame-rate independent.** ``tick(dt)`` integrates; nothing assumes
  a fixed step, so a 30 Hz broadcast and a 60 Hz tick both look right.
* **Plan first, execute second.** The plan (a list of passes ending in a shot)
  is built once at start from the starting shape, so the side panel is stable.
  Execution follows that sequence of shirt numbers but passes to each player's
  *live* position, and the defence reacts, so the outcome is not pre-ordained —
  a pressed lane can still be cut out.

Coordinates are pitch metres. Attack direction (+1 toward x=length) comes from
team centroids, matching the rest of the analytics.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .analytics.experimental import attacking_direction, pass_completion_probability
from .analytics.xg import xg_value
from .analytics.xt import xt_delta

# --- tuning -----------------------------------------------------------------
BALL_SPEED_PASS = 20.0      # m/s, a firm ground pass — quick enough to beat the shift
BALL_SPEED_SHOT = 28.0      # m/s
PLAYER_SPEED_ATT = 7.2      # m/s, attackers a touch quicker than the line
PLAYER_SPEED_DEF = 6.6
MIN_PASS_M = 8.0
MAX_PASS_M = 30.0
INTERCEPT_RADIUS_M = 1.05   # a defender this close to the ball in flight cuts it
INTERCEPT_ARMED_M = 3.0     # ...but only once the ball has cleared the passer
PRESS_STANDOFF_M = 1.7      # the presser contains rather than sitting on the ball
ARRIVE_RADIUS_M = 1.1
TACKLE_RADIUS_M = 1.2        # a defender this close to a dribbler wins the ball
CARRY_MAX_S = 3.5           # dribble this long without progress, then shoot
SHOOT_RANGE_M = 23.0        # inside this distance to goal, the plan ends in a shot
GOAL_HALF_WIDTH_M = 3.66    # 7.32 m goal
PLAN_MAX_STEPS = 6
SETTLE_S = 0.28             # ball rests at a receiver's feet before the next pass
PASS_LEAD_M = 3.0           # lead the receiver into space, so the ball is met
# A move lasts seconds; at 60 Hz this cap is minutes of recording, so it exists
# only as a guard against a sim someone forgot to stop.
MAX_TRAJECTORY_FRAMES = 20_000


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class _SimPlayer:
    __slots__ = ("id", "team", "number", "x", "y", "vx", "vy",
                 "anchor_x", "anchor_y", "zone_r", "role")

    def __init__(self, source: Any) -> None:
        self.id = source.id
        self.team = source.team
        self.number = int(source.number)
        self.x = float(source.x)
        self.y = float(source.y)
        self.vx = 0.0
        self.vy = 0.0
        # Filled in for the defending side: a zone to hold and how far it roams.
        self.anchor_x = self.x
        self.anchor_y = self.y
        self.zone_r = 14.0
        self.role = "attack"


class SimulationEngine:
    """Runs one attacking scenario. Inactive until ``build_and_start``."""

    def __init__(self) -> None:
        self.active = False
        self.playing = False
        self.rate = 1.0
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.attacking_team = "home"
        self.direction = 1
        self.players: list[_SimPlayer] = []
        self.ball = (52.5, 34.0)
        self.steps: list[dict] = []
        self.step_index = -1
        self.phase = "idle"          # idle | settle | travel | done
        self.outcome: str | None = None
        self._settle_t = 0.0
        self._ball_target = (52.5, 34.0)
        self._ball_speed = BALL_SPEED_PASS
        self._carrier_number: int | None = None
        self._start_progress = 0.0
        self._stats = self._empty_stats()
        # Recorded playback: every tick appends one frame, and each step's
        # start captures a full engine checkpoint. Seeking restores a
        # checkpoint — never reverse physics.
        self.trajectory: list[dict] = []
        self._checkpoints: dict[int, dict] = {}
        self._pending_truncate: int | None = None
        self._sequence_probability = 0.0

    # -- lifecycle -----------------------------------------------------------
    @staticmethod
    def _empty_stats() -> dict:
        return {
            "passesCompleted": 0,
            "passesAttempted": 0,
            "passAccuracy": 0,
            "longestPassM": 0.0,
            "topSpeedMps": 0.0,
            "xtGained": 0.0,
            "forwardProgressM": 0.0,
            "elapsedS": 0.0,
            "interceptions": 0,
            "shots": 0,
            "goals": 0,
        }

    def build_and_start(
        self,
        players: Iterable[Any],
        ball: tuple[float, float],
        possession: str,
        pitch_length: float,
        pitch_width: float,
    ) -> bool:
        """Seed from the frozen frame, plan the move, and begin playing.

        Returns False when there isn't enough on the pitch to plan a pass."""
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.attacking_team = possession
        source = list(players)
        self.players = [_SimPlayer(p) for p in source]
        if len([p for p in self.players if p.team == possession]) < 2:
            return False

        self.direction = attacking_direction(source, possession)
        self.ball = (float(ball[0]), float(ball[1]))
        self._assign_defensive_zones()

        self.steps = self._build_plan()
        if not self.steps:
            return False

        self._stats = self._empty_stats()
        self._start_progress = self._progress(self.ball[0])
        self.step_index = 0
        self.outcome = None
        self.active = True
        self.playing = True
        self.trajectory = []
        self._checkpoints = {}
        self._pending_truncate = None
        self._annotate_probabilities()
        self._begin_step(self.steps[0])
        self._record_frame()
        return True

    def pause(self) -> None:
        if self.active:
            self.playing = False

    def resume(self) -> None:
        if self.active and self.phase != "done":
            # Resuming after a backward seek overwrites the abandoned future:
            # frames past the restore point are dropped now (not at seek time,
            # so scrubbing alone never destroys the recording), and stale
            # checkpoints go with them — replay re-captures identical ones.
            if self._pending_truncate is not None:
                del self.trajectory[self._pending_truncate:]
                self._checkpoints = {
                    i: c for i, c in self._checkpoints.items() if i <= self.step_index
                }
                self._pending_truncate = None
                # Re-seed the step-boundary frame the truncation just removed.
                self._record_frame()
            self.playing = True

    def stop(self) -> None:
        self.active = False
        self.playing = False
        self.phase = "idle"
        self.step_index = -1
        self.steps = []
        self.trajectory = []
        self._checkpoints = {}
        self._pending_truncate = None
        self._sequence_probability = 0.0

    def set_rate(self, rate: float) -> None:
        self.rate = _clamp(float(rate), 0.25, 4.0)

    def seek_step(self, index: int) -> str | None:
        """Jump to a step's start frame and pause there. Returns an error
        reason, or None on success.

        Restores the full engine checkpoint captured when the step began, so
        resuming replays forward through the same deterministic engine —
        scrubbing is state restore plus normal playback, never reverse physics.
        Only steps that have actually played have a checkpoint; the forecast
        tail is not seekable until the move reaches it.

        The valid range spans the *recorded timeline*, not just the current
        step list: seeking backward restores that moment's shorter forecast,
        and steps the move actually played beyond it stay reachable through
        their checkpoints (each checkpoint restores its own step list)."""
        if not self.active:
            return "no simulation is running"
        hi = max(len(self.steps) - 1, max(self._checkpoints, default=-1))
        if index < 0 or index > hi:
            return f"step index {index} is out of range (0-{hi})"
        cp = self._checkpoints.get(index)
        if cp is None:
            return f"step {index} has not played yet"
        self.steps = [dict(s) for s in cp["steps"]]
        self.step_index = cp["stepIndex"]
        by_id = {p.id: p for p in self.players}
        for pid, x, y, vx, vy in cp["players"]:
            p = by_id.get(pid)
            if p is not None:
                p.x, p.y, p.vx, p.vy = x, y, vx, vy
        self.ball = cp["ball"]
        self._ball_target = cp["ballTarget"]
        self._ball_speed = cp["ballSpeed"]
        self._ball_origin = cp["ballOrigin"]
        self._carrier_number = cp["carrier"]
        self._settle_t = cp["settleT"]
        self._carry_t = 0.0
        self._stats = dict(cp["stats"])
        self.phase = "settle"
        self.outcome = None
        self.playing = False
        self._pending_truncate = cp["frame"]
        self._annotate_probabilities()
        return None

    # -- geometry helpers ----------------------------------------------------
    def _progress(self, x: float) -> float:
        """Distance up-pitch in the attacking direction (0 = own goal-line)."""
        return x if self.direction > 0 else self.pitch_length - x

    def _goal(self) -> tuple[float, float]:
        return (self.pitch_length if self.direction > 0 else 0.0, self.pitch_width / 2.0)

    def _dist_to_goal(self, x: float, y: float) -> float:
        gx, gy = self._goal()
        return math.hypot(gx - x, gy - y)

    def _attackers(self) -> list[_SimPlayer]:
        return [p for p in self.players if p.team == self.attacking_team]

    def _defenders(self) -> list[_SimPlayer]:
        return [p for p in self.players if p.team != self.attacking_team]

    def _by_number(self, number: int | None) -> _SimPlayer | None:
        if number is None:
            return None
        for p in self.players:
            if p.team == self.attacking_team and p.number == number:
                return p
        return None

    def _nearest_defender_dist(self, x: float, y: float) -> float:
        best = math.inf
        for d in self._defenders():
            best = min(best, math.hypot(d.x - x, d.y - y))
        return best if best != math.inf else 50.0

    def _lane_clearance(self, ax: float, ay: float, bx: float, by: float) -> float:
        """Smallest distance from any defender to the a→b passing segment.

        A pass is only as safe as the tightest point a defender can step into,
        so the planner scores lanes on this, not just on where the receiver
        stands. Uses the standard point-to-segment projection."""
        vx, vy = bx - ax, by - ay
        seg2 = vx * vx + vy * vy
        best = math.inf
        for d in self._defenders():
            if seg2 <= 1e-6:
                dist = math.hypot(d.x - ax, d.y - ay)
            else:
                t = _clamp(((d.x - ax) * vx + (d.y - ay) * vy) / seg2, 0.0, 1.0)
                px, py = ax + t * vx, ay + t * vy
                dist = math.hypot(d.x - px, d.y - py)
            best = min(best, dist)
        return best if best != math.inf else 50.0

    def _assign_defensive_zones(self) -> None:
        """Give each defender a home zone and a roam radius by role.

        Role is read from how far up their own defensive third a player starts:
        the deep block stays compact near goal, midfielders roam, the front line
        presses high. Anchors are frozen at kickoff so the shape has something to
        recover toward instead of collapsing onto the ball.
        """
        defs = self._defenders()
        if not defs:
            return
        # Progress *for the defending team* is measured toward their own goal,
        # i.e. the opposite direction to the attack.
        prog = [(self.pitch_length - self._progress(p.x)) for p in defs]
        lo, hi = min(prog), max(prog)
        span = max(1.0, hi - lo)
        for p, pr in zip(defs, prog):
            frac = (pr - lo) / span   # 0 = deepest (near own goal), 1 = highest
            p.anchor_x, p.anchor_y = p.x, p.y
            if frac < 0.34:
                p.role, p.zone_r = "back", 11.0
            elif frac < 0.7:
                p.role, p.zone_r = "mid", 16.0
            else:
                p.role, p.zone_r = "press", 21.0

    # -- planning ------------------------------------------------------------
    def _build_plan(self) -> list[dict]:
        """Greedy forward passing chain from the ball, ending in a shot.

        Uses the frozen shape only — execution reacts live. Each candidate is
        scored on forward progress, how open the receiver is, and xT gained, so
        the skeleton favours safe, penetrating balls, which is also what makes
        it usually survive the defence."""
        attackers = self._attackers()
        carrier = min(
            attackers,
            key=lambda p: math.hypot(p.x - self.ball[0], p.y - self.ball[1]),
            default=None,
        )
        if carrier is None:
            return []

        steps: list[dict] = []
        visited = {carrier.number}
        cur = carrier
        self._carrier_number = carrier.number

        for _ in range(PLAN_MAX_STEPS):
            if self._dist_to_goal(cur.x, cur.y) <= SHOOT_RANGE_M:
                steps.append(self._shot_step(len(steps), cur))
                break

            best, best_score = None, -math.inf
            cur_prog = self._progress(cur.x)
            for t in attackers:
                if t.number in visited:
                    continue
                d = math.hypot(t.x - cur.x, t.y - cur.y)
                if d < MIN_PASS_M or d > MAX_PASS_M:
                    continue
                forward = self._progress(t.x) - cur_prog
                if forward < -6.0:        # allow a small switch back, not a retreat
                    continue
                clearance = self._lane_clearance(cur.x, cur.y, t.x, t.y)
                if clearance < 3.2:       # lane is blocked -- a defender can cut it
                    continue
                openness = self._nearest_defender_dist(t.x, t.y)
                xtg = xt_delta((cur.x, cur.y), (t.x, t.y),
                               self.direction, self.pitch_length, self.pitch_width)
                # Clearance dominates: a safe, progressive ball beats a spectacular
                # one through traffic. That keeps the coached move believable.
                score = (1.2 * forward + 1.0 * openness
                         + 3.4 * clearance + 35.0 * xtg)
                if score > best_score:
                    best, best_score = t, score

            if best is None:
                # Nothing progressive on; take it on from here rather than stall.
                steps.append(self._shot_step(len(steps), cur))
                break

            steps.append(self._pass_step(len(steps), cur, best))
            visited.add(best.number)
            cur = best

        return steps

    def _pass_step(self, index: int, frm: _SimPlayer, to: _SimPlayer) -> dict:
        distance = math.hypot(to.x - frm.x, to.y - frm.y)
        return {
            "index": index,
            "type": "pass",
            "fromNumber": frm.number,
            "toNumber": to.number,
            "distanceM": round(distance, 1),
            "label": f"#{frm.number} → #{to.number}",
            "detail": self._pass_detail(frm, to, distance),
            "status": "pending",
            "successProbability": self._pass_probability(frm, to),
        }

    def _shot_step(self, index: int, frm: _SimPlayer) -> dict:
        distance = self._dist_to_goal(frm.x, frm.y)
        return {
            "index": index,
            "type": "shot",
            "fromNumber": frm.number,
            "toNumber": None,
            "distanceM": round(distance, 1),
            "label": f"#{frm.number} shoots",
            "detail": f"{distance:.0f} m strike on goal",
            "status": "pending",
            "successProbability": self._shot_probability(frm),
        }

    # -- probabilities -------------------------------------------------------
    # Both numbers come from the analytics the rest of the app already trusts:
    # passes are priced by the experimental pass-completion scorer, shots by
    # the xG model. Nothing here invents a formula.
    def _pass_probability(self, frm: _SimPlayer, to: _SimPlayer) -> float:
        completion = pass_completion_probability(
            self.players, frm, to, (frm.x, frm.y),
            self.direction, self.pitch_length, self.pitch_width,
        )
        return round(completion, 3)

    def _shot_probability(self, frm: _SimPlayer) -> float:
        return round(
            xg_value(frm.x, frm.y, self.direction,
                     self.pitch_length, self.pitch_width), 3)

    def _annotate_probabilities(self) -> None:
        """Refresh each step's cumulative sequenceProbability.

        Per-step probabilities are priced when a step is planned or rewritten
        (positions at decision time); this only re-multiplies the chain so the
        cumulative numbers stay consistent after any trim/extend/rewrite."""
        running = 1.0
        for s in self.steps:
            running *= s.get("successProbability", 1.0)
            s["sequenceProbability"] = round(running, 4)
        self._sequence_probability = round(running, 4) if self.steps else 0.0

    # -- recording -----------------------------------------------------------
    def _record_frame(self) -> None:
        """Append the current world to the trajectory buffer (one per tick)."""
        if len(self.trajectory) >= MAX_TRAJECTORY_FRAMES:
            return
        self.trajectory.append({
            "t": round(self._stats["elapsedS"], 3),
            "step": self.step_index,
            "ball": [round(self.ball[0], 2), round(self.ball[1], 2)],
            "players": [[p.id, round(p.x, 2), round(p.y, 2)] for p in self.players],
        })

    def _pass_detail(self, frm: _SimPlayer, to: _SimPlayer, distance: float) -> str:
        forward = self._progress(to.x) - self._progress(frm.x)
        if forward > 14:
            kind = "line-breaking ball"
        elif forward > 4:
            kind = "progressive pass"
        elif forward < -1:
            kind = "switch back"
        else:
            kind = "square ball"
        return f"{distance:.0f} m {kind}"

    # -- execution -----------------------------------------------------------
    # The plan built at kickoff is a *forecast* for the side panel. Execution
    # re-derives each pass against live positions at the moment of the pass,
    # because by then the defence has collapsed toward the ball and a lane that
    # was open at kickoff may be shut. This is both more realistic and far more
    # reliable than blindly firing the frozen chain into a defender.
    def _begin_step(self, step: dict) -> None:
        step["status"] = "active"
        # The carrier is whoever actually has the ball (set at kickoff or by the
        # previous completed pass) -- not the forecast's fromNumber, which may be
        # stale now that execution re-derives each pass live.
        if self._carrier_number is None:
            carrier = self._by_number(step.get("fromNumber"))
            if carrier is not None:
                self._carrier_number = carrier.number
        else:
            step["fromNumber"] = self._carrier_number
        self.phase = "settle"
        self._settle_t = SETTLE_S
        # Full engine checkpoint at the step boundary. seek_step restores one
        # of these; the frame index ties it to the trajectory buffer.
        self._checkpoints[step["index"]] = {
            "frame": len(self.trajectory),
            "steps": [dict(s) for s in self.steps],
            "stepIndex": self.step_index,
            "players": [(p.id, p.x, p.y, p.vx, p.vy) for p in self.players],
            "ball": self.ball,
            "ballTarget": self._ball_target,
            "ballSpeed": self._ball_speed,
            "ballOrigin": getattr(self, "_ball_origin", self.ball),
            "carrier": self._carrier_number,
            "settleT": self._settle_t,
            "stats": dict(self._stats),
        }

    def _select_receiver_live(self, carrier: _SimPlayer) -> _SimPlayer | None:
        """Best teammate to receive right now.

        Execution is less picky than the planner: a settled defence may have
        stepped into the ideal lane, and the right answer then is to recycle the
        ball, not to launch a hopeless shot from deep. So it prefers the best
        progressive option but always falls back to the safest available pass;
        it only returns None when literally no teammate is within pass range."""
        cur_prog = self._progress(carrier.x)
        best, best_score = None, -math.inf
        fallback, fallback_safe = None, -math.inf
        for t in self._attackers():
            if t.number == carrier.number:
                continue
            d = math.hypot(t.x - carrier.x, t.y - carrier.y)
            if d < MIN_PASS_M or d > MAX_PASS_M:
                continue
            clearance = self._lane_clearance(carrier.x, carrier.y, t.x, t.y)
            openness = self._nearest_defender_dist(t.x, t.y)
            # Safest ball of any direction, as a recycle fallback.
            safe = min(clearance, openness)
            if safe > fallback_safe:
                fallback, fallback_safe = t, safe
            forward = self._progress(t.x) - cur_prog
            if forward < -6.0 or clearance < 2.4:
                continue
            score = 1.2 * forward + 1.0 * openness + 3.4 * clearance
            if score > best_score:
                best, best_score = t, score
        return best if best is not None else fallback

    def _launch_ball(self, step: dict) -> None:
        """Decide and kick the pass/shot for the active step, live."""
        carrier = self._by_number(self._carrier_number)
        if carrier is None:
            self.outcome = "complete"
            self.phase = "done"
            self.playing = False
            return

        last_allowed = self.step_index >= PLAN_MAX_STEPS - 1
        in_range = self._dist_to_goal(carrier.x, carrier.y) <= SHOOT_RANGE_M

        if in_range or last_allowed:
            self._fire_shot(step, carrier)
            return
        receiver = self._select_receiver_live(carrier)
        if receiver is None:
            # No one in pass range and too far to shoot: drive at the goal.
            self._enter_carry()
            return
        self._fire_pass(step, carrier, receiver)

    def _fire_shot(self, step: dict, carrier: _SimPlayer) -> None:
        # Rewrite this step to the shot it became and trim the forecast tail.
        self._rewrite_shot(step, carrier)
        self.steps = self.steps[: self.step_index + 1]
        self._annotate_probabilities()
        self._ball_target = self._goal()
        self._ball_speed = BALL_SPEED_SHOT
        self.phase = "travel"
        self._ball_origin = self.ball
        self._carrier_number = None

    def _fire_pass(self, step: dict, carrier: _SimPlayer, receiver: _SimPlayer) -> None:
        self._rewrite_pass(step, carrier, receiver)
        ux = 1.0 if self.direction > 0 else -1.0
        lead = min(PASS_LEAD_M, math.hypot(receiver.x - self.ball[0],
                                           receiver.y - self.ball[1]) * 0.18)
        tx = _clamp(receiver.x + ux * lead, 1.0, self.pitch_length - 1.0)
        self._ball_target = (tx, receiver.y)
        self._ball_speed = BALL_SPEED_PASS
        self._stats["passesAttempted"] += 1
        self.phase = "travel"
        self._ball_origin = self.ball
        self._carrier_number = None

    def _enter_carry(self) -> None:
        self.phase = "carry"
        self._carry_t = 0.0

    def _rewrite_pass(self, step: dict, frm: _SimPlayer, to: _SimPlayer) -> None:
        distance = math.hypot(to.x - frm.x, to.y - frm.y)
        step.update(type="pass", fromNumber=frm.number, toNumber=to.number,
                    distanceM=round(distance, 1), label=f"#{frm.number} → #{to.number}",
                    detail=self._pass_detail(frm, to, distance),
                    successProbability=self._pass_probability(frm, to))
        self._annotate_probabilities()

    def _rewrite_shot(self, step: dict, frm: _SimPlayer) -> None:
        distance = self._dist_to_goal(frm.x, frm.y)
        step.update(type="shot", fromNumber=frm.number, toNumber=None,
                    distanceM=round(distance, 1), label=f"#{frm.number} shoots",
                    detail=f"{distance:.0f} m strike on goal",
                    successProbability=self._shot_probability(frm))
        self._annotate_probabilities()

    def tick(self, dt: float) -> None:
        if not self.active or not self.playing or self.phase == "done":
            return
        step = dt * self.rate
        self._stats["elapsedS"] += step

        if self.phase == "settle":
            self._settle_t -= step
            self._hold_ball_at_carrier()
            self._move_players(step)
            if self._settle_t <= 0:
                self._launch_ball(self.steps[self.step_index])
            self._finish_tick(step)
            return

        if self.phase == "travel":
            self._advance_ball(step)
            # Arrival may have advanced to the next step or ended the plan; in
            # either case there's no live pass left to move players onto or cut.
            if self.phase != "travel":
                self._finish_tick(step)
                return
            self._move_players(step)
            self._check_interception()
            self._finish_tick(step)
            return

        if self.phase == "carry":
            self._carry_tick(step)
            self._finish_tick(step)

    def _carry_tick(self, dt: float) -> None:
        """The carrier drives at goal until a shot, a forward pass, or a tackle.

        Movement is handled through the normal steering (the carrier's target is
        the goal while carrying); here we just keep the ball at their feet and
        decide when the dribble resolves."""
        self._carry_t += dt
        carrier = self._by_number(self._carrier_number)
        if carrier is None:
            self.outcome = "complete"
            self.phase = "done"
            self.playing = False
            return
        self._move_players(dt)
        # Ball travels just ahead of the carrier, toward goal.
        gx, gy = self._goal()
        d = math.hypot(gx - carrier.x, gy - carrier.y)
        ux, uy = ((gx - carrier.x) / d, (gy - carrier.y) / d) if d > 1e-3 else (1.0, 0.0)
        self.ball = (carrier.x + ux * 1.2, carrier.y + uy * 1.2)

        # Dispossessed if a defender gets touch-tight to the carrier.
        for defender in self._defenders():
            if math.hypot(defender.x - carrier.x, defender.y - carrier.y) <= TACKLE_RADIUS_M:
                step = self.steps[self.step_index]
                step["status"] = "failed"
                step["detail"] = f"#{carrier.number} dispossessed"
                self._stats["interceptions"] += 1
                self.outcome = "intercepted"
                self.phase = "done"
                self.playing = False
                self._carrier_number = None
                return

        step = self.steps[self.step_index]
        if self._dist_to_goal(carrier.x, carrier.y) <= SHOOT_RANGE_M or self._carry_t >= CARRY_MAX_S:
            self._fire_shot(step, carrier)
            return
        # Break the dribble for a genuinely progressive pass, if one has opened.
        receiver = self._select_receiver_live(carrier)
        if receiver is not None and (self._progress(receiver.x) - self._progress(carrier.x)) > 5.0:
            self._fire_pass(step, carrier, receiver)

    def _hold_ball_at_carrier(self) -> None:
        carrier = self._by_number(self._carrier_number)
        if carrier is not None:
            ux = 0.6 if self.direction > 0 else -0.6
            self.ball = (carrier.x + ux, carrier.y)

    def _advance_ball(self, dt: float) -> None:
        bx, by = self.ball
        tx, ty = self._ball_target
        dx, dy = tx - bx, ty - by
        d = math.hypot(dx, dy)
        travel = self._ball_speed * dt
        if d <= max(ARRIVE_RADIUS_M, travel):
            self.ball = (tx, ty)
            self._on_ball_arrival()
        else:
            self.ball = (bx + dx / d * travel, by + dy / d * travel)

    def _on_ball_arrival(self) -> None:
        step = self.steps[self.step_index]
        if step["type"] == "shot":
            _, gy = self._goal()
            on_target = abs(self.ball[1] - gy) <= GOAL_HALF_WIDTH_M + 1.5
            # A close, on-target strike is a goal; a hopeful long-range effort is
            # a chance created but saved or off. Distance is the shot's own range.
            scores = on_target and step["distanceM"] <= 20.0
            step["status"] = "done"
            self.outcome = "goal" if scores else "shot"
            self._stats["shots"] = self._stats.get("shots", 0) + 1
            if scores:
                self._stats["goals"] = self._stats.get("goals", 0) + 1
            self.phase = "done"
            self.playing = False
            self._recompute_derived()
            return

        # Completed pass.
        step["status"] = "done"
        self._stats["passesCompleted"] += 1
        self._stats["longestPassM"] = max(self._stats["longestPassM"], step["distanceM"])
        receiver = self._by_number(step["toNumber"])
        frm = self._by_number(step["fromNumber"])
        if receiver is not None and frm is not None:
            xtg = xt_delta((frm.x, frm.y), (receiver.x, receiver.y),
                           self.direction, self.pitch_length, self.pitch_width)
            self._stats["xtGained"] += max(0.0, xtg)
        self._carrier_number = step["toNumber"]

        self.step_index += 1
        if self.step_index >= PLAN_MAX_STEPS:
            # Hard cap on a move that just kept its shape: a sustained, if
            # unresolved, spell of possession.
            self.outcome = "complete"
            self.phase = "done"
            self.playing = False
            self._recompute_derived()
            return
        if self.step_index >= len(self.steps):
            # The forecast ran out but we still have the ball: extend the plan
            # with a final action from the receiver, so every move drives toward
            # a shot (or a carry that becomes one) instead of ending on a pass.
            target = receiver or self._by_number(self._carrier_number)
            if target is not None:
                self.steps.append(self._shot_step(self.step_index, target))
                self._annotate_probabilities()
        self._begin_step(self.steps[self.step_index])
        self._recompute_derived()

    def _check_interception(self) -> None:
        # Only passes are intercepted. A shot travelling through bodies is
        # blocked or saved, not turned over -- it resolves at the goal.
        if self.steps[self.step_index]["type"] == "shot":
            return
        bx, by = self.ball
        # A pass isn't live to interception until it has cleared the passer --
        # otherwise a defender containing the carrier cuts every ball at launch,
        # which is not how playing out of pressure works.
        ox, oy = getattr(self, "_ball_origin", self.ball)
        if math.hypot(bx - ox, by - oy) < INTERCEPT_ARMED_M:
            return
        for d in self._defenders():
            if math.hypot(d.x - bx, d.y - by) <= INTERCEPT_RADIUS_M:
                step = self.steps[self.step_index]
                step["status"] = "failed"
                self._stats["interceptions"] += 1
                self.outcome = "intercepted"
                self.phase = "done"
                self.playing = False
                self._carrier_number = None
                self._recompute_derived()
                return

    # -- steering ------------------------------------------------------------
    def _move_players(self, dt: float) -> None:
        top = self._stats["topSpeedMps"]
        for p in self.players:
            if p.team == self.attacking_team:
                tx, ty = self._attacker_target(p)
                speed = PLAYER_SPEED_ATT
            else:
                tx, ty = self._defender_target(p)
                speed = PLAYER_SPEED_DEF
            top = max(top, self._step_toward(p, tx, ty, speed, dt))
        self._stats["topSpeedMps"] = top

    def _step_toward(self, p: _SimPlayer, tx: float, ty: float,
                     speed: float, dt: float) -> float:
        dx, dy = tx - p.x, ty - p.y
        d = math.hypot(dx, dy)
        if d < 1e-4:
            p.vx = p.vy = 0.0
            return 0.0
        move = min(d, speed * dt)
        p.x = _clamp(p.x + dx / d * move, 0.0, self.pitch_length)
        p.y = _clamp(p.y + dy / d * move, 0.0, self.pitch_width)
        p.vx, p.vy = dx / d * (move / dt), dy / d * (move / dt)
        return math.hypot(p.vx, p.vy)

    def _attacker_target(self, p: _SimPlayer) -> tuple[float, float]:
        step = self.steps[self.step_index] if 0 <= self.step_index < len(self.steps) else None
        if step is None:
            return p.x, p.y
        ux = 1.0 if self.direction > 0 else -1.0
        # While carrying, the man on the ball drives straight at the goal.
        if self.phase == "carry" and p.number == self._carrier_number:
            gx, gy = self._goal()
            return _clamp(gx - ux * 6.0, 1.0, self.pitch_length - 1.0), gy
        # The intended receiver runs onto the ball's target.
        if step["type"] == "pass" and p.number == step["toNumber"]:
            return self._ball_target
        # The carrier makes a short supporting move after releasing.
        if p.number == step["fromNumber"]:
            return _clamp(p.x + ux * 4.0, 0.0, self.pitch_length), p.y
        # Everyone else drifts forward into space, holding their lane width.
        return _clamp(p.x + ux * 6.0, 0.0, self.pitch_length), p.y

    def _defender_target(self, p: _SimPlayer) -> tuple[float, float]:
        bx, by = self.ball
        # Nearest defender to the ball presses it directly.
        nearest = min(self._defenders(),
                      key=lambda d: math.hypot(d.x - bx, d.y - by), default=None)
        if nearest is not None and nearest.id == p.id:
            # Contain at a standoff rather than overlapping the ball, so the
            # carrier can release before pressure is fully on top of them.
            dx, dy = p.x - bx, p.y - by
            d = math.hypot(dx, dy)
            if d <= PRESS_STANDOFF_M:
                return p.x, p.y
            return bx + dx / d * PRESS_STANDOFF_M, by + dy / d * PRESS_STANDOFF_M
        # Others hold their zone but shift toward the ball, clamped to the zone.
        shift = 0.38
        tx = p.anchor_x + (bx - p.anchor_x) * shift
        ty = p.anchor_y + (by - p.anchor_y) * shift
        dx, dy = tx - p.anchor_x, ty - p.anchor_y
        d = math.hypot(dx, dy)
        if d > p.zone_r:
            tx = p.anchor_x + dx / d * p.zone_r
            ty = p.anchor_y + dy / d * p.zone_r
        return tx, ty

    # -- stats / output ------------------------------------------------------
    def _recompute_derived(self) -> None:
        attempted = self._stats["passesAttempted"]
        completed = self._stats["passesCompleted"]
        self._stats["passAccuracy"] = round(100 * completed / attempted) if attempted else 0
        self._stats["forwardProgressM"] = max(
            0.0, self._progress(self.ball[0]) - self._start_progress)

    def _finish_tick(self, dt: float) -> None:
        self._recompute_derived()
        self._record_frame()

    def write_back(self, players: Iterable[Any]) -> None:
        """Copy live sim positions onto the shared Player objects by id."""
        by_id = {p.id: p for p in self.players}
        for target in players:
            src = by_id.get(target.id)
            if src is None:
                continue
            target.x, target.y = src.x, src.y
            target.vx, target.vy = src.vx, src.vy

    def export_payload(self) -> dict | None:
        """Serialise the recorded move for download.

        Everything here already exists — the plan with its probabilities, the
        team-level stats, and the tick-by-tick trajectory buffer. Returns None
        when no simulation is active (stopped sims have cleared the buffer)."""
        if not self.active:
            return None
        snap = self.snapshot()
        by_id = {p.id: p for p in self.players}
        return {
            "attackingTeam": self.attacking_team,
            "pitch": {"length": self.pitch_length, "width": self.pitch_width},
            "outcome": self.outcome,
            "sequenceProbability": self._sequence_probability,
            "steps": snap["steps"],
            "stats": snap["stats"],
            "roster": [
                {"id": p.id, "team": p.team, "number": p.number}
                for p in by_id.values()
            ],
            # One frame per tick: elapsed seconds, step index, ball [x, y],
            # players as [id, x, y]. Positions are pitch metres.
            "trajectory": list(self.trajectory),
        }

    def snapshot(self) -> dict:
        if not self.active:
            return {"active": False}
        stats = dict(self._stats)
        stats["longestPassM"] = round(stats["longestPassM"], 1)
        stats["topSpeedMps"] = round(stats["topSpeedMps"], 1)
        stats["xtGained"] = round(stats["xtGained"], 3)
        stats["forwardProgressM"] = round(stats["forwardProgressM"], 1)
        stats["elapsedS"] = round(stats["elapsedS"], 1)
        steps = []
        for s in self.steps:
            d = dict(s)
            # A step is seekable once its start checkpoint exists; the panel
            # uses this to know which steps can be jumped to.
            d["reached"] = s["index"] in self._checkpoints
            steps.append(d)
        return {
            "active": True,
            "playing": self.playing,
            "phase": self.phase,
            "rate": self.rate,
            "attackingTeam": self.attacking_team,
            "currentStep": self.step_index,
            "stepCount": len(self.steps),
            "steps": steps,
            "stats": stats,
            "ballOwnerNumber": self._carrier_number,
            "outcome": self.outcome,
            "sequenceProbability": self._sequence_probability,
            "trajectoryFrames": len(self.trajectory),
            # Highest step start recorded so far — the scrub-forward horizon,
            # which can exceed the restored forecast after a backward seek.
            "maxReachedStep": max(self._checkpoints, default=-1),
        }

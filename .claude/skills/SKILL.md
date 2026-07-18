---
name: tacticalcanvas
description: Project context for TacticalCanvas — a projected, finger-interactive soccer tactical board synced to match replay, built for a 36-hour hardware hackathon. Load this before any work on this repo: architecture, the agreed tech stack, the decisions that override the original plan, and scope guardrails. Triggers on anything touching the pitch renderer, projector/dashboard frontend, FastAPI state hub, WebSocket protocol, vision/touch detection, hand tracking, homography calibration, tactical analytics (pitch control, pass probability, defender shadows, suggested positions), Metrica/StatsBomb match data, or "what should I build next".
---

# TacticalCanvas

## Running it

```bash
python tc.py start     # cleans up any previous run, then serves; Ctrl+C stops
python tc.py stop      # from another terminal
python tc.py restart
python tc.py status    # what's running, and whether the camera is free
```

`tc.py` supervises `run.py` (which is still runnable directly). `start` self-heals from a crashed previous run, so a stale process should never need hunting by hand. Two things make that hold, and both are load-bearing — don't remove them:

- **`taskkill /T`** in `tc.py` kills the process *tree*. The vision worker is a child; killing only the parent used to strand it.
- **The worker watches its parent** (`multiprocessing.parent_process().is_alive()`) and exits on its own. A force-killed parent never runs its cleanup, so this is the only thing that survives a hard kill upstairs. Without it the worker holds the camera and the next launch dies with "Could not open camera".

**Not Docker.** A container on Windows can't reach the USB camera without serious pain and can't open the OpenCV preview window at all — it removes exactly what this rig depends on. Uvicorn is already in use; it's what `run.py` calls.

Dashboard `http://localhost:8000/dashboard` · projector `http://localhost:8000/projector` (drag to the projector, press **F**).
Env: `TC_CAMERA=1` (camera index, default 1) · `TC_NO_VISION=1` (mouse only, no camera) · `TC_PORT`.

Calibrate once: dashboard → *Calibration markers* → in the vision window press **c**, click the four magenta targets **1,2,3,4**, press **s**. Saved to `cache/calibration.json` and reloaded on boot.

**Prototype deviation from the stack below:** the frontend is currently plain HTML + vendored PixiJS served by FastAPI, not React+Vite — it needed to run on a projector the same hour it was written, and zero build tooling was the way to get there. The renderer and WS client are isolated modules (`web/shared/`), so the React port is mechanical. The architecture, protocol, and process split are already correct.

## Concept

A projector throws a top-down pitch onto a wall. A match replay plays on the coach's laptop, and the projected board shows the real player positions for that exact moment. The coach pauses, walks up to the wall, physically drags a player's circle to where they *should* have been, and draws passing/run lines with their finger. A dashboard shows the stats for that frame plus AI overlays — pitch control, defender shadows, pass lanes, suggested positions — so the coach can compare "what happened" against "what should have happened."

The pitch: existing tactical boards are either 2D screens the coach clicks alone, or magnetic whiteboards with no data behind them. This fuses both, adds an AI second opinion, and makes it physical enough that players actually watch.

**36-hour hackathon. Optimize for a working 90-second demo, not for a product.**

## Architecture

Three processes. Not five.

```
┌──────────── One React app (Vite) ─────────────┐
│  /dashboard              /projector           │
│  video + controls        PixiJS pitch         │
│  stats + overlays        fullscreen output    │
│  shared pitch components + WS client          │
└───────────────────┬───────────────────────────┘
                    │ WebSocket (typed JSON)
┌───────────────────▼───────────────────────────┐
│  FastAPI Core                                 │
│  authoritative state • sync • editing         │
│  analytics modules (in-process) • SQLite      │
└───────────────────┬───────────────────────────┘
                    │ multiprocessing.Queue
┌───────────────────▼───────────────────────────┐
│  Vision Worker (separate process, on purpose) │
│  OpenCV • MediaPipe • homography • touch      │
└───────────────────────────────────────────────┘
```

Every runtime boundary costs startup complexity, sync bugs, and debugging time we don't have. The vision worker is the one split worth paying for: camera capture and hand inference stall or eat CPU, and that must never block the WebSocket hub.

## Tech stack (authoritative)

| Area | Choice | Notes |
|---|---|---|
| Web app | React + TypeScript + Vite | One app, `/dashboard` and `/projector` routes |
| Pitch rendering | PixiJS (WebGL) | Both routes share the renderer components |
| UI | Tailwind + shadcn/ui | |
| Client state | Zustand | Local UI state only — never a second source of truth |
| Core server | FastAPI + Uvicorn + Pydantic | Analytics live inside this codebase |
| Transport | Native WebSocket + typed JSON | No Protobuf/FlatBuffers/custom binary at this data volume |
| Vision | OpenCV + MediaPipe Hand Landmarker (Tasks API) | Separate process |
| Analytics | NumPy + SciPy | Numba only after profiling proves a need |
| Demo data | Metrica synchronized tracking data (25 Hz) | **Not** interpolated StatsBomb event data |
| Persistence | SQLite | Scenarios, annotations, calibration |
| Tooling | pnpm workspace + uv + Ruff + pytest | One-command local startup |

Vite over Next.js because this is a fully local browser UI: no SSR, SEO, server components, or server actions. If the frontend already exists in Next.js and the team is productive in it, **keep Next.js** — but still merge dashboard and projector into one project. Do not burn hackathon hours on a framework swap.

## Decisions that override the original plan

An earlier version of this plan is floating around. Where they conflict, this document wins:

- **One frontend project, not two.** The original had a separate Next.js dashboard and a separate projector window app.
- **Analytics live inside FastAPI**, not as a separate "AI Service." Pitch control, pass scoring, compactness, and suggested positions are ordinary Python functions over current state. A model service only earns its keep with GPU inference, independent scaling, separately deployed models, or concurrent matches — none apply.
- **Metrica tracking data, not interpolated StatsBomb events.** Interpolation destroys velocities, and velocities are load-bearing for defender shadows, pitch control, compactness, off-ball runs, and the credibility of sync itself.
- **Commands in, authoritative state out** — not a Redux-ish store clients mutate freely.
- **IR pen / mouse is the primary input; finger touch is the stretch.** See below.
- **SQLite, no infrastructure.** No Redis, Postgres, Kafka, GraphQL, K8s, cloud deploy, or inference server.

## Synchronization

A bare frame index is not a sufficient sync model. Use clocked playback state:

```ts
type PlaybackState = {
  frameIndex: number;
  mediaTimeMs: number;
  playbackRate: number;
  playing: boolean;
  revision: number;
  serverTimestampMs: number;
};
```

Flow: the video callback reports the **displayed** media timestamp → dashboard sends it to FastAPI → FastAPI maps it to a tracking frame → FastAPI broadcasts a state revision → the projector interpolates between received positions for smooth rendering.

Use `HTMLVideoElement.requestVideoFrameCallback()`. Do not sync off `timeupdate`, timers, or React state — they report intent, not what's on screen.

In edit mode the server freezes the source frame and increments `revision` on each accepted edit.

## WebSocket protocol

Clients send commands; the server owns state and broadcasts it.

Commands: `SET_PLAYBACK_TIME`, `ENTER_EDIT_MODE`, `DRAG_PLAYER_START`, `DRAG_PLAYER_MOVE`, `DRAG_PLAYER_END`, `ADD_PASS_LINE`, `ADD_RUN_LINE`, `TOGGLE_OVERLAY`, `RESET_SCENARIO`

Server broadcasts: `STATE_SNAPSHOT`, `STATE_PATCH`, `ERROR`, `LATENCY_STATUS`

Every message carries: `protocolVersion`, `scenarioId`, `clientId`, `sequenceNumber`, `timestamp`. This is what prevents stale drag events, reconnect corruption, and the projector rendering a different scenario than the dashboard.

## Vision worker

Talks to FastAPI over `multiprocessing.Queue` (a localhost WebSocket later if stronger isolation is ever needed). It emits **semantic events, never raw landmarks**:

```json
{ "type": "touch_move", "boardX": 0.63, "boardY": 0.41, "confidence": 0.92, "handId": 0 }
```

Homography calibration runs once at startup: project four corners, click them in the camera feed, map camera pixels → board coordinates. MediaPipe gives 21 landmarks per hand at ~30fps; index fingertip is landmark 8.

**Grab is a PINCH, not a touch.** Touch detection needs depth; the rig is monocular. Thumb-tip↔index-tip distance, normalised by palm length (wrist↔middle knuckle), is unambiguous from any angle and needs no depth. The fingertip gives the cursor *position*; the pinch gives the *verb*. This is what makes monocular viable, and it means a depth camera is an upgrade, not a prerequisite.

**Camera capture: force nothing.** Measured on this rig — forcing 720p+MJPG gives **4.8fps** (the format request is silently ignored, then the resolution exceeds USB bandwidth). Letting it negotiate gives 640x480 @ **~30fps** through MediaPipe. Keep the `cv2.CAP_DSHOW` backend hint though: letting the backend default picks MSMF, which takes **~83s** to open the camera versus ~0.5s, for identical throughput. Do not set resolution, FOURCC, FPS, exposure, or contrast. This is measured, not folklore — re-measure before overriding it.

Current state: [vision/worker.py](../../../vision/worker.py) is the real worker (calibration, pinch FSM with hysteresis, smoothing, jump rejection, events onto a `multiprocessing.Queue`). [hand_detection.py](../../../hand_detection.py) is the original standalone demo, kept only as a camera scratchpad — it is superseded.

## Touch is the riskiest thing in the project

Distinguishing pointing-near-wall from touching-wall from dragging — while the coach's body occludes the projector and their other hand wanders — is where this demo dies. So:

1. **IR pen or tracked stylus** — primary, reliable path
2. **Mouse / tablet** — always-works fallback
3. **Finger-only** — experimental mode, demoed if it's behaving

A depth camera (RealSense D435, Azure Kinect) makes touch far more reliable — fingertip within ~2cm of the wall plane counts as a touch. Without depth, this is genuinely hard.

Required regardless: touch-start dwell threshold, separate start/release thresholds (hysteresis), coordinate smoothing, max jump distance, drag ownership per hand, a calibration confidence indicator, and a keyboard shortcut that kills touch instantly. **These improve the demo more than another AI overlay does.**

## Data preprocessing

Never parse source files live during the demo. Preprocess to one array file:

```bash
python -m tools.prepare_match --tracking data/home.csv --events data/events.json --output cache/demo_match.npz
```

Normalized arrays: `timestamps [frames]`, `positions [frames, players, 2]`, `velocities [frames, players, 2]`, `ball_positions [frames, 2]`, `possession_team [frames]`.

## Analytics

Ship two or three overlays, not all of them. Priority order — Voronoi pitch control first (fastest to ship), then defender shadows, then pass-probability heatmap.

```
server/analytics/
  pitch_control.py        # Voronoi approximation; Spearman model only if time allows
  pass_probability.py     # logistic regression; hand-tuned weights are fine for a demo
  compactness.py          # line-to-line distances + team width
  suggested_positions.py  # gradient ascent on team pitch control within a movement budget
```

**Rendering rate ≠ analytics rate.** Render at 60fps; recompute tactics far less often:

- Player movement — display refresh rate
- Compactness/width — immediate (cheap)
- Voronoi pitch control — 10–15 Hz
- Detailed pitch control — after the drag settles
- Suggested positions — on demand
- Static replay overlays — precompute before the demo

Start the control grid at `50 × 32` and upscale in PixiJS.

## Debug overlay — build this before more features

A hidden diagnostic view showing: video timestamp, tracking frame, WebSocket latency, state revision, renderer FPS, vision FPS, touch confidence, calibration error, analytics compute time, last reconnect. This is how you find the source of lag during setup instead of guessing.

Also log vision events and state commands to JSONL. Replaying an interaction without the camera attached is worth hours during development and rehearsal.

## Scope guardrails

**Do the must-changes:** one frontend project · analytics inside FastAPI · versioned WS command/state protocol · sync on real displayed video-frame timestamps · Metrica tracking data · reliable IR-pen/mouse fallback · latency + calibration diagnostics.

**Only when convenient:** Vite migration · Zustand · SQLite · multiprocessing queues · TS types generated from Pydantic schemas.

**Never (this weekend):** microservices · cloud hosting · Redis/Kafka · deep-learning tactical models · multiple data providers · auth · mobile apps.

If asked to add something on the "never" list, say so and propose the boring local equivalent.

## Timeline

- **0–4** Projector + camera mounted; homography calibration end-to-end. *Riskiest hours in the project — if calibration isn't working by hour 4, pivot to mouse/tablet control and keep the projected board.*
- **4–10** WebSocket hub, projector drawing a static frame, dashboard playing video, state syncing.
- **10–18** Fingertip → board coordinate → drag a circle. Pass-line drawing. Pause/resume.
- **18–26** AI overlays (Voronoi first), stats sidebar.
- **26–32** Polish + one flashy stretch feature (probably ghost circles for suggested positions). Fix demo data. Rehearse.
- **32–36** Buffer + record a demo video. Something *will* break; the video is the insurance.

## The demo (design backwards from this)

90 seconds: load a goal-conceded moment → play the replay → pause at the defensive breakdown → drag the right-back into position → draw the covering run → toggle pitch control to show the gap is now closed → toggle the AI suggestion to show a *different* fix the model prefers.

That narrative sells the project far better than a feature tour. When a decision is ambiguous, pick whatever makes those 90 seconds more likely to survive contact with a judge.

The strongest stack here isn't the most advanced one. It's fewer processes, one authoritative state machine, real tracking data, deterministic replay, and a guaranteed input fallback.
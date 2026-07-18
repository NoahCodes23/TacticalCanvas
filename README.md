# TacticalCanvas

## Low-latency defaults

Start TacticalCanvas with:

```powershell
.\.venv\Scripts\python.exe tc.py start
```

The vision worker requests 640x480 MJPEG at 60 FPS, keeps only the latest
camera frame, runs MediaPipe asynchronously, disables the OpenCV preview, and
pushes vision changes to browsers immediately. If the webcam only supports 30
FPS, it falls back to the negotiated rate without accumulating stale frames.

Optional overrides:

```powershell
$env:TC_CAMERA_FPS="30"       # force a lower camera request
$env:TC_DETECT_WIDTH="480"    # MediaPipe preprocessing width; 0 disables resize
$env:TC_VISION_PREVIEW="1"    # enable the diagnostic OpenCV window
```

After changing the camera resolution or physically moving the camera/projector,
open the projector view, then click **Calibrate field** under Setup in the
dashboard. Six ArUco markers are rendered inside the known 105 x 68 field and
the running vision worker automatically collects them. Keep the camera still
until the dashboard reports success; no OpenCV window or corner clicking is
required. The solver uses all visible marker corners to learn a residual field
warp after perspective correction, which compensates for smooth webcam and
projector lens error on the playing surface. Recalibrate once after upgrading
to this version. `python tc.py calibrate` now points to this web flow.

Hand control uses a thumb/index pinch. The pointer is the midpoint of the two
fingertips: close the pinch to pick up the nearest piece, move while pinched,
and open the pinch to release it. Release requires a short sustained opening,
brief tracking gaps do not immediately drop a held piece, and a pinch can snap
to the nearest player from roughly one piece-width away.

For the projector itself, enable Game/Low Latency mode and disable motion
interpolation, keystone correction, noise reduction, and other image processing
where the hardware permits. Prefer a direct HDMI connection and the highest
native refresh rate shared by the PC and projector.

### Projector rendering quality

The dashboard and projector use **sharp** rendering by default: antialiasing is
enabled, the Pixi canvas is supersampled at least 1.5x (up to 2x based on display
DPI), and text textures remain at 2x. This makes small tactical labels readable
without changing pitch coordinates or camera calibration. The current render
scale appears in dashboard diagnostics.

If a low-power GPU cannot hold the desired frame rate, reload either page with
the performance preset to restore the original 1x canvas:

```text
http://localhost:8000/projector?quality=performance
```

For comparison and tuning, `?renderScale=1.25`, `1.5`, or `2` overrides the
canvas scale while retaining the selected quality preset. Values are clamped to
the safe 1x-2x range.

The projector field defaults to 82% of the available window so the entire
interaction area stays away from the camera's least reliable outer edge. For
testing, `?fieldFill=0.75` through `?fieldFill=0.96` changes that centered field
size without changing its 105 x 68 coordinates or invalidating calibration.

## Experimental tactical analysis

The dashboard has three independent **Experimental AI** switches. They are all
off by default, are reset when a new match is loaded, and add no analytics work
to the normal frame loop until at least one is enabled:

- **AI pass recommendations** draws every teammate option and highlights the
  top three with completion probability and expected-value score.
- **Technical indicator HUD** shows live context, team-shape metrics, and the
  feature breakdown behind the three highest-ranked next actions.
- **Receiver position targets** searches reachable nearby positions for the
  top receivers and draws a suggested off-ball move when it improves the score.

For a presentation, use the three numbered demo buttons:

1. **Freeze + rank passes** pauses on a frame, opens the indicator inspector,
   and colours the top three receiver choices.
2. **Live team shape** runs the technical indicators over the moving replay.
3. **Movement targets** pauses and enables the full pass/receiver-position demo.

The HUD includes a colour legend. Every metric row and recommended-pass card is
clickable (and keyboard accessible); the inspector explains what it measures,
shows the calculation, and tells you how to interpret high and low values.

The technical HUD includes possession and ball carrier, attack phase/channel,
arrival-time control at the ball, pressure and transition indices, viable and
progressive pass counts, channel overload, team tempo, field tilt, defensive
line height, attacking-line height, width, depth, convex-hull area, average
speed, high-intensity runs, sprints, ball pressure, opponent spacing, average
xT, and a formation-shape score. Each pass also exposes distance, forward
progress, passer/receiver pressure, passing-lane blockers and clearance,
defenders bypassed, destination control, xT gain, turnover cost, offside state,
completion probability, risk, and a plain-language explanation.

This is an explainable heuristic baseline, not a trained or validated model.
Its scoring implementation lives entirely in
`server/analytics/experimental.py`, so a learned completion model or xT grid can
replace it later without changing the WebSocket data or pitch renderer.

## LLM coach advice

When playback is paused, the dashboard shows **Get LLM coach advice** directly
under the pause/resume control. One request captures five snapshots spaced 400
ms apart (`-1.6s`, `-1.2s`, `-0.8s`, `-0.4s`, and now), plus the three most
recent recorded match events. It calculates the full experimental analysis for
each snapshot, collapses each to a compact **briefing** (see
`server/analytics/briefing.py`), and asks the coach model to translate that tiny
lead-in into concise, player-specific instructions. Each briefing carries a
`facts` list, and the prompt permits the model to state **only** numbers that
appear there — the whitelist that keeps it from inventing stats. The prompt also
limits the response to 4-6 short sentences under 100 words, with no headings or
metric dump.

Copy `.env.example` to `.env` and add your OpenAI key:

```dotenv
OPENAI_API_KEY=your-key-here
# TC_COACH_MODEL=gpt-4.1-mini   # optional; this is the default
```

The backend is provider-agnostic: point `TC_COACH_BASE_URL` (and the matching
key) at any OpenAI-compatible endpoint — DeepSeek, Groq, Together, Ollama, or
OpenRouter — to switch models without code changes. When the base URL is
OpenRouter, requests keep its Zero Data Retention routing (`provider.zdr=true`)
and attribution headers automatically.

Restart TacticalCanvas after changing `.env`. The browser calls the local
`/api/coach-advice` endpoint; only the Python server reads the key. Advice is
cached for the same match frame and state revision to avoid duplicate paid
requests. The LLM receives model data, not video images, and is explicitly told
that the underlying indicators are experimental heuristics rather than
validated predictions.

### Git safety

Calibration/latency and experimental analytics are intentionally separate
commits. To remove only the experiment after it is committed:

```powershell
git log --oneline -2
git revert <experimental-analytics-commit>
```

All existing calibration and latency work remains in the earlier commit.

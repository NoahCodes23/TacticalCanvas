import asyncio
import contextlib
import multiprocessing
import os
import queue as queue_mod
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import ingest, match_data
from .coach import CoachServiceError, request_coach_advice
from .protocol import PROTOCOL_VERSION, Envelope, server_message
from .state import AppState, now_ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
load_dotenv(os.path.join(ROOT, ".env"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cap uploads at 2 GB. A full-match broadcast at H.264 is ~1.2 GB; anything
# larger is almost certainly a mistake and would eat the FastAPI worker's
# memory as we stream it to disk.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}

TICK_HZ = 60
BROADCAST_EVERY = 2  # -> 30Hz snapshots; the clients interpolate between them

state = AppState()
clients: set[WebSocket] = set()
vision_queue: "multiprocessing.Queue | None" = None
vision_proc: "multiprocessing.Process | None" = None
coach_request_lock = asyncio.Lock()
coach_advice_cache: dict[tuple, dict] = {}


async def broadcast(message: dict) -> None:
    dead = []
    for ws in list(clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def snapshot_message() -> dict:
    return server_message(
        "STATE_SNAPSHOT",
        state.snapshot(),
        state.scenario_id,
        state.next_sequence(),
        now_ms(),
    )


def vision_message() -> dict:
    return server_message(
        "VISION_UPDATE",
        state.vision_snapshot(),
        state.scenario_id,
        state.next_sequence(),
        now_ms(),
    )


async def tick_loop() -> None:
    dt = 1.0 / TICK_HZ
    n = 0
    last = time.monotonic()
    while True:
        await asyncio.sleep(dt)
        now = time.monotonic()
        elapsed, last = now - last, now
        state.tick(elapsed)
        state.prune_cursors()
        n += 1
        if n % BROADCAST_EVERY == 0 and clients:
            await broadcast(snapshot_message())

async def vision_loop() -> None:
    while True:
        if vision_queue is None:
            await asyncio.sleep(0.02)
            continue
        drained = 0
        changed = False
        while drained < 64:
            try:
                evt = vision_queue.get_nowait()
            except (queue_mod.Empty, InterruptedError):
                break
            except Exception:
                break
            state.handle_vision_event(evt)
            changed = True
            drained += 1
        if changed and clients:
            # Vision controls are latency-sensitive; don't wait for the next
            # scheduled match snapshot. One broadcast covers this drained batch.
            await broadcast(vision_message())
        await asyncio.sleep(0.001)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global vision_queue, vision_proc

    if os.environ.get("TC_NO_VISION") == "1":
        print("[server] TC_NO_VISION=1 -- vision worker disabled (mouse input only)")
    else:
        camera = int(os.environ.get("TC_CAMERA", "1"))
        show_preview = os.environ.get("TC_VISION_PREVIEW", "0") == "1"
        ctx = multiprocessing.get_context("spawn")
        vision_queue = ctx.Queue(maxsize=32)
        from vision.worker import run as vision_run

        vision_proc = ctx.Process(
            target=vision_run,
            args=(vision_queue, camera, show_preview),
            daemon=True,
        )
        vision_proc.start()
        print(f"[server] vision worker started (pid {vision_proc.pid}, camera {camera})")

    tasks = [asyncio.create_task(tick_loop()), asyncio.create_task(vision_loop())]
    print("[server] dashboard  -> http://localhost:8000/dashboard")
    print("[server] projector  -> http://localhost:8000/projector")
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        if vision_proc is not None and vision_proc.is_alive():
            vision_proc.terminate()
            vision_proc.join(timeout=2)


app = FastAPI(title="TacticalCanvas", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "landing.html"))


@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(WEB_DIR, "dashboard.html"))


@app.get("/projector")
async def projector():
    return FileResponse(os.path.join(WEB_DIR, "projector.html"))


@app.post("/api/coach-advice")
async def coach_advice():
    """Analyze five recent paused snapshots without exposing the API key."""
    if state.playing:
        raise HTTPException(409, "Pause the match before requesting coach advice.")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured in .env.")

    model = os.environ.get("OPENROUTER_MODEL", "").strip() or None
    cache_key = (state.match_id, state.frame_index, state.revision, model)
    cached = coach_advice_cache.get(cache_key)
    if cached is not None:
        return JSONResponse({**cached, "cached": True})

    async with coach_request_lock:
        cached = coach_advice_cache.get(cache_key)
        if cached is not None:
            return JSONResponse({**cached, "cached": True})
        if state.playing:
            raise HTTPException(409, "The match resumed before analysis started.")

        raw_frames = state.coach_frame_inputs()
        frames = await asyncio.to_thread(state.analyze_coach_frames, raw_frames)
        recent_events = match_data.recent_events(state.media_time_ms / 1000.0, 3)
        try:
            result = await request_coach_advice(
                frames,
                state.match_label,
                api_key=api_key,
                model=model,
                recent_events=recent_events,
            )
        except CoachServiceError as error:
            raise HTTPException(502, str(error)) from error

        response = {
            **result,
            "cached": False,
            "frameCount": len(frames),
            "frameIndex": state.frame_index,
            "revision": state.revision,
            "matchId": state.match_id,
        }
        coach_advice_cache[cache_key] = response
        # Bound memory while keeping recent paid responses reusable.
        while len(coach_advice_cache) > 32:
            coach_advice_cache.pop(next(iter(coach_advice_cache)))
        return JSONResponse(response)


# --------------------------------------------------------------------------- #
# uploads: raw-body PUT so we don't depend on python-multipart
# --------------------------------------------------------------------------- #
def _safe_upload_name(raw: str) -> str:
    """Turn an arbitrary client-supplied filename into something safe on disk.
    Strips paths, collapses odd characters, keeps a video extension we allow."""
    base = os.path.basename(raw or "").strip() or "video"
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        ext = ".mp4"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    return f"{stem[:80]}{ext}"


def _dedupe_name(name: str) -> str:
    """If `name` already exists in uploads/, append -1, -2, ... until it doesn't."""
    stem, ext = os.path.splitext(name)
    i = 0
    candidate = name
    while os.path.exists(os.path.join(UPLOAD_DIR, candidate)):
        i += 1
        candidate = f"{stem}-{i}{ext}"
    return candidate


@app.put("/upload")
async def upload(request: Request, name: str = "video.mp4"):
    """Streaming PUT: writes the raw request body straight to data/uploads/.
    The client supplies the display filename via `?name=`; we sanitise it and
    de-duplicate so parallel uploads don't clobber each other."""
    safe = _dedupe_name(_safe_upload_name(name))
    path = os.path.join(UPLOAD_DIR, safe)
    total = 0
    try:
        with open(path, "wb") as f:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    f.close()
                    os.unlink(path)
                    raise HTTPException(413, "file too large (>2 GB)")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        # Clean up the half-written file so the uploads list stays honest.
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise HTTPException(500, f"upload failed: {e}") from e
    return JSONResponse({"name": safe, "size": total, "url": f"/uploads/{safe}"})


@app.get("/api/uploads")
async def list_uploads():
    items = []
    for entry in sorted(os.listdir(UPLOAD_DIR)):
        p = os.path.join(UPLOAD_DIR, entry)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(entry)[1].lower() not in ALLOWED_VIDEO_EXTS:
            continue
        items.append({"name": entry, "size": os.path.getsize(p)})
    # Newest first so a fresh upload floats to the top of the landing list.
    items.sort(key=lambda it: os.path.getmtime(os.path.join(UPLOAD_DIR, it["name"])),
               reverse=True)
    return {"items": items}


@app.get("/api/matches")
async def list_prepared_matches():
    return {"items": match_data.list_matches()}


# --------------------------------------------------------------------------- #
# scenario ingest: run tools/prepare_video against an uploaded clip
# --------------------------------------------------------------------------- #
@app.post("/api/scenarios/ingest")
async def start_ingest(request: Request):
    """Body: {"videoName": "<file in data/uploads/>", "label": "<optional>"}.
    Returns the job id; poll GET /api/scenarios/ingest/{id} for progress."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "expected JSON body")
    video_name = body.get("videoName")
    if not isinstance(video_name, str) or not video_name.strip():
        raise HTTPException(400, "videoName required")
    if video_name != os.path.basename(video_name) or video_name.startswith("."):
        raise HTTPException(400, "bad videoName")
    label = body.get("label") if isinstance(body.get("label"), str) else None
    try:
        job = ingest.create_job(video_name, label=label)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    asyncio.create_task(ingest.run(job, label=label))
    return JSONResponse(job.to_dict())


@app.get("/api/scenarios/ingest")
async def list_ingest_jobs():
    return {"items": ingest.list_jobs()}


@app.get("/api/scenarios/ingest/{job_id}")
async def get_ingest_job(job_id: str):
    job = ingest.get_job(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.delete("/api/scenarios/ingest/{job_id}")
async def cancel_ingest_job(job_id: str):
    job = ingest.get_job(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    ok = await ingest.cancel(job)
    if not ok:
        raise HTTPException(409, f"job is {job.status}, cannot cancel")
    return job.to_dict()


@app.get("/uploads/{name}")
async def get_upload(name: str):
    # Reject anything that isn't a plain filename in our upload dir; blocks
    # ../ traversal without needing to normalise.
    if name != os.path.basename(name) or name.startswith("."):
        raise HTTPException(400, "bad filename")
    path = os.path.join(UPLOAD_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    # FileResponse handles Range requests, which the <video> element uses to
    # seek without downloading the whole clip.
    return FileResponse(path)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    await ws.send_json(snapshot_message())
    try:
        while True:
            raw = await ws.receive_json()
            try:
                env = Envelope(**raw)
            except Exception as e:
                await ws.send_json(server_message(
                    "ERROR", {"reason": f"bad envelope: {e}"},
                    state.scenario_id, state.next_sequence(), now_ms()))
                continue

            if env.protocolVersion != PROTOCOL_VERSION:
                await ws.send_json(server_message(
                    "ERROR",
                    {"reason": f"protocol {env.protocolVersion} != {PROTOCOL_VERSION}"},
                    state.scenario_id, state.next_sequence(), now_ms()))
                continue

            await handle_command(ws, env)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[server] websocket error: {e}")
    finally:
        clients.discard(ws)
        state.drag_end(f"client:{id(ws)}")

async def handle_command(ws: WebSocket, env: Envelope) -> None:
    p = env.payload
    t = env.type
    owner = f"client:{id(ws)}"

    if t == "PING":
        await ws.send_json(server_message(
            "PONG", {"t": p.get("t")}, state.scenario_id,
            state.next_sequence(), now_ms()))
        return

    if t == "SET_PLAYING":
        state.set_playing(bool(p.get("playing", True)))
    elif t == "SET_PLAYBACK_TIME":
        # Fires many times a second from the video's frame callback. We still
        # broadcast a snapshot each call (matches every other command) --
        # snapshots are already gated by BROADCAST_EVERY in tick_loop, but here
        # we get one per command which is what keeps overlays glued to the
        # video frame the coach is actually looking at.
        try:
            mt = float(p.get("mediaTimeMs", 0.0))
        except (TypeError, ValueError):
            await ws.send_json(server_message(
                "ERROR", {"reason": f"bad mediaTimeMs {p.get('mediaTimeMs')!r}"},
                state.scenario_id, state.next_sequence(), now_ms()))
            return
        playing_val = p.get("playing")
        state.set_playback_time(mt, playing_val if isinstance(playing_val, bool) else None)
    elif t == "ENTER_EDIT_MODE":
        state.enter_edit_mode()
    elif t == "EXIT_EDIT_MODE":
        state.exit_edit_mode()
    elif t == "RESET_SCENARIO":
        state.reset_scenario()
    elif t == "LOAD_MATCH":
        match_id = p.get("matchId")
        if not isinstance(match_id, str) or not state.load_match(match_id):
            await ws.send_json(server_message(
                "ERROR", {"reason": f"could not load match {match_id!r}"},
                state.scenario_id, state.next_sequence(), now_ms()))
            return
    elif t == "TOGGLE_CALIBRATION":
        state.toggle_calibration()
    elif t == "TOGGLE_OFFSIDE":
        state.toggle_offside()
    elif t == "TOGGLE_COMPACTNESS":
        state.toggle_compactness()
    elif t == "TOGGLE_SHADOWS":
        state.toggle_shadows()
    elif t == "TOGGLE_PITCH_CONTROL":
        state.toggle_pitch_control()
    elif t == "TOGGLE_FORMATION":
        state.toggle_formation()
    elif t == "TOGGLE_SUGGESTED":
        state.toggle_suggested()
    elif t == "SET_EXPERIMENT":
        name = p.get("name")
        enabled = p.get("enabled") if "enabled" in p else None
        valid_enabled = enabled is None or isinstance(enabled, bool)
        if not isinstance(name, str) or not valid_enabled or not state.set_experiment(name, enabled):
            await ws.send_json(server_message(
                "ERROR", {"reason": f"invalid experiment setting {name!r}={enabled!r}"},
                state.scenario_id, state.next_sequence(), now_ms()))
            return
    elif t == "SET_SHADOW_SECONDS":
        try:
            state.set_shadow_seconds(float(p.get("seconds", 2.0)))
        except (TypeError, ValueError):
            await ws.send_json(server_message(
                "ERROR", {"reason": f"bad seconds {p.get('seconds')!r}"},
                state.scenario_id, state.next_sequence(), now_ms()))
            return
    elif t == "DRAG_PLAYER_START":
        state.drag_start(p["playerId"], p["boardX"], p["boardY"], owner)
    elif t == "DRAG_PLAYER_MOVE":
        state.drag_move(p["playerId"], p["boardX"], p["boardY"], owner)
    elif t == "DRAG_PLAYER_END":
        state.drag_end(owner)
    else:
        await ws.send_json(server_message(
            "ERROR", {"reason": f"unknown command {t}"},
            state.scenario_id, state.next_sequence(), now_ms()))
        return

    await broadcast(snapshot_message())


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

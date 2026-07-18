import asyncio
import contextlib
import multiprocessing
import os
import queue as queue_mod
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .protocol import PROTOCOL_VERSION, Envelope, server_message
from .state import AppState, now_ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")

TICK_HZ = 60
BROADCAST_EVERY = 2  # -> 30Hz snapshots; the clients interpolate between them

state = AppState()
clients: set[WebSocket] = set()
vision_queue: "multiprocessing.Queue | None" = None
vision_proc: "multiprocessing.Process | None" = None


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
            await asyncio.sleep(0.1)
            continue
        drained = 0
        while drained < 64:
            try:
                evt = vision_queue.get_nowait()
            except (queue_mod.Empty, InterruptedError):
                break
            except Exception:
                break
            state.handle_vision_event(evt)
            drained += 1
        await asyncio.sleep(0.004)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global vision_queue, vision_proc

    if os.environ.get("TC_NO_VISION") == "1":
        print("[server] TC_NO_VISION=1 -- vision worker disabled (mouse input only)")
    else:
        camera = int(os.environ.get("TC_CAMERA", "1"))
        ctx = multiprocessing.get_context("spawn")
        vision_queue = ctx.Queue(maxsize=256)
        from vision.worker import run as vision_run

        vision_proc = ctx.Process(
            target=vision_run, args=(vision_queue, camera, True), daemon=True
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
    return RedirectResponse("/dashboard")


@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(WEB_DIR, "dashboard.html"))


@app.get("/projector")
async def projector():
    return FileResponse(os.path.join(WEB_DIR, "projector.html"))


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

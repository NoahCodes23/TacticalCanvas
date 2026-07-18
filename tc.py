"""TacticalCanvas process and calibration commands.

Usage: python tc.py [start|stop|restart|status|calibrate] [--voice]

--voice also launches voice_agent.py once the board is up (or TC_VOICE=1).
Off by default: it holds the microphone and spends OpenRouter credit, which
you do not want on every ordinary start.
"""

import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("TC_PORT", "8000"))
CALIB_PATH = os.path.join(ROOT, "cache", "field-calibration.json")

sys.stdout.reconfigure(line_buffering=True)

def port_busy(port: int = PORT) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0

def pids_on_port(port: int = PORT) -> set[int]:
    """Whoever is LISTENING on the port. netstat exists on every Windows."""
    out = subprocess.run(
        ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
    ).stdout
    pids = set()
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0] == "TCP" and f[1].endswith(f":{port}") and f[3] == "LISTENING":
            try:
                pids.add(int(f[4]))
            except ValueError:
                pass
    return pids

def process_name(pid: int) -> str | None:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True,
    ).stdout.strip()
    if out.startswith('"'):
        return out.split('","')[0].strip('"')

    # Some Windows configurations return no tasklist row for a process that
    # Get-Process can still inspect. Keep restart/stop from rejecting our own
    # Python server as an unknown foreign process.
    fallback = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).ProcessName",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return fallback or None


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    
def stop_stale(quiet: bool = False) -> bool:
    """Kill a previous run. Returns False if something we don't own has the port."""
    found = False
    for pid in pids_on_port():
        name = process_name(pid) or "?"
        if "python" not in name.lower():
            print(f"  port {PORT} is held by {name} (pid {pid}) -- that isn't us.")
            print(f"  Not touching it. Stop it yourself, or set TC_PORT to something else.")
            return False
        if not quiet:
            print(f"  stopping previous run ({name}, pid {pid})")
        kill_tree(pid)
        found = True
    if found:
        for _ in range(20):  # give the port a moment to actually free up
            if not port_busy():
                break
            time.sleep(0.25)
    return True

def camera_free() -> bool | None:
    """None = couldn't tell (opencv missing)."""
    try:
        import cv2
    except ImportError:
        return None
    idx = int(os.environ.get("TC_CAMERA", "1"))
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    cap.release()
    return ok

def cmd_start(voice: bool = False) -> int:
    if not stop_stale():
        return 1

    if os.path.exists(CALIB_PATH):
        print(f"calibration: loading {CALIB_PATH}")
    else:
        print("calibration: not found (use Calibrate field in the dashboard)")
    print(f"starting TacticalCanvas on port {PORT} ...")
    proc = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT)

    for _ in range(60):  # wait for it to actually bind before printing links
        if port_busy():
            break
        if proc.poll() is not None:
            print("server exited during startup -- see the error above")
            return proc.returncode or 1
        time.sleep(0.25)

    # The voice agent's MCP tools drive the board over its WebSocket, so it can
    # only start once the port is actually bound -- hence launching it here and
    # not alongside run.py. Its failure is never the board's failure: a missing
    # API key or a mic already in use must not take the projector down mid-demo.
    voice_proc = None
    if voice:
        print("starting voice agent ...")
        voice_proc = subprocess.Popen([sys.executable, "voice_agent.py"], cwd=ROOT)

    print(f"\n  dashboard  http://localhost:{PORT}/dashboard")
    print(f"  projector  http://localhost:{PORT}/projector")
    if voice:
        print("  voice      talk into the mic (transcript is mixed into this log)")
    print("\n  Ctrl+C to stop\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nstopping ...")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_tree(proc.pid)
        print("stopped")
    finally:
        # Ctrl+C in a shared console already reaches the voice agent, but an
        # exiting server must take it down too -- a stranded agent keeps the mic
        # and talks to a board that is no longer there.
        if voice_proc and voice_proc.poll() is None:
            kill_tree(voice_proc.pid)
    return 0

def cmd_stop() -> int:
    if not port_busy():
        print("not running")
        return 0
    ok = stop_stale()
    print("stopped" if ok else "could not stop it")
    return 0 if ok else 1


def cmd_restart(voice: bool = False) -> int:
    cmd_stop()
    return cmd_start(voice)


def cmd_calibrate() -> int:
    """Point users to the in-dashboard calibration owned by the vision worker."""
    if not port_busy():
        print("TacticalCanvas is not running. Start it with 'python tc.py start'.")
        return 1
    print(f"open http://localhost:{PORT}/dashboard")
    print("under Setup, click 'Calibrate field' and follow the live marker status")
    return 0


def cmd_status() -> int:
    running = port_busy()
    print(f"server    : {'running' if running else 'stopped'} (port {PORT})")
    for pid in pids_on_port():
        print(f"            pid {pid} ({process_name(pid)})")
    if running:
        print("camera    : held by the running vision worker (expected)")
    else:
        free = camera_free()
        print(f"camera    : {'free' if free else 'BUSY -- something else has it' if free is False else 'unknown'}")
    print(f"calibration: {'saved' if os.path.exists(CALIB_PATH) else 'missing'} ({CALIB_PATH})")
    return 0


COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "restart": cmd_restart,
    "status": cmd_status,
    "calibrate": cmd_calibrate,
}

# Only start and restart take --voice; the rest ignore it.
TAKES_VOICE = {"start", "restart"}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--voice"]
    voice = "--voice" in sys.argv[1:] or os.environ.get("TC_VOICE") == "1"
    action = args[0] if args else "start"
    if action not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(COMMANDS[action](voice) if action in TAKES_VOICE else COMMANDS[action]())

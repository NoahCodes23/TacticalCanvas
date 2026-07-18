import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("TC_PORT", "8000"))

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
    return out.split('","')[0].strip('"') if out.startswith('"') else None


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

def cmd_start() -> int:
    if not stop_stale():
        return 1

    print(f"starting TacticalCanvas on port {PORT} ...")
    proc = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT)

    for _ in range(60):  # wait for it to actually bind before printing links
        if port_busy():
            break
        if proc.poll() is not None:
            print("server exited during startup -- see the error above")
            return proc.returncode or 1
        time.sleep(0.25)

    print(f"\n  dashboard  http://localhost:{PORT}/dashboard")
    print(f"  projector  http://localhost:{PORT}/projector")
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
    return 0

def cmd_stop() -> int:
    if not port_busy():
        print("not running")
        return 0
    ok = stop_stale()
    print("stopped" if ok else "could not stop it")
    return 0 if ok else 1


def cmd_restart() -> int:
    cmd_stop()
    return cmd_start()


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
    return 0


COMMANDS = {"start": cmd_start, "stop": cmd_stop, "restart": cmd_restart, "status": cmd_status}

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "start"
    if action not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(COMMANDS[action]())

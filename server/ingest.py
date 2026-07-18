"""Background jobs that turn an uploaded video into a cache/match_N.npz.

We shell out to ``python -m tools.prepare_video`` rather than importing it in
process: the pipeline loads YOLO into memory and would eat the FastAPI worker's
RSS + hold GIL time on every WebSocket tick. A subprocess also means a bad clip
(or a user hitting cancel) can't crash the server.

State is in-memory only. If the server restarts mid-run, orphaned subprocesses
are reaped by the OS and their jobs are simply forgotten -- acceptable for a
hackathon demo, and matches how tc.py already handles restarts.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / "cache"
_UPLOAD_DIR = _ROOT / "data" / "uploads"
_WEIGHTS_DIR = _ROOT / "weights"

# Where the three required assets live. Configurable via env so a demo box can
# point at pretrained weights sitting anywhere on disk without editing code.
DEFAULT_PLAYER_WEIGHTS  = Path(os.environ.get("TC_PLAYER_WEIGHTS",  _WEIGHTS_DIR / "players.pt"))
DEFAULT_PITCH_WEIGHTS   = Path(os.environ.get("TC_PITCH_WEIGHTS",   _WEIGHTS_DIR / "pitch.pt"))
DEFAULT_PITCH_TEMPLATE  = Path(os.environ.get("TC_PITCH_TEMPLATE",  _WEIGHTS_DIR / "pitch_template.json"))

LOG_LINES_KEPT = 400   # ring-buffer per job; the UI shows the tail


@dataclass
class Job:
    id: str
    video_name: str
    match_id: str                  # e.g. "match_4"
    out_path: Path
    status: str = "pending"        # pending | running | success | failed | cancelled
    started_at: float = 0.0
    finished_at: float = 0.0
    exit_code: int | None = None
    error: str | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES_KEPT))
    _proc: asyncio.subprocess.Process | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "videoName": self.video_name,
            "matchId": self.match_id,
            "status": self.status,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "exitCode": self.exit_code,
            "error": self.error,
            "log": list(self.log),
        }


_jobs: dict[str, Job] = {}


def _next_match_id() -> str:
    """Pick the lowest match_N.npz slot that isn't taken or reserved by a job."""
    taken: set[int] = set()
    if _CACHE_DIR.is_dir():
        for p in _CACHE_DIR.glob("match_*.npz"):
            try:
                taken.add(int(p.stem.split("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    for j in _jobs.values():
        if j.status in ("pending", "running"):
            try:
                taken.add(int(j.match_id.split("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    n = 1
    while n in taken:
        n += 1
    return f"match_{n}"


def _assets_ok() -> tuple[bool, str]:
    for label, p in [
        ("player weights",  DEFAULT_PLAYER_WEIGHTS),
        ("pitch weights",   DEFAULT_PITCH_WEIGHTS),
        ("pitch template",  DEFAULT_PITCH_TEMPLATE),
    ]:
        if not p.exists():
            return False, f"missing {label} at {p} (override via TC_* env vars)"
    return True, ""


def list_jobs() -> list[dict]:
    return sorted(
        (j.to_dict() for j in _jobs.values()),
        key=lambda d: d["startedAt"] or 0,
        reverse=True,
    )


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def create_job(video_name: str, label: str | None = None) -> Job:
    """Register a pending job for ``video_name`` (must live under data/uploads/).
    Caller must schedule ``run(job)`` on the event loop -- this is sync so the
    HTTP handler can return the job id straight away.
    """
    video_path = _UPLOAD_DIR / video_name
    if not video_path.is_file():
        raise FileNotFoundError(f"no such upload: {video_name}")
    ok, why = _assets_ok()
    if not ok:
        raise RuntimeError(why)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    match_id = _next_match_id()
    job = Job(
        id=uuid.uuid4().hex[:12],
        video_name=video_name,
        match_id=match_id,
        out_path=_CACHE_DIR / f"{match_id}.npz",
    )
    if label:
        job.log.append(f"label: {label}")
    _jobs[job.id] = job
    return job


async def run(job: Job, label: str | None = None) -> None:
    """Kick off the subprocess, stream its stdout+stderr into job.log."""
    video_path = _UPLOAD_DIR / job.video_name
    cmd = [
        sys.executable, "-m", "tools.prepare_video",
        "--video",           str(video_path),
        "--player-weights",  str(DEFAULT_PLAYER_WEIGHTS),
        "--pitch-weights",   str(DEFAULT_PITCH_WEIGHTS),
        "--pitch-template",  str(DEFAULT_PITCH_TEMPLATE),
        "--out",             str(job.out_path),
        "--label",           label or f"Uploaded: {job.video_name}",
    ]
    job.log.append("$ " + " ".join(shlex.quote(c) for c in cmd))
    job.status = "running"
    job.started_at = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_ROOT),
        )
        job._proc = proc
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                job.log.append(line)
        rc = await proc.wait()
        job.exit_code = rc
        job.finished_at = time.time()
        if job.status == "cancelled":
            return
        if rc == 0 and job.out_path.exists():
            job.status = "success"
            job.log.append(f"✓ wrote {job.out_path.name}")
        else:
            job.status = "failed"
            job.error = f"exit {rc}"
            # Clean up a half-written npz so list_matches doesn't advertise it.
            if job.out_path.exists() and rc != 0:
                try:
                    job.out_path.unlink()
                except OSError:
                    pass
    except Exception as e:
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"
        job.finished_at = time.time()
        job.log.append(f"! {job.error}")


async def cancel(job: Job) -> bool:
    if job.status not in ("pending", "running"):
        return False
    job.status = "cancelled"
    if job._proc is not None and job._proc.returncode is None:
        try:
            job._proc.terminate()
        except ProcessLookupError:
            pass
    job.log.append("cancelled by user")
    return True

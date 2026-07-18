"""Preprocess Metrica sample tracking data into the compact .npz arrays the
server loads at runtime.

Metrica ships three sample games (github.com/metrica-sports/sample-data):

  * Sample_Game_1, Sample_Game_2  -- classic Metrica CSV format
        (separate Home/Away tracking CSVs, 3 header rows, each player two
         columns of pitch-normalised x,y, ball last, 25 Hz)
  * Sample_Game_3                 -- EPTS format
        (metadata XML defining the channel order + a `frame:p1;p2;...:ball`
         tracking text file)

We never parse these live during the demo.  Run once:

    python -m tools.prepare_match                 # all three, from data/metrica
    python -m tools.prepare_match --game 1        # just one
    python -m tools.prepare_match --root some/dir --out cache

Each output cache/match_N.npz holds:
    timestamps      [frames]            seconds
    positions       [frames, players, 2] metres  (0..105, 0..68)
    ball_positions  [frames, 2]          metres
    player_teams    [players]            0 = home, 1 = away
    player_numbers  [players]            shirt numbers
    fps             scalar
    label           scalar str          e.g. "Test Match 1"
    events_json     scalar str          JSON list of match events, each
                                         {t, team, type, subtype, from, to},
                                         sorted by start time (seconds). Drives
                                         the dashboard's live event feed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _ROOT / "data" / "metrica" / "data"
_DEFAULT_OUT = _ROOT / "cache"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs, then back-fill any leading NaNs. 1-D in/out.

    A player who is subbed off (goes NaN) freezes at their last position; a
    player not yet on the pitch sits at their first tracked position. Neither
    matters for a replay demo and it keeps every frame renderable.
    """
    a = a.astype(np.float64, copy=True)
    valid = ~np.isnan(a)
    if not valid.any():
        a[:] = 0.0
        return a
    idx = np.where(valid, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    first = int(np.argmax(valid))
    if first > 0:
        a[:first] = a[first]
    return a


def _to_metres(norm_xy: np.ndarray) -> np.ndarray:
    """(..., 2) pitch-normalised (0..1) -> metres. Metrica origin is top-left."""
    out = norm_xy.astype(np.float32, copy=True)
    out[..., 0] *= PITCH_LENGTH
    out[..., 1] *= PITCH_WIDTH
    return out


def _pack(
    timestamps: np.ndarray,
    positions: np.ndarray,
    ball: np.ndarray,
    teams: np.ndarray,
    numbers: np.ndarray,
    fps: float,
    label: str,
    out_path: Path,
    events: list | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        timestamps=timestamps.astype(np.float32),
        positions=positions.astype(np.float32),
        ball_positions=ball.astype(np.float32),
        player_teams=teams.astype(np.int16),
        player_numbers=numbers.astype(np.int16),
        fps=np.float32(fps),
        label=np.str_(label),
        # Stored as one JSON string so the loader can stay allow_pickle=False.
        events_json=np.str_(json.dumps(events or [])),
    )
    print(
        f"  wrote {out_path.name}: {positions.shape[0]} frames, "
        f"{positions.shape[1]} players @ {fps:.0f}Hz "
        f"({positions.shape[0] / fps:.0f}s), {len(events or [])} events "
        f"-> {out_path.stat().st_size / 1e6:.1f} MB"
    )


# --------------------------------------------------------------------------- #
# events (shared shape across both formats)
# --------------------------------------------------------------------------- #
# Each event is normalised to {t, team, type, subtype, from, to}; the server
# maps type/subtype to display labels so wording can change without re-running
# this tool. `t` is the start time in seconds on the tracking timeline.
def _events_from_csv(path: Path) -> list[dict]:
    """Games 1 & 2: <stem>_RawEventsData.csv."""
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["Start Time [s]"])
            except (TypeError, ValueError, KeyError):
                continue
            team = "home" if (row.get("Team") or "").strip().lower() == "home" else "away"
            events.append({
                "t": round(t, 2),
                "team": team,
                "type": (row.get("Type") or "").strip(),
                "subtype": (row.get("Subtype") or "").strip(),
                "from": (row.get("From") or "").strip(),
                "to": (row.get("To") or "").strip(),
            })
    events.sort(key=lambda e: e["t"])
    return events


def _events_from_json(path: Path, team_side: dict[str, str]) -> list[dict]:
    """Game 3: <stem>_events.json (EPTS). `team_side` maps team id -> home/away."""
    if not path.exists():
        return []
    doc = json.loads(path.read_text())
    data = doc.get("data", []) if isinstance(doc, dict) else doc
    events: list[dict] = []
    for e in data:
        start = e.get("start") or {}
        t = start.get("time")
        if t is None:
            continue
        team = e.get("team") or {}
        sub = e.get("subtypes")
        if isinstance(sub, list):
            subname = ", ".join(s.get("name", "") for s in sub if isinstance(s, dict))
        elif isinstance(sub, dict):
            subname = sub.get("name", "") or ""
        else:
            subname = ""
        events.append({
            "t": round(float(t), 2),
            "team": team_side.get(team.get("id"), "home"),
            "type": ((e.get("type") or {}).get("name") or "").strip(),
            "subtype": subname.strip(),
            "from": ((e.get("from") or {}).get("name") or "").strip(),
            "to": ((e.get("to") or {}).get("name") or "").strip(),
        })
    events.sort(key=lambda e: e["t"])
    return events


# --------------------------------------------------------------------------- #
# Metrica CSV (Games 1 & 2)
# --------------------------------------------------------------------------- #
def _parse_metrica_csv_team(path: Path):
    """Return (time[frames], players=[(shirt, x[frames], y[frames])], ball[frames,2])."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)                # row 0: team labels
        numbers_row = next(reader)  # row 1: shirt numbers at each x-column
        labels_row = next(reader)   # row 2: Period, Frame, Time [s], Player.., Ball

    player_cols: list[tuple[int, int]] = []  # (shirt, x_column_index)
    ball_col: int | None = None
    for i, label in enumerate(labels_row):
        lab = label.strip()
        if lab.startswith("Player"):
            shirt = numbers_row[i].strip()
            player_cols.append((int(shirt) if shirt.isdigit() else i, i))
        elif lab == "Ball":
            ball_col = i

    data = np.genfromtxt(path, delimiter=",", skip_header=3)
    time = data[:, 2]
    players = [(shirt, data[:, xc], data[:, xc + 1]) for shirt, xc in player_cols]
    ball = (
        np.stack([data[:, ball_col], data[:, ball_col + 1]], axis=1)
        if ball_col is not None
        else np.full((data.shape[0], 2), np.nan)
    )
    return time, players, ball


def build_from_metrica_csv(game_dir: Path, label: str) -> dict:
    stem = game_dir.name  # e.g. Sample_Game_1
    home_path = game_dir / f"{stem}_RawTrackingData_Home_Team.csv"
    away_path = game_dir / f"{stem}_RawTrackingData_Away_Team.csv"

    t_home, home_players, ball = _parse_metrica_csv_team(home_path)
    _t_away, away_players, _ = _parse_metrica_csv_team(away_path)

    n = min(t_home.shape[0], _t_away.shape[0])
    timestamps = t_home[:n]

    teams: list[int] = []
    numbers: list[int] = []
    cols: list[np.ndarray] = []  # each (n, 2) metres
    for team_idx, roster in ((0, home_players), (1, away_players)):
        for shirt, xs, ys in roster:
            xs, ys = xs[:n], ys[:n]
            if np.isnan(xs[0]):  # keep the starting XI, drop unused subs
                continue
            xy = np.stack([_fill_nan(xs), _fill_nan(ys)], axis=1)
            cols.append(_to_metres(xy))
            teams.append(team_idx)
            numbers.append(int(shirt))

    positions = np.stack(cols, axis=1)  # (frames, players, 2)
    ball_m = _to_metres(np.stack([_fill_nan(ball[:n, 0]), _fill_nan(ball[:n, 1])], axis=1))
    fps = _infer_fps(timestamps)
    events = _events_from_csv(game_dir / f"{stem}_RawEventsData.csv")
    return dict(
        timestamps=timestamps, positions=positions, ball=ball_m,
        teams=np.array(teams), numbers=np.array(numbers), fps=fps, label=label,
        events=events,
    )


# --------------------------------------------------------------------------- #
# Metrica EPTS (Game 3)
# --------------------------------------------------------------------------- #
def build_from_metrica_epts(game_dir: Path, label: str) -> dict:
    stem = game_dir.name  # Sample_Game_3
    meta_path = game_dir / f"{stem}_metadata.xml"
    track_path = game_dir / f"{stem}_tracking.txt"

    tree = ET.parse(meta_path)
    root = tree.getroot()

    def tag(el: ET.Element) -> str:
        return el.tag.rsplit("}", 1)[-1]  # strip namespace if present

    fps = 25.0
    for el in root.iter():
        if tag(el) == "FrameRate" and el.text:
            fps = float(el.text)
            break

    # player id -> (shirt, team_id); teams mapped to 0/1 in document order
    team_order: list[str] = []
    player_meta: dict[str, tuple[int, str]] = {}
    for el in root.iter():
        if tag(el) == "Team":
            tid = el.get("id")
            if tid and tid not in team_order:
                team_order.append(tid)
        elif tag(el) == "Player":
            pid = el.get("id")
            team_id = el.get("teamId")
            shirt = 0
            for child in el:
                if tag(child) == "ShirtNumber" and child.text:
                    shirt = int(child.text)
            if pid:
                player_meta[pid] = (shirt, team_id or "")

    # channelId ("player1_x") -> playerId, from <PlayerChannel>
    channel_player: dict[str, str] = {}
    for el in root.iter():
        if tag(el) == "PlayerChannel":
            cid, pid = el.get("id"), el.get("playerId")
            if cid and pid:
                channel_player[cid] = pid

    # first DataFormatSpecification gives the column order + end frame
    spec = None
    for el in root.iter():
        if tag(el) == "DataFormatSpecification":
            spec = el
            break
    if spec is None:
        raise ValueError("no DataFormatSpecification in EPTS metadata")
    end_frame = int(spec.get("endFrame", "0"))

    ordered_pids: list[str] = []
    for ref in spec.iter():
        if tag(ref) == "PlayerChannelRef":
            cid = ref.get("playerChannelId", "")
            if cid.endswith("_x"):
                ordered_pids.append(channel_player[cid])
    n_players = len(ordered_pids)

    teams = np.array([team_order.index(player_meta[p][1]) for p in ordered_pids])
    numbers = np.array([player_meta[p][0] for p in ordered_pids])

    # tracking: "frame : p1;p2;...;pN : ballx,bally"
    xs = [[] for _ in range(n_players)]
    ys = [[] for _ in range(n_players)]
    ball_x: list[float] = []
    ball_y: list[float] = []

    def num(s: str) -> float:
        s = s.strip()
        return float(s) if s and s.upper() != "NAN" else np.nan

    with track_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split(":")
            if len(parts) < 2:
                continue
            if int(parts[0]) > end_frame:
                break
            fields = parts[1].split(";")
            for i in range(n_players):
                xy = fields[i].split(",") if i < len(fields) else ("", "")
                xs[i].append(num(xy[0]))
                ys[i].append(num(xy[1]))
            if len(parts) >= 3 and parts[2].strip():
                bx, by = (parts[2].split(",") + [""])[:2]
                ball_x.append(num(bx))
                ball_y.append(num(by))
            else:
                ball_x.append(np.nan)
                ball_y.append(np.nan)

    cols = [
        _to_metres(np.stack([_fill_nan(np.array(xs[i])), _fill_nan(np.array(ys[i]))], axis=1))
        for i in range(n_players)
    ]
    positions = np.stack(cols, axis=1)
    frames = positions.shape[0]
    ball_m = _to_metres(
        np.stack([_fill_nan(np.array(ball_x)), _fill_nan(np.array(ball_y))], axis=1)
    )
    timestamps = np.arange(frames) / fps
    team_side = {tid: ("home" if i == 0 else "away") for i, tid in enumerate(team_order)}
    events = _events_from_json(game_dir / f"{stem}_events.json", team_side)
    return dict(
        timestamps=timestamps, positions=positions, ball=ball_m,
        teams=teams, numbers=numbers, fps=fps, label=label,
        events=events,
    )


def _infer_fps(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return 25.0
    dt = float(np.median(np.diff(timestamps[:1000])))
    return round(1.0 / dt) if dt > 0 else 25.0


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
GAMES = {
    1: ("Sample_Game_1", "Test Match 1", build_from_metrica_csv),
    2: ("Sample_Game_2", "Test Match 2", build_from_metrica_csv),
    3: ("Sample_Game_3", "Test Match 3", build_from_metrica_epts),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess Metrica sample data -> cache/*.npz")
    ap.add_argument("--root", type=Path, default=_DEFAULT_DATA,
                    help="dir containing Sample_Game_1/2/3 (default: data/metrica/data)")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output dir (default: cache)")
    ap.add_argument("--game", type=int, choices=[1, 2, 3], action="append",
                    help="build only these games (repeatable); default all")
    args = ap.parse_args(argv)

    games = args.game or [1, 2, 3]
    built, failed = 0, 0
    for g in games:
        dirname, label, builder = GAMES[g]
        game_dir = args.root / dirname
        out_path = args.out / f"match_{g}.npz"
        print(f"[{label}] {game_dir}")
        if not game_dir.is_dir():
            print(f"  SKIP: {game_dir} not found "
                  f"(clone github.com/metrica-sports/sample-data into data/metrica)")
            failed += 1
            continue
        try:
            _pack(out_path=out_path, **builder(game_dir, label))
            built += 1
        except Exception as e:  # one bad game must not kill the others
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nDone: {built} built, {failed} skipped/failed. Output -> {args.out}")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())

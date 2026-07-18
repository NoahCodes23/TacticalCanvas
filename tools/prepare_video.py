"""Convert a fixed-camera soccer video into cache/match_N.npz matching the
schema in tools/prepare_match.py, so the dashboard can replay it exactly
like the Metrica scenarios.

Pipeline (all offline, one clip at a time):

  1. YOLO player+ball detection per frame (ultralytics).
  2. supervision.ByteTrack for tracker IDs.
  3. YOLO-pose pitch-keypoint detection; ONE homography for the whole clip,
     solved from the median keypoint pixel over the first ~120 frames.
     Assumes the camera is fixed -- see the skill doc.
  4. HSV jersey-colour KMeans(2) for team assignment (home/away).
  5. Keep the longest-lived N (<=22) tracker ids, fill gaps, write .npz.

Requires: ultralytics, supervision, opencv-python, scikit-learn, numpy.

Model weights (paths passed in):
  --player-weights  YOLO detecting {player, goalkeeper, referee, ball}
  --pitch-weights   YOLO-pose with pitch keypoints (Roboflow template)
  --pitch-template  JSON mapping each keypoint class name -> [x_m, y_m]
                    in Metrica coords (0..105, 0..68). Must line up with
                    pitch-weights' class order.

Usage:
    python -m tools.prepare_video \\
        --video clip.mp4 \\
        --player-weights weights/players.pt \\
        --pitch-weights  weights/pitch.pt  \\
        --pitch-template weights/pitch_template.json \\
        --out cache/match_4.npz \\
        --label "Recorded footage"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from ultralytics import YOLO

from tools.prepare_match import PITCH_LENGTH, PITCH_WIDTH, _fill_nan, _pack

CLS_BALL = "ball"
PLAYER_LIKE = {"player", "goalkeeper"}
TARGET_PLAYERS = 22


# --------------------------------------------------------------------------- #
# homography (fixed camera -> solve once)
# --------------------------------------------------------------------------- #
def _load_pitch_template(path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text())
    return {k: (float(v[0]), float(v[1])) for k, v in data.items()}


def solve_homography(
    video_path: Path,
    pitch_model: YOLO,
    template: dict[str, tuple[float, float]],
    warmup_frames: int = 120,
    conf_thresh: float = 0.5,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    per_name: dict[str, list[tuple[float, float]]] = {}
    names = pitch_model.names
    name_of = (lambda i: names[int(i)]) if isinstance(names, dict) else (lambda i: str(int(i)))
    try:
        for _ in range(warmup_frames):
            ok, frame = cap.read()
            if not ok:
                break
            result = pitch_model.predict(frame, verbose=False)[0]
            kp = result.keypoints
            if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
                continue
            xy = kp.xy.cpu().numpy()[0]                                   # (K, 2)
            confs = kp.conf.cpu().numpy()[0] if kp.conf is not None else np.ones(len(xy))
            for i, (pt, c) in enumerate(zip(xy, confs)):
                if c < conf_thresh or (pt[0] == 0 and pt[1] == 0):
                    continue
                per_name.setdefault(name_of(i), []).append((float(pt[0]), float(pt[1])))
    finally:
        cap.release()

    src_pts, dst_pts = [], []
    for name, pixels in per_name.items():
        if name not in template or len(pixels) < 5:
            continue
        src_pts.append(np.median(np.asarray(pixels), axis=0))
        dst_pts.append(template[name])
    if len(src_pts) < 4:
        raise RuntimeError(
            f"only {len(src_pts)} pitch keypoints matched template; need >=4. "
            "Check --pitch-template and --pitch-weights class names."
        )
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed.")
    inliers = int(mask.sum()) if mask is not None else len(src)
    print(f"  homography: {inliers}/{len(src)} keypoint inliers")
    return H


def _px_to_metres(pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    if pts_xy.size == 0:
        return pts_xy.reshape(0, 2)
    h = cv2.perspectiveTransform(pts_xy.reshape(-1, 1, 2).astype(np.float32), H)
    return h.reshape(-1, 2)


# --------------------------------------------------------------------------- #
# jersey colour (upper half of bbox, HSV median) for team clustering
# --------------------------------------------------------------------------- #
def _jersey_hsv(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1]); y2 = min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1 : y1 + (y2 - y1) // 2, x1:x2]           # upper half = jersey
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video",           type=Path, required=True)
    ap.add_argument("--player-weights",  type=Path, required=True)
    ap.add_argument("--pitch-weights",   type=Path, required=True)
    ap.add_argument("--pitch-template",  type=Path, required=True)
    ap.add_argument("--out",             type=Path, required=True)
    ap.add_argument("--label",           default="Recorded footage")
    ap.add_argument("--conf",            type=float, default=0.3)
    ap.add_argument("--every",           type=int, default=1,
                    help="process every Nth frame (1 = all).")
    args = ap.parse_args(argv)

    template = _load_pitch_template(args.pitch_template)

    print("[1/4] loading models")
    player_model = YOLO(str(args.player_weights))
    pitch_model  = YOLO(str(args.pitch_weights))

    print("[2/4] solving homography (fixed camera assumption)")
    H = solve_homography(args.video, pitch_model, template)

    print("[3/4] detecting + tracking")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))
    cnames = player_model.names
    name_of = (lambda i: cnames[int(i)]) if isinstance(cnames, dict) else (lambda i: str(int(i)))

    # (positions in pitch metres per kept frame; ball = None if missing)
    frame_players: list[dict[int, tuple[float, float]]] = []
    frame_ball:    list[tuple[float, float] | None]    = []
    hsv_samples:   dict[int, list[np.ndarray]]         = {}
    MAX_HSV = 40

    fidx = kept_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % args.every != 0:
            fidx += 1
            continue

        result = player_model.predict(frame, conf=args.conf, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        if det.class_id is None or len(det) == 0:
            frame_players.append({})
            frame_ball.append(None)
            fidx += 1; kept_frames += 1
            continue

        # ---- ball: highest-confidence ball detection, if any -----------------
        ball_mask = np.array([name_of(c) == CLS_BALL for c in det.class_id])
        ball_xy = None
        if ball_mask.any():
            b = det[ball_mask]
            best = int(np.argmax(b.confidence))
            box = b.xyxy[best]
            cx = 0.5 * (box[0] + box[2])
            cy = box[3]                                        # ground contact
            m = _px_to_metres(np.array([[cx, cy]]), H)[0]
            ball_xy = (float(m[0]), float(m[1]))
        frame_ball.append(ball_xy)

        # ---- players + goalkeepers, tracked ---------------------------------
        pmask = np.array([name_of(c) in PLAYER_LIKE for c in det.class_id])
        players = det[pmask]
        tracked = tracker.update_with_detections(players)

        this: dict[int, tuple[float, float]] = {}
        if len(tracked) > 0 and tracked.tracker_id is not None:
            xyxy = tracked.xyxy
            foot = np.stack([0.5 * (xyxy[:, 0] + xyxy[:, 2]), xyxy[:, 3]], axis=1)
            metres = _px_to_metres(foot, H)
            for tid, (mx, my), box in zip(tracked.tracker_id, metres, xyxy):
                this[int(tid)] = (float(mx), float(my))
                bucket = hsv_samples.setdefault(int(tid), [])
                if len(bucket) < MAX_HSV:
                    h = _jersey_hsv(frame, box)
                    if h is not None:
                        bucket.append(h)
        frame_players.append(this)

        fidx += 1; kept_frames += 1
        if kept_frames % 200 == 0:
            print(f"  frame {fidx} (kept {kept_frames})")
    cap.release()

    n_frames = len(frame_players)
    if n_frames == 0:
        raise RuntimeError("no frames processed from video.")
    print(f"  processed {n_frames} frames")

    # ---- pick top-K longest-lived tracker ids ------------------------------
    counts: dict[int, int] = {}
    for f in frame_players:
        for tid in f:
            counts[tid] = counts.get(tid, 0) + 1
    kept = sorted(counts, key=lambda t: counts[t], reverse=True)[:TARGET_PLAYERS]
    if len(kept) < TARGET_PLAYERS:
        print(f"  WARN: only {len(kept)} stable tracks (< {TARGET_PLAYERS}).")
    print(f"  kept {len(kept)} track ids:", kept)

    # ---- team assignment via jersey HSV KMeans(2) ---------------------------
    print("[4/4] team clustering + packing")
    tid_hsv: dict[int, np.ndarray] = {}
    for tid in kept:
        samples = hsv_samples.get(tid, [])
        if samples:
            tid_hsv[tid] = np.median(np.stack(samples, axis=0), axis=0)

    team_of: dict[int, int] = {tid: 0 for tid in kept}
    if len(tid_hsv) >= 2:
        ids = list(tid_hsv)
        X = np.stack([tid_hsv[t] for t in ids], axis=0).astype(np.float32)
        # weight hue (jersey colour) up, value (lighting) down.
        Xw = X * np.array([2.0, 1.0, 0.3], dtype=np.float32)
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xw).labels_
        raw = {t: int(l) for t, l in zip(ids, labels)}
        # deterministic: cluster with smaller mean hue -> home (0)
        by_cluster: dict[int, list[float]] = {}
        for t, l in raw.items():
            by_cluster.setdefault(l, []).append(float(tid_hsv[t][0]))
        order = sorted(by_cluster, key=lambda l: np.mean(by_cluster[l]))
        remap = {order[0]: 0, order[1]: 1} if len(order) == 2 else {order[0]: 0}
        for t in kept:
            team_of[t] = remap.get(raw.get(t, 0), 0)

    P = len(kept)
    teams   = np.array([team_of[t] for t in kept], dtype=np.int16)
    numbers = np.arange(1, P + 1, dtype=np.int16)

    # ---- positions matrix + gap fill ---------------------------------------
    positions = np.full((n_frames, P, 2), np.nan, dtype=np.float32)
    ball      = np.full((n_frames, 2),   np.nan, dtype=np.float32)
    for i, f in enumerate(frame_players):
        for j, tid in enumerate(kept):
            if tid in f:
                positions[i, j] = f[tid]
        if frame_ball[i] is not None:
            ball[i] = frame_ball[i]

    for j in range(P):
        positions[:, j, 0] = _fill_nan(positions[:, j, 0])
        positions[:, j, 1] = _fill_nan(positions[:, j, 1])
    ball[:, 0] = _fill_nan(ball[:, 0])
    ball[:, 1] = _fill_nan(ball[:, 1])

    np.clip(positions[..., 0], 0, PITCH_LENGTH, out=positions[..., 0])
    np.clip(positions[..., 1], 0, PITCH_WIDTH,  out=positions[..., 1])
    np.clip(ball[..., 0],      0, PITCH_LENGTH, out=ball[..., 0])
    np.clip(ball[..., 1],      0, PITCH_WIDTH,  out=ball[..., 1])

    effective_fps = fps / args.every
    timestamps = (np.arange(n_frames, dtype=np.float32) / effective_fps)

    _pack(
        timestamps=timestamps,
        positions=positions,
        ball=ball,
        teams=teams,
        numbers=numbers,
        fps=effective_fps,
        label=args.label,
        out_path=args.out,
        events=[],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

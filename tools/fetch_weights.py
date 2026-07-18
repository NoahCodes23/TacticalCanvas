"""Prepare weights/ so tools/prepare_video can run.

What this script guarantees:

  1. weights/pitch_template.json is generated from the same 32-point
     PitchConfiguration the Roboflow sports models use, rescaled to Metrica
     coords (0..105, 0..68 metres). Keys are written under both "0".."31" and
     "01".."32" so it matches whichever naming convention the pitch model
     you download happens to expose.

  2. It checks whether players.pt and pitch.pt already exist. If not, it
     prints the exact Universe pages to visit and what to name the files.
     There is no reliable public API to pull the pretrained YOLO weights
     programmatically -- Roboflow's Python SDK downloads training datasets,
     not the fine-tuned .pt files. Trying to fake it silently would just
     produce broken homography later.

Usage:
    python -m tools.fetch_weights                       # write template + status
    python -m tools.fetch_weights --out weights/        # custom directory
    python -m tools.fetch_weights --template-only       # skip weight check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _ROOT / "weights"

# Metrica pitch (the coordinate system the rest of TacticalCanvas uses).
PITCH_L, PITCH_W = 105.0, 68.0

# Roboflow's PitchConfiguration is defined on a 120m x 70m pitch in
# centimetres. We compute their 32 vertices in cm, then linearly rescale to
# Metrica's (105, 68). Homography handles arbitrary world scale, so a linear
# rescale is safe -- and downstream code already expects Metrica units.
_LENGTH_CM = 12000
_WIDTH_CM  = 7000
_PEN_BOX_W = 4100       # penalty area width  (along y, the short axis)
_PEN_BOX_L = 2015       # penalty area length (along x, into the pitch)
_GOAL_BOX_W = 1832
_GOAL_BOX_L = 550
_CENTRE_R  = 915
_PEN_SPOT  = 1100


def _vertices_cm() -> list[tuple[float, float]]:
    L, W = _LENGTH_CM, _WIDTH_CM
    pw, pl = _PEN_BOX_W, _PEN_BOX_L
    gw, gl = _GOAL_BOX_W, _GOAL_BOX_L
    cr, ps = _CENTRE_R, _PEN_SPOT
    return [
        (0,              0),                    # 1  top-left corner
        (0,              (W - pw) / 2),         # 2  left pen-box top
        (0,              (W - gw) / 2),         # 3  left goal-box top
        (0,              (W + gw) / 2),         # 4  left goal-box bottom
        (0,              (W + pw) / 2),         # 5  left pen-box bottom
        (0,              W),                    # 6  bottom-left corner
        (gl,             (W - gw) / 2),         # 7  left goal-box front top
        (gl,             (W + gw) / 2),         # 8  left goal-box front bottom
        (ps,             W / 2),                # 9  left penalty spot
        (pl,             (W - pw) / 2),         # 10 left pen-box front top
        (pl,             (W - gw) / 2),         # 11
        (pl,             (W + gw) / 2),         # 12
        (pl,             (W + pw) / 2),         # 13 left pen-box front bottom
        (L / 2,          0),                    # 14 halfway line top
        (L / 2,          W / 2 - cr),           # 15 centre-circle top
        (L / 2,          W / 2 + cr),           # 16 centre-circle bottom
        (L / 2,          W),                    # 17 halfway line bottom
        (L - pl,         (W - pw) / 2),         # 18 right pen-box front top
        (L - pl,         (W - gw) / 2),         # 19
        (L - pl,         (W + gw) / 2),         # 20
        (L - pl,         (W + pw) / 2),         # 21 right pen-box front bottom
        (L - ps,         W / 2),                # 22 right penalty spot
        (L - gl,         (W - gw) / 2),         # 23 right goal-box front top
        (L - gl,         (W + gw) / 2),         # 24 right goal-box front bottom
        (L,              0),                    # 25 top-right corner
        (L,              (W - pw) / 2),         # 26 right pen-box top
        (L,              (W - gw) / 2),         # 27 right goal-box top
        (L,              (W + gw) / 2),         # 28 right goal-box bottom
        (L,              (W + pw) / 2),         # 29 right pen-box bottom
        (L,              W),                    # 30 bottom-right corner
        (L / 2 - cr,     W / 2),                # 31 centre-circle left
        (L / 2 + cr,     W / 2),                # 32 centre-circle right
    ]


def _to_metrica(v_cm: tuple[float, float]) -> tuple[float, float]:
    sx = PITCH_L / (_LENGTH_CM / 100)
    sy = PITCH_W / (_WIDTH_CM  / 100)
    return (round(v_cm[0] / 100 * sx, 3), round(v_cm[1] / 100 * sy, 3))


def write_template(path: Path) -> None:
    verts = [_to_metrica(v) for v in _vertices_cm()]
    # Provide both "0".."31" (0-indexed) and "01".."32" (1-indexed, zero-padded)
    # so the JSON matches whichever class-name style the pitch model uses.
    tpl: dict[str, list[float]] = {}
    for i, (x, y) in enumerate(verts):
        tpl[str(i)] = [x, y]
        tpl[f"{i + 1:02d}"] = [x, y]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tpl, indent=2))


# --------------------------------------------------------------------------- #
# weight status: report only, don't pretend we can auto-download
# --------------------------------------------------------------------------- #
_WEIGHT_SOURCES = [
    ("players.pt",
     "https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc",
     "Detects {player, goalkeeper, referee, ball}. "
     "Open the page, pick the latest version, click 'Download Weights → YOLOv8'."),
    ("pitch.pt",
     "https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi",
     "YOLO-pose model with 32 pitch keypoints. Same download flow."),
]


def report_weights(out: Path) -> int:
    """Return the number of missing weight files."""
    missing = 0
    for name, url, why in _WEIGHT_SOURCES:
        p = out / name
        if p.exists():
            print(f"  ✓ {p} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ✗ {p} missing")
            print(f"      {why}")
            print(f"      {url}")
            print(f"      After downloading, rename to {name} and drop in {out}/")
            missing += 1
    return missing


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help="target directory (default: weights/)")
    ap.add_argument("--template-only", action="store_true",
                    help="only write pitch_template.json; skip the weight check")
    args = ap.parse_args(argv)

    tpl_path = args.out / "pitch_template.json"
    print(f"[template] writing {tpl_path}")
    write_template(tpl_path)
    print(f"  32 pitch keypoints, keys 0..31 and 01..32 (both styles)")

    if args.template_only:
        return 0

    print(f"[weights] checking {args.out}/")
    missing = report_weights(args.out)
    if missing:
        print(f"\n{missing} weight file(s) still missing. Re-run once downloaded.")
        return 1
    print("\nAll assets present. tools/prepare_video should now run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Auto-formation detection from live player positions.

Sort each team's outfielders (GK dropped) by depth-from-own-goal, then read off
lines from the largest gaps in that sorted list -- 4-3-3 shows as three tight
clusters with two clear gaps, 4-4-2 as different clusters with different gaps.

We try both a 3-line split and a 4-line split, and only accept the 4-line one if
its extra cut is close in size to the smaller of the two 3-line cuts. That
prevents a formation like 4-3-3 from being called "4-3-2-1" just because the two
strikers happen to be a metre apart in depth on that frame.
"""


K4_GAP_RATIO = 0.6   # how close the 3rd gap must be to the 2nd to accept a 4th line
MIN_OUTFIELD = 6     # under this many outfielders the shape isn't stable


def _split_at(depths_sorted: list[float], cut_between: list[int]) -> list[int]:
    """Given cut positions (index i = the gap between player i and i+1), return
    the line sizes."""
    cuts = sorted(cut_between)
    sizes, start = [], 0
    n = len(depths_sorted)
    for c in cuts:
        sizes.append(c + 1 - start)
        start = c + 1
    sizes.append(n - start)
    return sizes


def detect_formation(depths: list[float]) -> str:
    """Return a formation label like "4-3-3" from a list of outfielder depths
    (GK already excluded). Empty string if there isn't enough data."""
    if len(depths) < MIN_OUTFIELD:
        return ""
    ds = sorted(depths)
    gaps = sorted(
        [(ds[i + 1] - ds[i], i) for i in range(len(ds) - 1)],
        key=lambda g: g[0],
        reverse=True,
    )
    if len(gaps) < 2:
        return ""

    lines3 = _split_at(ds, [gaps[0][1], gaps[1][1]])
    label3 = "-".join(str(x) for x in lines3)

    if len(gaps) >= 3 and gaps[2][0] >= K4_GAP_RATIO * gaps[1][0]:
        lines4 = _split_at(ds, [gaps[0][1], gaps[1][1], gaps[2][1]])
        if min(lines4) >= 1:
            return "-".join(str(x) for x in lines4)
    return label3

"""
Stage 4b, part 1: geometric QA of the clicked anchors (anchors.json).

Checks every frame in both views and flags clicks that break racket geometry:
  - throat should sit 20 to 60 % of the way from butt to tip
  - throat should lie close to the butt->tip line
  - edge point should sit in the head (40 to 105 % along the axis)
  - the 2D axis angle should not jump vs. its neighbours
If a frame fails, it tries all 24 relabellings of the four points and, if one
fits the neighbouring frames far better, writes the corrected labels.

Usage:  python pipeline/anchor_qa.py data/anchors.json
Output: anchors_clean.json and anchor_qa_report.csv, next to the input file
"""
import csv, itertools, json, sys
import numpy as np

import os
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "anchors.json")
DIR = os.path.dirname(os.path.abspath(PATH))
d = json.load(open(PATH))
PTS = d["points"]


def geometry(fd):
    P = {p: np.array(fd[p], float) for p in PTS}
    ax = P["tip"] - P["butt"]
    L = np.linalg.norm(ax)
    u = ax / L
    rel = lambda q: P[q] - P["butt"]
    perp = lambda q: abs(u[0] * rel(q)[1] - u[1] * rel(q)[0])  # 2D cross product
    return dict(angle=np.degrees(np.arctan2(ax[1], ax[0])), L=L,
                t_throat=rel("throat") @ u / L, t_edge=rel("edge") @ u / L,
                throat_off=perp("throat") / L)


def problems(g, angle_jump):
    out = []
    if not 0.2 < g["t_throat"] < 0.6: out.append(f"throat at {g['t_throat']:.2f} of axis")
    if g["throat_off"] > 0.15: out.append("throat off the butt-tip line")
    if not 0.4 < g["t_edge"] < 1.05: out.append(f"edge at {g['t_edge']:.2f} of axis")
    if angle_jump > 25: out.append(f"axis angle jumps {angle_jump:.0f} deg")
    return out


clean = json.loads(json.dumps(d))  # deep copy; frame entries keep the original keys only
report = []
for view, vd in d["views"].items():
    frames = sorted(vd["frames"], key=int)
    geo = {f: geometry(vd["frames"][f]) for f in frames}
    ang = dict(zip(frames, np.degrees(np.unwrap(np.radians([geo[f]["angle"] for f in frames])))))
    for i, f in enumerate(frames):
        nbrs = [frames[j] for j in (i - 1, i + 1) if 0 <= j < len(frames)]
        jump = abs(ang[f] - np.median([ang[n] for n in nbrs]))
        issues = problems(geo[f], jump)
        action = ""
        if issues:
            # try relabelling against the mean of the neighbouring frames
            ref = {p: np.mean([vd["frames"][n][p] for n in nbrs], axis=0) for p in PTS}
            vals = [np.array(vd["frames"][f][p], float) for p in PTS]
            cost = lambda perm: sum(np.linalg.norm(vals[k] - ref[p]) for k, p in zip(perm, PTS))
            orig = cost(range(4))
            best = min(itertools.permutations(range(4)), key=cost)
            if cost(best) < 0.3 * orig and list(best) != [0, 1, 2, 3]:
                for k, p in zip(best, PTS):
                    clean["views"][view]["frames"][f][p] = vd["frames"][f][PTS[k]]
                fixed = problems(geometry(clean["views"][view]["frames"][f]), 0)
                mapping = ", ".join(f"{p}<-{PTS[k]}" for k, p in zip(best, PTS) if PTS[k] != p)
                action = f"relabelled ({mapping})" + ("" if not fixed else f"; still: {fixed}")
            else:
                action = "check by eye (no relabelling fits)"
            clean.setdefault("qa_flags", {}).setdefault(view, {})[f] = action
        report.append([view, f, round(geo[f]["angle"], 1), round(geo[f]["L"], 1),
                       round(geo[f]["t_throat"], 2), round(geo[f]["t_edge"], 2),
                       "; ".join(issues), action])

json.dump(clean, open(os.path.join(DIR, "anchors_clean.json"), "w"), indent=1)
with open(os.path.join(DIR, "anchor_qa_report.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["view", "frame", "axis_angle_deg", "length_px", "throat_frac", "edge_frac", "issues", "action"])
    w.writerows(report)

flagged = [r for r in report if r[6]]
print(f"{len(report)} frames checked, {len(flagged)} flagged")
for r in flagged:
    print(f"  {r[0]} frame {r[1]}: {r[6]}  ->  {r[7]}")

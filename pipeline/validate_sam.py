"""
SAM 2 as referee, laptop side (CPU only, no torch): per-frame 2D racket axis angles compared
on the frames where SAM's mask is trusted (elongation >= 1.6).
  clicks vs SAM  -> quality of our hand-clicked anchors
  sensor vs SAM  -> independent check of the sensor-only reconstruction, as projected into
                    each video by the referee fit (out/overlay_points.json, butt->tip = points 0->2)
Angles are undirected lines, compared modulo 180 deg: a mask's PCA axis has no butt/tip direction.
Input:  out/sam2/sam_axes.json (pipeline/sam2_axis.py), out/overlay_points.json, data/anchors_clean.json
Output: out/sam2/sam_validation.json, out/fig4_sam_validation.png
"""
import json, os, numpy as np, matplotlib
import paths; os.chdir(paths.ROOT)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import FS, IMPACT
from sam2_axis import line_angle, line_diff, MIN_ELONGATION, DISAGREE_DEG

sam = json.load(open("out/sam2/sam_axes.json")); ov = json.load(open("out/overlay_points.json"))
anch = json.load(open("data/anchors_clean.json"))


def stats(x):
    x = np.abs(np.asarray(x, float))
    return dict(median=round(float(np.median(x)), 1), mean=round(float(x.mean()), 1), max=round(float(x.max()), 1)) if len(x) else None


res = dict(note="2D axis line angles, modulo 180 deg; SAM frames used only if mask elongation >= min_elongation",
           min_elongation=MIN_ELONGATION, disagree_deg=DISAGREE_DEG,
           sam_run={k: sam[k] for k in ("model", "device", "frame_offset")}, views={})
for view in [v for v in anch["views"] if v in sam["views"]]:
    vd, sv = anch["views"][view], sam["views"][view]["frames"]
    pred = {str(f): np.array(P) for f, P in zip(ov[view]["frames"], ov[view]["pred"])}
    per = []
    for f in sorted(vd["frames"], key=int):
        a, s = vd["frames"][f], sv[f]
        trusted = s is not None and s["elongation"] >= MIN_ELONGATION
        click = line_angle(np.subtract(a["tip"], a["butt"])); sensor = line_angle(pred[f][2] - pred[f][0])
        per.append(dict(frame=int(f), imu_sample=a["imu_sample"], t_from_contact_ms=round((a["imu_sample"] - IMPACT) / FS * 1000, 1),
                        trusted=trusted, elongation=None if s is None else s["elongation"],
                        clicks_vs_sam_deg=round(line_diff(click, s["angle_deg"]), 1) if trusted else None,
                        sensor_vs_sam_deg=round(line_diff(sensor, s["angle_deg"]), 1) if trusted else None))
    out = dict(total_frames=len(per), trusted_frames=sum(q["trusted"] for q in per))
    for phase, sel in (("forward_swing", lambda q: q["imu_sample"] < IMPACT), ("follow_through", lambda q: q["imu_sample"] >= IMPACT)):
        qs = [q for q in per if sel(q)]; tq = [q for q in qs if q["trusted"]]
        out[phase] = dict(frames=len(qs), trusted=len(tq),
                          clicks_vs_sam=stats([q["clicks_vs_sam_deg"] for q in tq]),
                          sensor_vs_sam=stats([q["sensor_vs_sam_deg"] for q in tq]))
    out["clicks_disagree_over_8deg"] = [dict(frame=q["frame"], imu_sample=q["imu_sample"], diff_deg=q["clicks_vs_sam_deg"])
                                        for q in per if q["trusted"] and abs(q["clicks_vs_sam_deg"]) > DISAGREE_DEG]
    out["untrusted"] = [dict(frame=q["frame"], imu_sample=q["imu_sample"],
                             reason="no mask" if q["elongation"] is None else f"elongation {q['elongation']}")
                        for q in per if not q["trusted"]]
    out["per_frame"] = per
    res["views"][view] = out
json.dump(res, open("out/sam2/sam_validation.json", "w"), indent=1)

fmt = lambda s: "n/a" if s is None else f"median {s['median']:4.1f}  mean {s['mean']:4.1f}  max {s['max']:4.1f}"
for view, out in res["views"].items():
    print(f"{view}: {out['trusted_frames']}/{out['total_frames']} trusted frames")
    for phase in ("forward_swing", "follow_through"):
        o = out[phase]
        print(f"  {phase:15s} ({o['trusted']:2d}/{o['frames']:2d} trusted)  clicks vs SAM: {fmt(o['clicks_vs_sam'])}   sensor vs SAM: {fmt(o['sensor_vs_sam'])}")
    for d in out["clicks_disagree_over_8deg"]:
        print(f"  clicks and SAM disagree by {d['diff_deg']:+.1f} deg on frame {d['frame']}")
    if out["untrusted"]:
        print("  untrusted: " + ", ".join(f"{u['frame']} ({u['reason']})" for u in out["untrusted"]))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
panels = (("clicks_vs_sam", "Click quality: clicked axis vs SAM 2 axis (SAM prompted on the first frame only)"),
          ("sensor_vs_sam", "Independent check: sensor-only axis vs SAM 2 axis"))
for i, (key, title) in enumerate(panels):
    for v, c in (("front", "#185FA5"), ("rear", "#1D9E75")):
        if v not in res["views"]: continue
        out = res["views"][v]; per = out["per_frame"]; m = out["forward_swing"][key]
        tt = [q["t_from_contact_ms"] for q in per]
        e = [abs(q[f"{key}_deg"]) if q["trusted"] else np.nan for q in per]   # NaN breaks the line at untrusted frames
        ax[i].plot(tt, e, "o-", color=c, ms=4, label=f"{v} view, forward-swing median " + ("n/a" if m is None else f"{m['median']}°"))
        if out["untrusted"]:   # an empty clip_on=False line has a bbox at the figure origin and wrecks tight_layout
            ax[i].plot([q["t_from_contact_ms"] for q in per if not q["trusted"]], [0] * len(out["untrusted"]),
                       "x", color=c, ms=7, mew=1.5, clip_on=False)
    if any(o["untrusted"] for o in res["views"].values()):
        ax[i].plot([], [], "x", color="#888", mew=1.5, label="SAM mask not trusted (excluded)")
    ax[i].axvline(0, color="#D85A30"); ax[i].axvspan(-500, 0, color="#1D9E75", alpha=.06)
    ax[i].set_ylabel("axis angle difference (°)"); ax[i].set_title(title, fontsize=11); ax[i].legend(frameon=False, loc="upper left")
top = np.nanmax([abs(q["clicks_vs_sam_deg"]) for out in res["views"].values() for q in out["per_frame"] if q["trusted"]] or [0])
ax[0].axhline(DISAGREE_DEG, ls=":", color="#999"); ax[0].set_ylim(0, max(20, 1.15 * top))
ax[1].set_ylim(0, 90); ax[1].set_xlabel("time from contact (ms)")
fig.tight_layout(); fig.savefig("out/fig4_sam_validation.png", dpi=160)
print("wrote out/sam2/sam_validation.json, out/fig4_sam_validation.png")

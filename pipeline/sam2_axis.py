"""
Automatic racket axis from SAM 2 masks: an independent referee for the clicked anchors
and for the sensor-only reconstruction. Nothing here feeds back into the orientation.

For each view:
  1. decode the anchor frames from the video (sequentially: seeking in H.264 is not
     frame-exact on every OpenCV build, and the laptop check must match the GPU run)
  2. prompt SAM 2 ONCE, on the first frame, with the 4 clicked points, so every later
     mask is independent of our clicks
  3. let SAM 2 track the racket through the rest of the window by itself
  4. PCA on each mask -> 2D axis angle + elongation (confidence)
  5. leakage guard: if the mask looks like it grew onto the hand or forearm (area jumps
     more than 50% between frames, or elongation below 1.6), rerun the view once with one
     negative point at the wrist, 0.12 racket lengths beyond the butt click on the first
     frame, and keep whichever attempt has fewer suspect frames

Angles are undirected line angles in [-90, 90) degrees (image x right, y down): PCA cannot
tell butt from tip, so every comparison is modulo 180 degrees.

Usage (repo-relative paths, run from anywhere):
  python pipeline/sam2_axis.py --check-only             # laptop: verify frame numbering, no torch needed
  python pipeline/sam2_axis.py --sam-dir /path/to/sam2  # GPU: full run, see hpc/roihu/README.md
Output in out/sam2/:
  first_frame_check_<view>.png, sam_axes.json, sam_vs_clicks.csv, overlays/<view>/<frame>.jpg, run_log.txt
"""
import argparse, contextlib, csv, json, os, platform, sys, tempfile, time
from pathlib import Path
import numpy as np
import cv2
import paths

ANCHORS = paths.DATA / "anchors_clean.json"
OUT = paths.OUT / "sam2"
MODELS = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}   # checkpoint name -> config suffix
DISAGREE_DEG = 8.0        # flag frames where clicks and SAM differ by more than this
MIN_ELONGATION = 1.6      # below this the mask is too round (face-on, blurred or leaking) to trust its axis
AREA_JUMP = 0.5           # leakage guard: relative mask-area change between consecutive frames
WRIST_BEYOND_BUTT = 0.12  # wrist estimate, in racket lengths beyond the butt click
FONT = cv2.FONT_HERSHEY_SIMPLEX


def line_angle(d):
    """Angle of the undirected image line along d, in [-90, 90) degrees."""
    return float((np.degrees(np.arctan2(d[1], d[0])) + 90.0) % 180.0 - 90.0)


def line_diff(a, b):
    """Signed difference a - b between two line angles, folded into [-90, 90)."""
    return float((a - b + 90.0) % 180.0 - 90.0)


def mask_axis(mask):
    """Long axis of a binary mask via PCA, or None if the mask is (almost) empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))   # ascending eigenvalues
    return dict(angle_deg=round(line_angle(evecs[:, 1]), 2),
                elongation=round(float(np.sqrt(evals[1] / max(evals[0], 1e-9))), 2),
                centroid=[round(float(v), 1) for v in c], area_px=int(len(xs)))


def suspect_frames(axes, frames):
    """Frames where the mask may have leaked onto the hand or forearm, or vanished."""
    bad, prev = [], None
    for f in frames:
        a = axes[f]
        if a is None or a["elongation"] < MIN_ELONGATION or (
                prev is not None and abs(a["area_px"] - prev["area_px"]) > AREA_JUMP * prev["area_px"]):
            bad.append(f)
        prev = a
    return bad


def read_frames(video, frames, offset):
    """{anchor frame: BGR image}, with video index = anchor frame - offset, decoded sequentially."""
    want = {int(f) - offset: f for f in frames}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    imgs, idx = {}, 0
    while idx <= max(want):
        ok, img = cap.read()
        if not ok:
            break
        if idx in want:
            imgs[want[idx]] = img
        idx += 1
    cap.release()
    missing = [f for f in frames if f not in imgs]
    if missing:
        raise RuntimeError(f"{Path(video).name}: could not read frames {missing} (frame offset {offset})")
    return imgs


def save_check(view, f, img, clicks, offset):
    """Clicked points on the decoded first frame. If they miss the racket, the frame numbering is off."""
    out = img.copy()
    h, w = out.shape[:2]
    pts = np.array(list(clicks.values()), float)
    lo = np.maximum(pts.min(0) - 50, 0).astype(int)
    hi = np.minimum(pts.max(0) + 50, [w, h]).astype(int)
    crop = img[lo[1]:hi[1], lo[0]:hi[0]]
    z = min(3.0, 520 / crop.shape[1], 380 / crop.shape[0])
    inset = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    for name, (x, y) in clicks.items():
        cv2.circle(inset, (int((x - lo[0]) * z), int((y - lo[1]) * z)), 7, (0, 0, 255), 2, cv2.LINE_AA)
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, name, (p[0] + 8, p[1] - 8), FONT, .5, (0, 0, 255), 1, cv2.LINE_AA)
    ih, iw = inset.shape[:2]                      # zoomed inset in the corner away from the racket
    cx, cy = pts.mean(0)
    x0 = w - iw - 10 if cx < w / 2 else 10
    y0 = h - ih - 10 if cy < h / 2 else 46
    out[y0:y0 + ih, x0:x0 + iw] = inset
    cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(out, f"{view}: anchor frame {f} = video index {int(f) - offset} (offset {offset}). "
                     f"Red circles should sit on the racket.", (10, 25), FONT, .6, (255, 255, 255), 1, cv2.LINE_AA)
    path = OUT / f"first_frame_check_{view}.png"
    cv2.imwrite(str(path), out)
    return path


def draw_overlay(img, mask, ax, click, title):
    """Green = SAM mask and axis, red = clicked butt->tip axis."""
    out = img.copy()
    if mask is not None and mask.any():
        out[mask] = (0.6 * out[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    b, t = np.array(click["butt"], float), np.array(click["tip"], float)
    if ax is not None:
        c, a = np.array(ax["centroid"]), np.radians(ax["angle_deg"])
        u = 0.5 * np.linalg.norm(t - b) * np.array([np.cos(a), np.sin(a)])
        cv2.line(out, tuple((c - u).astype(int).tolist()), tuple((c + u).astype(int).tolist()), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(out, tuple(b.astype(int).tolist()), tuple(t.astype(int).tolist()), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 25), FONT, .7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pick_device(req, torch):
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return req


def load_predictor(a, log):
    import torch
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:                            # not pip-installed: use the clone directly
        sys.path.insert(0, str(a.sam_dir))
        from sam2.build_sam import build_sam2_video_predictor
    device = pick_device(a.device, torch)
    ckpt = Path(a.checkpoint) if a.checkpoint else Path(a.sam_dir) / "checkpoints" / f"sam2.1_hiera_{a.model}.pt"
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    cfg = f"configs/sam2.1/sam2.1_hiera_{MODELS[a.model]}.yaml"
    log(f"torch {torch.__version__}, device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    t = time.perf_counter()
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
    log(f"model {a.model}: {cfg} + {ckpt.name}, loaded in {time.perf_counter() - t:.1f}s")
    return predictor, device, ckpt


def track(predictor, jpg_dir, points, labels, device):
    """Prompt SAM 2 on frame 0 only, then let it propagate. Returns {frame index: bool mask}."""
    import torch
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    masks = {}
    with torch.inference_mode(), amp:
        state = predictor.init_state(video_path=jpg_dir)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, points=points, labels=labels)
        for k, _, logits in predictor.propagate_in_video(state):
            masks[k] = (logits[0, 0] > 0.0).cpu().numpy()
    return masks


def run_view(predictor, device, view, vd, names, offset, log):
    t0 = time.perf_counter()
    frames = sorted(vd["frames"], key=int)
    imgs = read_frames(paths.VIDEOS / vd["video"], frames, offset)
    first = vd["frames"][frames[0]]
    save_check(view, frames[0], imgs[frames[0]], {p: first[p] for p in names}, offset)
    pos = np.array([first[p] for p in names], np.float32)
    butt, tip = np.array(first["butt"], float), np.array(first["tip"], float)
    h, w = imgs[frames[0]].shape[:2]
    wrist = np.clip(butt + WRIST_BEYOND_BUTT * (butt - tip), [0, 0], [w - 1, h - 1]).round(1)
    log(f"[{view}] {len(frames)} frames {frames[0]}..{frames[-1]} decoded in {time.perf_counter() - t0:.1f}s, "
        f"prompt: {len(pos)} clicks on frame {frames[0]} only")

    with tempfile.TemporaryDirectory(prefix="_frames_", dir=OUT) as jpg_dir:
        for k, f in enumerate(frames):            # SAM 2 wants 00000.jpg, 00001.jpg, ...
            cv2.imwrite(os.path.join(jpg_dir, f"{k:05d}.jpg"), imgs[f], [cv2.IMWRITE_JPEG_QUALITY, 95])

        def attempt(points, labels, tag):
            t = time.perf_counter()
            by_idx = track(predictor, jpg_dir, points, labels, device)
            masks = {f: by_idx.get(k) for k, f in enumerate(frames)}
            axes = {f: None if m is None else mask_axis(m) for f, m in masks.items()}
            bad = suspect_frames(axes, frames)
            log(f"[{view}] {tag}: tracked in {time.perf_counter() - t:.1f}s, {len(bad)} suspect frames {bad}")
            return masks, axes, bad

        masks, axes, bad = attempt(pos, np.ones(len(pos), np.int32), "clicks only")
        first_bad, used_wrist = bad, False
        if bad:
            m2, a2, b2 = attempt(np.vstack([pos, wrist]).astype(np.float32), np.r_[np.ones(len(pos)), 0].astype(np.int32),
                                 f"retry with negative wrist point {wrist.tolist()}")
            if len(b2) < len(bad):
                masks, axes, bad, used_wrist = m2, a2, b2, True
            log(f"[{view}] keeping the {'retry with the wrist point' if used_wrist else 'first attempt'}")

    frames_out, rows = {}, []
    os.makedirs(OUT / "overlays" / view, exist_ok=True)
    for f in frames:
        click, ax = vd["frames"][f], axes[f]
        click_ang = line_angle(np.subtract(click["tip"], click["butt"]))
        if ax is None:
            frames_out[f] = None
            rows.append([view, f, click["imu_sample"], round(click_ang, 1), "", "", "", "", "no mask"])
            title = f"{view} {f}: no SAM mask"
        else:
            trusted = ax["elongation"] >= MIN_ELONGATION
            diff = line_diff(ax["angle_deg"], click_ang)
            frames_out[f] = dict(ax, trusted=bool(trusted))
            flag = ("DISAGREE" if abs(diff) > DISAGREE_DEG else "ok") if trusted else "untrusted (elongation < 1.6)"
            rows.append([view, f, click["imu_sample"], round(click_ang, 1), ax["angle_deg"], round(diff, 1),
                         ax["elongation"], ax["area_px"], flag])
            title = f"{view} {f}: SAM - clicks {diff:+.1f} deg, elongation {ax['elongation']}" + ("" if trusted else " (untrusted)")
        cv2.imwrite(str(OUT / "overlays" / view / f"{f}.jpg"), draw_overlay(imgs[f], masks[f], ax, click, title),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    n_trusted = sum(1 for v in frames_out.values() if v and v["trusted"])
    log(f"[{view}] trusted frames {n_trusted}/{len(frames)} (elongation >= {MIN_ELONGATION}), "
        f"view done in {time.perf_counter() - t0:.1f}s")
    return dict(video=vd["video"], prompt_frame=frames[0], prompt_points=names,
                wrist_negative_point=wrist.tolist() if used_wrist else None,
                suspect_frames_first_try=first_bad, suspect_frames=bad, trusted_frames=n_trusted,
                frames=frames_out), rows


def make_log(path):
    fh = open(path, "w", encoding="utf-8")
    def log(msg=""):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sam-dir", default=os.environ.get("SAM2_DIR", str(paths.ROOT.parent / "sam2")),
                    help="SAM 2 clone (default: $SAM2_DIR or ../sam2 next to the repo)")
    ap.add_argument("--checkpoint", help="checkpoint file (default: <sam-dir>/checkpoints/sam2.1_hiera_<model>.pt); "
                                         "must match --model")
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="video frame index = anchor frame number - offset (1 if the anchors are 1-based)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="auto: cuda, then mps, then cpu")
    ap.add_argument("--check-only", action="store_true",
                    help="only draw the clicks on the first frame of each view; no torch or sam2 needed")
    a = ap.parse_args()

    anch = json.load(open(ANCHORS))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check_only:
        for view, vd in anch["views"].items():
            f0 = min(vd["frames"], key=int)
            img = read_frames(paths.VIDEOS / vd["video"], [f0], a.frame_offset)[f0]
            p = save_check(view, f0, img, {k: vd["frames"][f0][k] for k in anch["points"]}, a.frame_offset)
            print(f"wrote {p.relative_to(paths.ROOT)}")
        return

    t_start = time.perf_counter()
    log = make_log(OUT / "run_log.txt")
    log(f"SAM 2 racket-axis run {time.strftime('%Y-%m-%d %H:%M:%S')} on {platform.node()} ({platform.machine()}), "
        f"frame offset {a.frame_offset}")
    try:
        predictor, device, ckpt = load_predictor(a, log)
    except SystemExit as e:                        # missing GPU or checkpoint: say so in run_log.txt too
        log(f"ERROR: {e}")
        raise

    result = dict(model=a.model, checkpoint=ckpt.name, device=device, frame_offset=a.frame_offset,
                  min_elongation=MIN_ELONGATION,
                  angle_convention="undirected line angle in [-90, 90) deg, image x right / y down",
                  views={})
    rows = []
    for view, vd in anch["views"].items():
        result["views"][view], r = run_view(predictor, device, view, vd, anch["points"], a.frame_offset, log)
        rows += r

    json.dump(result, open(OUT / "sam_axes.json", "w"), indent=1)
    with open(OUT / "sam_vs_clicks.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["view", "frame", "imu_sample", "click_angle_deg", "sam_angle_deg", "sam_minus_click_deg",
                     "elongation", "area_px", "flag"])
        wr.writerows(rows)

    for view in result["views"]:
        d = [abs(r[5]) for r in rows if r[0] == view and r[8] in ("ok", "DISAGREE")]
        log(f"[{view}] clicks vs SAM on {len(d)} trusted frames: median {np.median(d):.1f} deg, max {max(d):.1f} deg"
            if d else f"[{view}] no trusted frames")
    for r in rows:
        if r[8] == "DISAGREE":
            log(f"  re-check {r[0]} frame {r[1]}: clicks and SAM differ by {r[5]:+.1f} deg")
    log(f"total {time.perf_counter() - t_start:.1f}s; outputs in {OUT}")


if __name__ == "__main__":
    main()

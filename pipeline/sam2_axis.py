"""
Stage 4b, part 2 (idea #1): automatic racket axis from SAM 2 masks.

For each view:
  1. extract the IMU-window frames from the video as JPEGs
  2. prompt SAM 2 once, on the first frame, with the 4 clicked points from the JSON
  3. let SAM 2 track the racket mask through the rest of the window by itself
  4. PCA on each mask -> 2D long axis angle + an elongation score (confidence)
  5. compare with the clicked butt->tip angle frame by frame

Because SAM only sees clicks on ONE frame, its axes are an independent measurement
for stage 10 (validation), and big disagreements point at bad clicks.

Setup (once):
  git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .
  cd checkpoints && ./download_ckpts.sh        # or fetch sam2.1_hiera_small.pt only
Usage:
  python sam2_axis.py anchors_clean.json --video-dir videos --sam-dir path/to/sam2
Output:
  sam_axes.json, sam_vs_clicks.csv, overlays/<view>/<frame>.png
"""
import argparse, csv, json, os
import numpy as np
import cv2

FRAME_OFFSET = 0     # set to 1 if your frame numbers are 1-based; check overlays/<view>/first_frame_check.png
DISAGREE_DEG = 8.0   # flag frames where clicks and SAM differ by more than this
MIN_ELONGATION = 1.6 # below this the mask is too round (racket face-on / edge-on) to trust its PCA axis


def mask_axis(mask, prev_dir=None):
    """Long axis of a binary mask via PCA. Returns (angle_deg, elongation, centroid, unit_dir)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))   # ascending eigenvalues
    u = evecs[:, 1]
    if prev_dir is not None and u @ prev_dir < 0:         # PCA sign is arbitrary: keep it continuous
        u = -u
    elong = float(np.sqrt(evals[1] / max(evals[0], 1e-9)))
    return float(np.degrees(np.arctan2(u[1], u[0]))), elong, c, u


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def extract_frames(video, frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    for k, f in enumerate(frames):              # SAM 2 wants 00000.jpg, 00001.jpg, ...
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - FRAME_OFFSET)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {f} from {video}")
        cv2.imwrite(os.path.join(out_dir, f"{k:05d}.jpg"), img)
    cap.release()


def run_view(predictor, view, vd, pts_names, video_dir, torch):
    frames = sorted(vd["frames"], key=int)
    jpg_dir = os.path.join("frames", view)
    extract_frames(os.path.join(video_dir, vd["video"]), frames, jpg_dir)

    first = vd["frames"][frames[0]]
    prompt = np.array([first[p] for p in pts_names], np.float32)

    # sanity image: clicks drawn on the extracted first frame (catches off-by-one frame numbering)
    os.makedirs(os.path.join("overlays", view), exist_ok=True)
    img0 = cv2.imread(os.path.join(jpg_dir, "00000.jpg"))
    for x, y in prompt:
        cv2.circle(img0, (int(x), int(y)), 6, (0, 0, 255), 2)
    cv2.imwrite(os.path.join("overlays", view, "first_frame_check.png"), img0)

    masks = {}
    with torch.inference_mode():
        state = predictor.init_state(video_path=jpg_dir)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        points=prompt, labels=np.ones(len(prompt), np.int32))
        for k, _, logits in predictor.propagate_in_video(state):
            masks[frames[k]] = (logits[0] > 0.0).cpu().numpy().squeeze()

    b, t = np.array(first["butt"]), np.array(first["tip"])
    prev = (t - b) / np.linalg.norm(t - b)
    out, rows = {}, []
    for f in frames:
        res = mask_axis(masks[f], prev)
        click = vd["frames"][f]
        cax = np.subtract(click["tip"], click["butt"])
        click_ang = float(np.degrees(np.arctan2(cax[1], cax[0])))
        if res is None:
            out[f] = None
            rows.append([view, f, round(click_ang, 1), "", "", "", "no mask"])
            continue
        ang, elong, c, prev = res
        diff = wrap(ang - click_ang)
        trusted = elong >= MIN_ELONGATION
        flag = "" if not trusted else ("DISAGREE" if abs(diff) > DISAGREE_DEG else "ok")
        if not trusted:
            flag = "mask too round, ignore"
        out[f] = dict(angle_deg=round(ang, 2), elongation=round(elong, 2),
                      centroid=[round(float(v), 1) for v in c], area_px=int(masks[f].sum()),
                      trusted=bool(trusted))
        rows.append([view, f, round(click_ang, 1), round(ang, 1), round(diff, 1), round(elong, 2), flag])

        # overlay: green = SAM axis, red = clicked axis
        img = cv2.imread(os.path.join(jpg_dir, f"{frames.index(f):05d}.jpg"))
        img[masks[f]] = (0.6 * img[masks[f]] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        half = 0.5 * np.linalg.norm(cax)
        p0, p1 = (c - half * prev).astype(int), (c + half * prev).astype(int)
        cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), (0, 255, 0), 2)
        cv2.line(img, tuple(map(int, click["butt"])), tuple(map(int, click["tip"])), (0, 0, 255), 2)
        cv2.putText(img, f"{view} {f}  diff {diff:+.1f} deg", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join("overlays", view, f"{f}.png"), img)
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anchors")
    ap.add_argument("--video-dir", default=".")
    ap.add_argument("--sam-dir", default="sam2")
    ap.add_argument("--model", default="small", choices=["tiny", "small", "base_plus", "large"])
    a = ap.parse_args()

    import torch
    from sam2.build_sam import build_sam2_video_predictor
    short = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}[a.model]
    ckpt = os.path.join(a.sam_dir, "checkpoints", f"sam2.1_hiera_{a.model}.pt")
    cfg = f"configs/sam2.1/sam2.1_hiera_{short}.yaml"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)

    d = json.load(open(a.anchors))
    result, rows = {}, []
    for view, vd in d["views"].items():
        result[view], r = run_view(predictor, view, vd, d["points"], a.video_dir, torch)
        rows += r

    json.dump(result, open("sam_axes.json", "w"), indent=1)
    with open("sam_vs_clicks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "frame", "click_angle_deg", "sam_angle_deg", "diff_deg", "elongation", "flag"])
        w.writerows(rows)
    diffs = [abs(r[4]) for r in rows if r[6] in ("ok", "DISAGREE")]
    if diffs:
        print(f"clicks vs SAM: median {np.median(diffs):.1f} deg, max {max(diffs):.1f} deg over {len(diffs)} trusted frames")
    for r in rows:
        if r[6] == "DISAGREE":
            print(f"  re-check {r[0]} frame {r[1]}: clicks and SAM differ by {r[4]} deg")


if __name__ == "__main__":
    main()

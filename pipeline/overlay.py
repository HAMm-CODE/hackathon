import json, os, cv2, numpy as np, subprocess
import paths; os.chdir(paths.ROOT)
ov = json.load(open("out/overlay_points.json")); anch = json.load(open("data/anchors_clean.json"))
vids = {"front": "data/videos/swing_angle_1.mp4", "rear": "data/videos/swing_angle_2.mp4"}
frames = {v: [] for v in vids}
for v, path in vids.items():
    cap = cv2.VideoCapture(path)
    for f, P, U in zip(ov[v]["frames"], ov[v]["pred"], ov[v]["clicks"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        P = np.array(P); U = np.array(U)
        post = f > anch["views"][v]["impact_frame"]
        col = (60, 160, 255) if post else (80, 220, 80)
        cv2.line(img, tuple(U[0].astype(int)), tuple(U[2].astype(int)), (60, 60, 230), 2)
        cv2.line(img, tuple(P[0].astype(int)), tuple(P[2].astype(int)), col, 3)
        cv2.circle(img, tuple(P[3].astype(int)), 6, col, 2)
        cx, cy = U.mean(0).astype(int); x0 = int(np.clip(cx - 320, 0, 1280 - 640)); y0 = int(np.clip(cy - 200, 0, 720 - 400))
        crop = img[y0:y0 + 400, x0:x0 + 640].copy()
        label = f"{v}  frame {f}  " + ("after impact (low confidence)" if post else "sensor-only racket (green) vs video (red)")
        cv2.rectangle(crop, (0, 0), (640, 34), (0, 0, 0), -1); cv2.putText(crop, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        frames[v].append(crop)
n = min(len(frames["front"]), len(frames["rear"]))
out = cv2.VideoWriter("out/_overlay_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 6, (1280, 400))
for i in range(n): out.write(np.hstack([frames["front"][i], frames["rear"][i]]))
out.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "out/_overlay_raw.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "out/overlay_side_by_side.mp4"])
os.remove("out/_overlay_raw.mp4")
cv2.imwrite("out/overlay_still.jpg", np.vstack([np.hstack([frames["front"][i], frames["rear"][i]]) for i in (2, 8, 13)]))
print("frames", n)

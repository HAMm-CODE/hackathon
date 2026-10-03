"""
SWINQ hackathon: sensor-only racket orientation, with video used only as a referee.

python swing_pipeline.py            -> writes everything into ./out/
Switches (environment variables):
  GYRO_SIGN=+1|-1   gyro rotation convention (+1 = right-handed in the accelerometer frame; default +1)
"""
import json, os
import paths
os.chdir(paths.ROOT)          # all paths below are relative to the repo root
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp
import core
from core import load, skew, smooth_pre, integrate, fit_rigid, predict, FS, G0, IMPACT, CLIP_G
import clipmodels as cm
from fastval import frames_of, rot_at, fit_C, axis_err, proj, body_pts, PTS

OUT = "out"; os.makedirs(OUT, exist_ok=True)
SIGN = float(os.environ.get("GYRO_SIGN", "1"))
GEOM = {"butt": [0.39, 0, 0], "throat": [0.13, 0, 0], "tip": [-0.296, 0, 0], "edge": [-0.12, 0.123, 0], "sweet": [-0.13, 0, 0]}
ANCH = json.load(open("data/anchors_clean.json"))
acc, gyr_raw = load()
gyr = SIGN * gyr_raw
t = np.arange(len(acc)) / FS
R = {}  # results

# ---------------------------------------------------------------- 1. saturation and impact
clipped = np.where(np.abs(acc).max(1) / G0 > CLIP_G)[0]
R["saturation"] = dict(first=int(clipped.min()), last=int(clipped.max()), n=int(len(clipped)), ms=round(len(clipped) / FS * 1e3, 1))
hp = gyr_raw[IMPACT:] - savgol_filter(gyr_raw[IMPACT:], 15, 2, axis=0)
env = np.convolve(np.degrees(np.linalg.norm(hp, axis=1)), np.ones(5) / 5, "same")
bridge_n = int(np.argmax(env < 0.25 * env.max()) if (env < 0.25 * env.max()).any() else 17)
bridge_n = max(bridge_n, np.argmax(env) + 1)
R["impact"] = dict(sample=IMPACT, t_s=round(IMPACT / FS, 4), shock_bridge_samples=bridge_n, shock_bridge_ms=round(bridge_n / FS * 1e3, 1),
                   pre_impact_gyro_noise_dps=round(float(np.percentile(np.degrees(np.linalg.norm(
                       gyr_raw[:IMPACT] - savgol_filter(gyr_raw[:IMPACT], 15, 2, axis=0), axis=1)), 50)), 2))

def robust_gyro(g):
    """Post-impact: smooth the ringing; bridge the shock window detected from the vibration envelope."""
    g = g.copy()
    g[IMPACT:] = savgol_filter(g[IMPACT:], 15, 2, axis=0)
    a, b = IMPACT - 1, IMPACT + bridge_n
    for i in range(3):
        g[a:b + 1, i] = np.linspace(g[a, i], g[b, i], b - a + 1)
    return g

g_rob = robust_gyro(gyr)

# ---------------------------------------------------------------- 2. saturation recovery + stress test
cm.acc, cm.g, cm.al, cm.R = acc, *smooth_pre(gyr), integrate(gyr, 1)
stress = cm.stress((8, 10, 12, 14))
R["stress_test"] = {str(k): {m: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in d.items()} for m, d in v.items()} for k, v in stress.items()}
train = np.arange(0, clipped.min())
x_rb, _ = fit_rigid(acc, cm.g, cm.al, cm.R, train, 1)
recon = predict(x_rb, cm.g, cm.al, cm.R, clipped, 1)
# uncertainty: residual bootstrap of the moving-pivot rigid model
rng = np.random.default_rng(0); boots = []
A = cm.design(cm.g, cm.al, cm.R, train, 1); b = acc[train].reshape(-1); res = A @ x_rb - b
for _ in range(300):
    xb, *_ = np.linalg.lstsq(A, A @ x_rb + rng.choice(res, len(res)), rcond=None)
    boots.append(predict(xb, cm.g, cm.al, cm.R, clipped, 1)[:, 0] / G0)
boots = np.array(boots)
model_err = stress[8]["rigid body + moving pivot"]["rmse"]
peak = recon[:, 0].max() / G0
R["saturation_recovery"] = dict(model="rigid body + moving pivot (gyro drives it; gyro was never clipped)",
                                peak_ax_g=round(float(peak), 1),
                                peak_range_g=[round(float(np.percentile(boots.max(1), 5) - model_err), 1),
                                              round(float(np.percentile(boots.max(1), 95) + model_err), 1)],
                                at_sample=int(clipped[recon[:, 0].argmax()]))
acc_rec = acc.copy(); acc_rec[clipped, 0] = np.maximum(recon[:, 0], acc[clipped, 0])

# ---------------------------------------------------------------- 3. orientation (sensor only)
gs, al = smooth_pre(g_rob)
Rrel = integrate(g_rob, 1)
idx = np.array([k for k in range(IMPACT) if k not in set(clipped)])
x, res = fit_rigid(acc, gs, al, Rrel, idx, 1)
p = x[:3]; w = x[3:6]
up0 = w / np.linalg.norm(w)
xh = np.cross([0, 1, 0], up0); xh /= np.linalg.norm(xh); yh = np.cross(up0, xh)
R0 = Rot.from_matrix(np.vstack([xh, yh, up0]))
Rw = R0 * Rrel
k = IMPACT - 1
v_ss = Rw[k].apply(np.cross(gs[k], np.array(GEOM["sweet"]) + p))   # lever arm from the pivot (pivot = -p in sensor frame)
Rw = Rot.from_euler("z", -np.arctan2(v_ss[1], v_ss[0])) * Rw          # heading convention: +x = swing direction at contact
R["orientation"] = dict(gyro_sign=SIGN, pivot_to_sensor_m=np.round(p, 3).tolist(), pivot_radius_m=round(float(np.linalg.norm(p)), 3),
                        gravity_fit_norm_g=round(float(np.linalg.norm(w) / G0), 2),
                        note="heading is unobservable without a magnetometer: +x is defined as the swing direction at contact")

# ---------------------------------------------------------------- 4. metrics
def speed_at(k, pt):
    return float(np.linalg.norm(np.cross(gs[k], np.array(GEOM[pt]) + p)))
wmag = np.degrees(np.linalg.norm(gyr, axis=1))
kpk = int(np.argmax(wmag[:IMPACT + 1]))
face = Rw[k].apply([0, 0, 1]); axis_tip = Rw[k].apply([-1, 0, 0])
v_imp = Rw[k].apply(np.cross(gs[k], np.array(GEOM["sweet"]) + p))
roll_pre = float(np.degrees(np.sum(gyr[:IMPACT, 0]) / FS))
rot_pre = float(np.degrees(Rrel[IMPACT - 1].magnitude()))
R["metrics"] = dict(
    peak_angular_speed_dps=round(float(wmag[kpk]), 0), peak_at_ms_before_contact=round((IMPACT - kpk) / FS * 1e3, 1),
    angular_speed_at_contact_dps=round(float(wmag[k]), 0),
    rotational_head_speed_at_contact_kmh=round(speed_at(k, "sweet") * 3.6, 0),
    rotational_tip_speed_at_contact_kmh=round(speed_at(k, "tip") * 3.6, 0),
    head_speed_note="speed from the racket rotating about the hand (fitted pivot); the hand's own forward speed adds on top and is not observable without drift",
    rotation_window_start_to_contact_deg=round(rot_pre, 0),
    roll_about_long_axis_to_contact_deg=round(roll_pre, 0),
    forward_swing_duration_ms=round((IMPACT - int(np.argmax(wmag[:IMPACT] > 0.3 * wmag[kpk]))) / FS * 1e3, 0),
)

R["gravity_dependent_metrics_NOT_RELIABLE"] = dict(
    reason="no rest period in the 0.96 s window, so gravity cannot be separated from hand acceleration; the upright-camera check failed",
    face_tilt_at_contact_deg=round(float(np.degrees(np.arcsin(face[2]))), 1),
    racket_axis_elevation_at_contact_deg=round(float(np.degrees(np.arcsin(axis_tip[2]))), 1),
    swing_path_elevation_at_contact_deg=round(float(np.degrees(np.arcsin(v_imp[2] / np.linalg.norm(v_imp)))), 1))

# experimental: impact offset from the gyro step (free-body impulse; ratio cancels the unknown ball impulse)
pre = np.arange(IMPACT - 15, IMPACT); post = np.arange(IMPACT + 2, IMPACT + 14)
def step(a):
    c1 = np.polyfit(pre, gyr[pre, a], 1); c2 = np.polyfit(post, gyr[post, a], 1)
    return np.polyval(c2, IMPACT + 1) - np.polyval(c1, IMPACT + 1)
dw = np.array([step(a) for a in range(3)])
rng = np.random.default_rng(1)
Ix = rng.uniform(0.0012, 0.0020, 5000); Iy = rng.uniform(0.014, 0.020, 5000); xcm = rng.uniform(0.15, 0.22, 5000)
yoff = -(-xcm) * (dw[0] / dw[1]) * (Ix / Iy)          # x_p is negative (impact is toward the tip from the CM)
R["impact_location_experimental"] = dict(gyro_step_dps=np.round(np.degrees(dw), 0).tolist(),
                                         across_face_offset_cm=round(float(np.median(yoff) * 100), 1),
                                         range_90_cm=[round(float(np.percentile(yoff, 5) * 100), 1), round(float(np.percentile(yoff, 95) * 100), 1)],
                                         note="free-body impulse model, sign of +Y side; the impact shock also disturbs the gyro, so treat as indicative")

# ---------------------------------------------------------------- 5. noise robustness (Monte Carlo)
def orient_from(gyro_meas):
    gg = robust_gyro(gyro_meas); return integrate(gg, 1)
nom = Rrel
errs_imp, errs_end, speeds = [], [], []
for i in range(120):
    noisy = gyr + rng.normal(0, np.radians(5), gyr.shape) + rng.uniform(-np.radians(3), np.radians(3), 3)
    Rn = orient_from(noisy)
    errs_imp.append(np.degrees((Rn[IMPACT - 1] * nom[IMPACT - 1].inv()).magnitude()))
    errs_end.append(np.degrees((Rn[-1] * nom[-1].inv()).magnitude()))
    gsn, _ = smooth_pre(robust_gyro(noisy))
    speeds.append(np.linalg.norm(np.cross(gsn[k], np.array(GEOM["sweet"]) + p)) * 3.6)
R["noise_test"] = dict(added="white 5 deg/s + random bias up to 3 deg/s per axis, 120 runs",
                       orientation_error_at_contact_deg=dict(median=round(float(np.median(errs_imp)), 2), p95=round(float(np.percentile(errs_imp, 95)), 2)),
                       orientation_error_at_end_deg=dict(median=round(float(np.median(errs_end)), 2), p95=round(float(np.percentile(errs_end, 95)), 2)),
                       head_speed_spread_kmh=round(float(np.std(speeds)), 2))

# ---------------------------------------------------------------- 6. video as referee
val = {}; overlay = {}; CAM = {}
for view in ("front", "rear"):
    fr, smp, uv = frames_of(ANCH, view)
    n = len(fr); prem = np.array(smp) < IMPACT
    best = None
    for sh in np.arange(-14, 14.1, 3.5):
        Rt = rot_at(Rw, np.array(smp) + sh)
        for es in (1, -1):
            r = fit_C(Rt[prem], uv[prem], es, starts=8, seed=int(sh * 10) % 97)
            if best is None or r.cost < best[0].cost:
                best = (r, es, sh)
    r, es, sh = best
    Rt = rot_at(Rw, np.array(smp) + sh)
    err, px, P = axis_err(Rt, uv, r.x, es)
    C = Rot.from_rotvec(r.x[:3]).as_matrix()
    CAM[view] = dict(C=C, shift=sh, scale=float(np.exp(r.x[3])))
    up_img = (C @ [0, 0, 1])[:2]
    roll = float(np.degrees(np.arctan2(up_img[0], -up_img[1])))
    val[view] = dict(time_offset_samples=float(sh), edge_side=es,
                     forward_swing=dict(frames=int(prem.sum()), mean_axis_err_deg=round(float(np.abs(err[prem]).mean()), 1),
                                        median=round(float(np.median(np.abs(err[prem]))), 1), max=round(float(np.abs(err[prem]).max()), 1),
                                        mean_point_err_px=round(float(px[prem].mean()), 1)),
                     follow_through_predicted=dict(frames=int((~prem).sum()), mean_axis_err_deg=round(float(np.abs(err[~prem]).mean()), 1)),
                     camera_roll_from_sensor_gravity_deg=round(roll, 1),
                     per_frame=[dict(frame=int(f), imu_sample=float(s), axis_err_deg=round(float(e), 1), px=round(float(q), 1))
                                for f, s, e, q in zip(fr, smp, err, px)])
    overlay[view] = dict(frames=[int(f) for f in fr], pred=P.round(1).tolist(), clicks=uv.tolist())
R["validation"] = val

json.dump(overlay, open(f"{OUT}/overlay_points.json", "w"))

# ---------------------------------------------------------------- 7. exports
pts = {kk: np.array(vv) for kk, vv in GEOM.items()}
kf = dict(sample_rate_hz=FS, impact_sample=IMPACT, shock_bridge=[IMPACT, IMPACT + bridge_n], gyro_sign=SIGN,
          geometry_m=GEOM, pivot_to_sensor_m=p.tolist(), frames=[])
D = np.diag([1, -1, -1]) @ CAM["front"]["C"]          # world -> display (three.js: x right, y up, camera on +z) = front video view
kf["display_frame"] = "front camera view (fitted camera rotation only; used for display, not for the reconstruction)"
for i in range(len(acc)):
    Ri = Rot.from_matrix(D) * Rw[i]
    kf["frames"].append(dict(i=i, t=round(i / FS, 5), q_xyzw=np.round(Ri.as_quat(), 6).tolist(),
                             omega_dps=round(float(np.degrees(np.linalg.norm(gyr[i]))), 1),
                             tip=np.round(Ri.apply(pts["tip"] + p), 4).tolist(), butt=np.round(Ri.apply(pts["butt"] + p), 4).tolist(),
                             sweet_speed_kmh=round(float(np.linalg.norm(np.cross(gs[i], pts["sweet"] + p)) * 3.6), 1),
                             confidence="high" if i < IMPACT else ("low" if i <= IMPACT + bridge_n else "medium"),
                             acc_g=np.round(acc_rec[i] / G0, 3).tolist(), acc_reconstructed=bool(i in set(clipped))))
kf["cameras_display"] = {}
for view in CAM:   # camera looking direction and up vector expressed in the display frame
    M = D @ CAM[view]["C"].T
    kf["cameras_display"][view] = dict(forward=(M @ [0, 0, 1]).round(4).tolist(), up=(-(M @ [0, 1, 0])).round(4).tolist())
json.dump(kf, open(f"{OUT}/keyframes.json", "w"))
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)
np.savez(f"{OUT}/arrays.npz", acc=acc, acc_rec=acc_rec, gyr=gyr, g_rob=g_rob, clipped=clipped, boots=boots, env=env)
print(json.dumps({kk: vv for kk, vv in R.items() if kk != "validation"}, indent=1))
for v in val: print(v, {kk: vv for kk, vv in val[v].items() if kk != "per_frame"})

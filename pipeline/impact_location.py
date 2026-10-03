"""
Stage 2b + 11b (idea #2): where on the strings did the ball hit?

Gyro estimate
  An off-centre hit twists the racket about its long axis (sensor X). The instant
  change in that spin is   d_omega_x = - d * J / I_x
    d   = offset of the ball across the face (along Y), metres
    J   = ball impulse = m_ball * delta_v_ball   (N s)
    I_x = racket twisting inertia about the long axis (kg m^2)
  (sign shown for a ball striking the +Z face; it flips for the other face)
  We measure d_omega by extrapolating the smooth pre-impact gyro across contact and
  comparing with the mean of two samples just after contact. Averaging two consecutive
  samples cancels most of the string ringing, which sits near the Nyquist frequency.
  I_x and delta_v are uncertain, so d is reported as a Monte Carlo range.

Video estimate (independent)
  At the impact frame, the ball's perpendicular distance from the clicked butt->tip
  line, scaled by the clicked head-edge distance (= half the head width), gives the
  offset in metres and which side of the axis it was on. Also gives how far up the
  head (towards tip or throat) the contact was.

Usage:
  python impact_location.py raw_data.csv anchors_clean.json \
      --ball-front 702 598 --ball-rear 655 410 [--rm rm.json]
  (--ball-* = ball centre in pixels at the impact frame, from your ball tracker)
  (--rm = 3x3 sensor-to-racket rotation from the stage 7 joint fit, once you have it)
"""
import argparse, json
import numpy as np
import pandas as pd

M_BALL = 0.057                  # kg
DV_RANGE = (35.0, 60.0)         # m/s, change in ball speed through contact (in + out)
IX_RANGE = (0.0012, 0.0020)     # kg m^2, typical racket polar moment
HEAD_HALF_WIDTH = 0.13          # m, half of a ~26 cm head; 'edge' click = widest point of the head
PRE = 15                        # samples used for the pre-impact trend (36 ms)
POST = (2, 3)                   # samples after impact to average (contact lasts ~4-5 ms = 2 samples)


def load_gyro(path, cols):
    df = pd.read_csv(path)
    if cols:
        g = df[cols].to_numpy(float)
    else:
        named = [c for c in df.columns if c.lower().strip() in ("gx", "gy", "gz", "gyro_x", "gyro_y", "gyro_z")]
        g = df[named].to_numpy(float) if len(named) == 3 else df.select_dtypes("number").iloc[:, 3:6].to_numpy(float)
    return np.radians(g)          # deg/s -> rad/s


def gyro_step(g, impact):
    t_pre = np.arange(impact - PRE, impact)
    t_post = np.array([impact + k for k in POST])
    step = np.empty(3)
    for a in range(3):
        coef = np.polyfit(t_pre, g[t_pre, a], 1)          # local linear trend before contact
        step[a] = g[t_post, a].mean() - np.polyval(coef, t_post).mean()
    return step


def offset_from_twist(d_omega_x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    J = M_BALL * rng.uniform(*DV_RANGE, n)
    Ix = rng.uniform(*IX_RANGE, n)
    d = -d_omega_x * Ix / J
    return np.percentile(d, [5, 50, 95])


def offset_from_video(fr, ball):
    b, t, e = (np.array(fr[k], float) for k in ("butt", "tip", "edge"))
    ball = np.array(ball, float)
    u = (t - b) / np.linalg.norm(t - b)
    perp = lambda p: u[0] * (p - b)[1] - u[1] * (p - b)[0]   # signed 2D distance from the axis
    edge_px, ball_px = perp(e), perp(ball)
    across = ball_px / edge_px * HEAD_HALF_WIDTH            # + = same side as the clicked edge
    along = (ball - b) @ u / np.linalg.norm(t - b)          # 0 = butt, 1 = tip
    return across, along, abs(edge_px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imu_csv")
    ap.add_argument("anchors")
    ap.add_argument("--gyro-cols", nargs=3, default=None)
    ap.add_argument("--ball-front", nargs=2, type=float)
    ap.add_argument("--ball-rear", nargs=2, type=float)
    ap.add_argument("--rm", default=None, help="json file holding a 3x3 sensor-to-racket rotation")
    a = ap.parse_args()

    d = json.load(open(a.anchors))
    impact = d["impact_sample"]
    g = load_gyro(a.imu_csv, a.gyro_cols)
    step = gyro_step(g, impact)
    frame_name = "sensor"
    if a.rm:
        step = np.array(json.load(open(a.rm)), float) @ step
        frame_name = "racket"
    lo, mid, hi = offset_from_twist(step[0])

    print(f"Gyro step at contact ({frame_name} frame), rad/s: x {step[0]:+.2f}  y {step[1]:+.2f}  z {step[2]:+.2f}")
    print(f"  = {np.degrees(step[0]):+.0f} deg/s twist about the long axis")
    print(f"Gyro offset across the face: {mid*100:+.1f} cm  (90% range {lo*100:+.1f} to {hi*100:+.1f} cm)")
    print("  sign assumes the ball hit the +Z face; grip damping makes this a slight underestimate")

    for view, ball in (("front", a.ball_front), ("rear", a.ball_rear)):
        if ball is None:
            continue
        vd = d["views"][view]
        across, along, edge_px = offset_from_video(vd["frames"][str(vd["impact_frame"])], ball)
        side = "edge side" if across > 0 else "far side"
        print(f"Video ({view}): {abs(across)*100:.1f} cm to the {side} of the axis, "
              f"{along:.2f} of the way from butt to tip "
              f"(head half-width visible as {edge_px:.0f} px; trust the view where this is larger)")


if __name__ == "__main__":
    main()

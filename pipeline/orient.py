"""Sensor-only orientation: gyro integration + gravity tilt + heading convention."""
from core import *
from clipmodels import m_rigid

# racket geometry in the sensor frame (metres). X toward butt, Y across face, Z out of face.
GEOM = {"butt": [0.39, 0, 0], "throat": [0.13, 0, 0], "tip": [-0.296, 0, 0],
        "edge": [-0.12, 0.123, 0], "sweet": [-0.13, 0, 0]}

def sensor_only(sign=1.0, bias=None, acc=None, gyr=None):
    if acc is None:
        acc, gyr = load()
    g, al = smooth_pre(sign * gyr)
    Rrel = integrate(gyr, sign, bias)
    clipped = set(np.where(np.abs(acc).max(1) / G0 > CLIP_G)[0])
    idx = np.array([k for k in range(IMPACT) if k not in clipped])
    x, res = fit_rigid(acc, g, al, Rrel, idx, pivot_terms=1)
    p, w = x[:3], x[3:6]
    up0 = w / np.linalg.norm(w)                 # world 'up' expressed in the body frame at sample 0
    # R0 maps body(0) -> world with world z = up
    z = up0
    xh = np.cross([0, 1, 0], z); xh /= np.linalg.norm(xh)
    yh = np.cross(z, xh)
    R0 = Rot.from_matrix(np.vstack([xh, yh, z]))   # rows = world axes in body coords -> body->world
    R = R0 * Rrel
    # heading convention: world +x = horizontal direction of sweet-spot velocity at impact
    k = IMPACT - 1
    v = R[k].apply(np.cross(g[k], np.array(GEOM["sweet"]) - p))   # velocity about the fitted pivot
    yaw = -np.arctan2(v[1], v[0])
    R = Rot.from_euler("z", yaw) * R
    return dict(R=R, p=p, w=w, g=g, alpha=al, res=res, gravity_norm_g=np.linalg.norm(w) / G0)

if __name__ == "__main__":
    s = sensor_only()
    print("pivot->sensor p (m):", s["p"].round(3), "|w|/g:", round(s["gravity_norm_g"], 2))
    R = s["R"]
    for k in (0, 100, 150, 199):
        print(k, "long axis (to tip):", R[k].apply([-1, 0, 0]).round(2), "face normal:", R[k].apply([0, 0, 1]).round(2))

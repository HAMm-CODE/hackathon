import json
from scipy.optimize import least_squares
from scipy.spatial.transform import Slerp
from orient import *
PTS = ["butt", "throat", "tip", "edge"]
def wrap(a): return (a + 180) % 360 - 180
def frames_of(anch, view):
    vd = anch["views"][view]; fr = sorted(vd["frames"], key=int)
    s = np.array([vd["frames"][f]["imu_sample"] for f in fr])
    uv = np.array([[vd["frames"][f][p] for p in PTS] for f in fr], float)
    return fr, s, uv
def rot_at(R, samples):
    return Slerp(np.arange(len(R)), R)(np.clip(samples, 0, len(R) - 1)).as_matrix()
def body_pts(es):
    b = np.array([GEOM[p] for p in PTS], float); b[3, 1] *= es; return b
def proj(Rt, C, scale, body):
    w = np.einsum("nij,pj->npi", Rt, body)          # world points (n,4,3)
    cam = w @ C.T
    return scale * cam[..., :2]
def res_fn(p, Rt, uv, body):
    P = proj(Rt, Rot.from_rotvec(p[:3]).as_matrix(), np.exp(p[3]), body)
    P = P + (uv - P).mean(1, keepdims=True)
    return (P - uv).ravel()
def fit_C(Rt, uv, es, starts=20, seed=0, x0s=None):
    rng = np.random.default_rng(seed); body = body_pts(es); best = None
    L = np.linalg.norm(uv[:, 2] - uv[:, 0], axis=1).max()
    inits = x0s if x0s is not None else [np.r_[Rot.random(random_state=rng.integers(1e9)).as_rotvec(), np.log(L / .686)] for _ in range(starts)]
    for x0 in inits:
        r = least_squares(res_fn, x0, args=(Rt, uv, body), loss="soft_l1", f_scale=8)
        if best is None or r.cost < best.cost: best = r
    return best
def axis_err(Rt, uv, x, es):
    P = proj(Rt, Rot.from_rotvec(x[:3]).as_matrix(), np.exp(x[3]), body_pts(es)); P = P + (uv - P).mean(1, keepdims=True)
    ang = lambda Q: np.degrees(np.arctan2(Q[:, 2, 1] - Q[:, 0, 1], Q[:, 2, 0] - Q[:, 0, 0]))
    return wrap(ang(P) - ang(uv)), np.linalg.norm(P - uv, axis=2).mean(1), P

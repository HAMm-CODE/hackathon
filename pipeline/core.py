import numpy as np, pandas as pd
from paths import DATA
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

FS = 416.0; G0 = 9.80665; IMPACT = 200; CLIP_G = 15.95

def load(path=None):
    path = path or DATA / "raw_data.csv"
    d = pd.read_csv(path)
    acc = d[["ax", "ay", "az"]].to_numpy(float) * G0          # m/s^2
    gyr = np.radians(d[["gx", "gy", "gz"]].to_numpy(float))   # rad/s
    return acc, gyr

def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

def smooth_pre(gyr, win=9, poly=3):
    """Savitzky-Golay on the pre-impact part only (never smooth across impact)."""
    g = gyr.copy()
    g[:IMPACT] = savgol_filter(gyr[:IMPACT], win, poly, axis=0)
    g[IMPACT:] = savgol_filter(gyr[IMPACT:], win, poly, axis=0)
    alpha = np.zeros_like(g)
    alpha[:IMPACT] = savgol_filter(gyr[:IMPACT], win, poly, deriv=1, delta=1/FS, axis=0)
    alpha[IMPACT:] = savgol_filter(gyr[IMPACT:], win, poly, deriv=1, delta=1/FS, axis=0)
    return g, alpha

def integrate(gyr, sign=1.0, bias=None):
    """Body-to-world rotations relative to sample 0 (R_rel[0] = I). Exact step with mean rate."""
    w = sign * gyr - (0 if bias is None else bias)
    R = [Rot.identity()]
    for k in range(1, len(w)):
        dq = Rot.from_rotvec(0.5 * (w[k - 1] + w[k]) / FS)
        R.append(R[-1] * dq)          # body-frame increment (right multiply)
    return Rot.concatenate(R)

def design(omega, alpha, Rrel, idx, pivot_terms=0):
    """Rows for f = (W^2 + A) p + Rrel^T w  [+ Rrel^T (v1 t)]  -> unknown x = [p, w, (v1)]."""
    rows, Rm = [], Rrel.as_matrix()
    for k in idx:
        W = skew(omega[k]); A = skew(alpha[k])
        blocks = [W @ W + A, Rm[k].T]
        if pivot_terms:
            blocks.append(Rm[k].T * (k / FS))
        rows.append(np.hstack(blocks))
    return np.vstack(rows)

def fit_rigid(acc, omega, alpha, Rrel, idx, pivot_terms=0):
    A = design(omega, alpha, Rrel, idx, pivot_terms)
    b = acc[idx].reshape(-1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = (A @ x - b).reshape(-1, 3)
    return x, res

def predict(x, omega, alpha, Rrel, idx, pivot_terms=0):
    return (design(omega, alpha, Rrel, idx, pivot_terms) @ x).reshape(-1, 3)

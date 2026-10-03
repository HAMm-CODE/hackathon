from core import *
import os
SIGN = float(os.environ.get("GYRO_SIGN", "1"))
acc, gyr = load(); g, al = smooth_pre(SIGN * gyr); R = integrate(gyr, SIGN)
REAL = np.arange(182, 200)

def m_rigid(train, test, pt):
    x, _ = fit_rigid(acc, g, al, R, train, pt)
    return predict(x, g, al, R, test, pt)[:, 0]

def m_centripetal(train, test, lo_win=60):
    tr = train[train >= train.max() - lo_win]
    feat = lambda k: np.column_stack([np.ones(len(k)), k / FS, g[k, 1]**2 + g[k, 2]**2, al[k, 2], al[k, 1]])
    c, *_ = np.linalg.lstsq(feat(tr), acc[tr, 0], rcond=None)
    return feat(test) @ c

def m_poly(train, test, deg=2, n=15):
    tr = train[-n:]
    c = np.polyfit(tr, acc[tr, 0], deg)
    return np.polyval(c, test)

MODELS = {"rigid body, fixed pivot": lambda tr, te: m_rigid(tr, te, 0),
          "rigid body + moving pivot": lambda tr, te: m_rigid(tr, te, 1),
          "centripetal regression (local)": m_centripetal,
          "polynomial extrapolation (no physics)": m_poly}

def stress(levels=(8, 10, 12, 14)):
    out = {}
    pre = np.arange(0, 182)
    for c in levels:
        first = np.where(np.abs(acc[pre, 0]) / G0 > c)[0]
        start = first[0]
        test = np.arange(start, 182)          # artificially clipped samples, true values known
        train = np.arange(0, start)
        truth = acc[test, 0] / G0
        out[c] = {}
        for name, f in MODELS.items():
            pred = f(train, test) / G0
            out[c][name] = dict(n=len(test), rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
                                max_err=float(np.max(np.abs(pred - truth))))
    return out

if __name__ == "__main__":
    s = stress()
    for c, r in s.items():
        print(f"clip at {c} g:")
        for k, v in r.items():
            print(f"   {k:40s} n={v['n']:3d} rmse {v['rmse']:5.2f} g  max {v['max_err']:5.2f} g")
    train = np.arange(0, 182)
    for name, f in MODELS.items():
        p = f(train, REAL) / G0
        print(f"REAL gap  {name:40s} peak {p.max():5.1f} g at sample {REAL[p.argmax()]}")

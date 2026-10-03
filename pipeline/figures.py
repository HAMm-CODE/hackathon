import json, os, numpy as np, matplotlib
import paths; os.chdir(paths.ROOT)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import FS, G0, IMPACT
r = json.load(open("out/results.json")); a = np.load("out/arrays.npz")
t = (np.arange(400) - IMPACT) / FS * 1000
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
cl = a["clipped"]; boots = a["boots"]
ax[0].plot(t, a["acc"][:, 0] / G0, color="#444", lw=1.4, label="measured ax (clipped at 16 g)")
ax[0].fill_between(t[cl], np.percentile(boots, 5, 0), np.percentile(boots, 95, 0), color="#D85A30", alpha=.25, label="reconstruction 90% band")
ax[0].plot(t[cl], a["acc_rec"][cl, 0] / G0, color="#D85A30", lw=2, label="physics reconstruction")
ax[0].axhline(15.96, ls=":", color="#999"); ax[0].set_ylabel("long-axis accel (g)"); ax[0].legend(frameon=False, loc="upper left")
ax[0].set_title("Saturation: 18 samples (43 ms) clipped, recovered from the unclipped gyro")
w = np.degrees(np.linalg.norm(a["gyr"], axis=1)); wr = np.degrees(np.linalg.norm(a["g_rob"], axis=1))
ax[1].plot(t, w, color="#bbb", lw=1, label="raw gyro |ω|")
ax[1].plot(t, wr, color="#185FA5", lw=2, label="robust gyro (shock bridged)")
b = r["impact"]["shock_bridge_samples"]
ax[1].axvspan(0, b / FS * 1000, color="#D85A30", alpha=.12, label=f"impact shock window ({r['impact']['shock_bridge_ms']} ms)")
ax[1].axvline(0, color="#D85A30", lw=1); ax[1].set_ylabel("angular speed (°/s)"); ax[1].set_xlabel("time from contact (ms)")
ax[1].legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("out/fig1_sensor.png", dpi=160)

st = r["stress_test"]; levels = list(st); models = list(st[levels[0]])
fig, ax = plt.subplots(figsize=(10, 4.2)); wdt = 0.2
cols = ["#B4B2A9", "#D85A30", "#7F77DD", "#5F5E5A"]
for i, m in enumerate(models):
    vals = [st[l][m]["rmse"] for l in levels]
    ax.bar(np.arange(len(levels)) + (i - 1.5) * wdt, vals, wdt, label=m, color=cols[i])
ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"clip at {l} g\n({st[l][models[0]]['n']} samples hidden)" for l in levels])
ax.set_ylabel("RMSE on hidden samples (g)"); ax.set_ylim(0, 10); ax.legend(frameon=False, ncol=2)
ax.set_title("Stress test: we clip good data on purpose and check who recovers the truth (centripetal bar at 8 g = 28 g, off scale)")
fig.tight_layout(); fig.savefig("out/fig2_stress_test.png", dpi=160)

fig, ax = plt.subplots(figsize=(10, 4))
for v, c in (("front", "#185FA5"), ("rear", "#1D9E75")):
    pf = r["validation"][v]["per_frame"]
    tt = [(q["imu_sample"] - IMPACT) / FS * 1000 for q in pf]; e = [abs(q["axis_err_deg"]) for q in pf]
    ax.plot(tt, e, "o-", color=c, ms=4, label=f"{v} view")
ax.axvline(0, color="#D85A30"); ax.axvspan(-500, 0, color="#1D9E75", alpha=.06)
ax.text(-460, 160, "forward swing: sensor-only matches video", color="#0F6E56"); ax.text(20, 160, "after impact: gyro disturbed by shock", color="#993C1D")
ax.set_ylabel("racket axis angle error (°)"); ax.set_xlabel("time from contact (ms)"); ax.set_ylim(0, 180); ax.legend(frameon=False, loc="center left")
ax.set_title("Video as referee: only camera pose and a ±1 frame sync offset are fitted")
fig.tight_layout(); fig.savefig("out/fig3_validation.png", dpi=160)
print("ok")

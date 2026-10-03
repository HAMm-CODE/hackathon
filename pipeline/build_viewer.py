"""Rebuild viewer/swing_viewer.html from out/keyframes.json and out/results.json."""
import json
import paths

k = json.load(open(paths.OUT / "keyframes.json")); r = json.load(open(paths.OUT / "results.json"))
F = k["frames"]; m = r["metrics"]; v = r["validation"]; st = r["stress_test"]["8"]
data = dict(fs=k["sample_rate_hz"], impact=k["impact_sample"], bridge=k["shock_bridge"], p=[round(x, 4) for x in k["pivot_to_sensor_m"]],
            geom=k["geometry_m"], cams=k["cameras_display"],
            q=[f["q_xyzw"] for f in F], tip=[f["tip"] for f in F], w=[f["omega_dps"] for f in F],
            v=[f["sweet_speed_kmh"] for f in F], conf=[{"high": 2, "medium": 1, "low": 0}[f["confidence"]] for f in F],
            rec=[1 if f["acc_reconstructed"] else 0 for f in F],
            summary=dict(peak=m["peak_angular_speed_dps"], rot=m["rotation_window_start_to_contact_deg"], roll=m["roll_about_long_axis_to_contact_deg"],
                         dur=m["forward_swing_duration_ms"], head=m["rotational_head_speed_at_contact_kmh"], tipv=m["rotational_tip_speed_at_contact_kmh"],
                         satpk=r["saturation_recovery"]["peak_ax_g"], satr=r["saturation_recovery"]["peak_range_g"], satms=r["saturation"]["ms"],
                         st_phys=st["rigid body + moving pivot"]["rmse"], st_poly=st["polynomial extrapolation (no physics)"]["rmse"],
                         noise=r["noise_test"]["orientation_error_at_contact_deg"]["median"], noise95=r["noise_test"]["orientation_error_at_contact_deg"]["p95"],
                         vf=v["front"]["forward_swing"]["median"], vr=v["rear"]["forward_swing"]["median"],
                         vfm=v["front"]["forward_swing"]["mean_axis_err_deg"], vrm=v["rear"]["forward_swing"]["mean_axis_err_deg"],
                         shock=r["impact"]["shock_bridge_ms"]))
html = (paths.VIEWER / "template.html").read_text()
html = html.replace("__DATA__", "const SWING=" + json.dumps(data, separators=(",", ":")) + ";").replace("__SHOCK__", str(round(r["impact"]["shock_bridge_ms"])))
(paths.VIEWER / "swing_viewer.html").write_text(html)
print("wrote", paths.VIEWER / "swing_viewer.html")

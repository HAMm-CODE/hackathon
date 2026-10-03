# SWINQ swing reconstruction

Sensor-only 3D orientation of a tennis racket through one forehand, from the SWINQ dampener IMU.
The two videos are only a referee: the check fits each camera's angle, its scale, and a sync offset of at most one frame.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run_all.sh                                            # about 1 to 2 minutes
```

Then open `viewer/swing_viewer.html` in a browser (it loads three.js from a CDN, so it needs internet).

### Blender render (tested with Blender 5.0.1)

```bash
blender -b -P blender/render_racket.py -- --preview     # 4 PNG stills in out/, check the framing first
blender -b -P blender/render_racket.py                  # out/racket_render.mp4, about 4x slow motion
blender -b -P blender/render_racket.py -- --view rear --engine CYCLES --samples 32
blender -P blender/render_racket.py -- --no-render      # open the built scene in the Blender UI
```

On macOS the binary is `/Applications/Blender.app/Contents/MacOS/Blender`; on Windows `"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"`.
EEVEE takes a few seconds per frame on a laptop GPU; use `--engine WORKBENCH` for a fast grey draft.

## Repo layout

```
swinq-swing-reconstruction/
  README.md            this file
  CLAUDE.md            context for Claude Code
  requirements.txt
  run_all.sh           rebuild every output from data/
  data/
    raw_data.csv       IMU: 400 samples, accel in g, gyro in deg/s, 416 Hz
    anchors.json       team's clicked racket points (both videos, 58 frames)
    anchors_clean.json after anchor_qa.py (front frame 118 relabelled)
    videos/            swing_angle_1.mp4 (front), swing_angle_2.mp4 (rear)
  pipeline/
    swing_pipeline.py  main pipeline: saturation, orientation, metrics, stress and noise tests, video check
    core.py            loading, filtering, quaternion integration, rigid-body model
    clipmodels.py      saturation-recovery models and the stress test
    fastval.py         weak-perspective camera fit used by the video check
    orient.py          racket geometry and an older orientation helper
    figures.py         the three slide figures
    overlay.py         side-by-side overlay video
    build_viewer.py    fills viewer/template.html with the latest results
    anchor_qa.py       finds and fixes bad clicks
    sam2_axis.py       racket axis per frame from SAM 2 masks (independent check; GPU, see hpc/roihu/)
    validate_sam.py    clicks vs SAM 2 and sensor vs SAM 2, figure 4 (laptop, CPU)
    impact_location.py earlier impact-location script with a video cross-check
    paths.py           repo-relative paths
  hpc/roihu/           SAM 2 setup and Slurm job for CSC Roihu (GH200)
  blender/
    render_racket.py   Blender scene and render from out/keyframes.json
  viewer/
    template.html      viewer source
    swing_viewer.html  generated, self-contained viewer
  out/                 generated results (committed so the demo works without rerunning)
  docs/
    pitch.md           three-minute pitch
```

## Results (default: right-handed gyro)

| What | Result |
|---|---|
| Accelerometer saturation | 18 samples (43.3 ms) clipped at 16 g; gyro never clipped |
| Recovered true peak | 19.3 g (likely 16.9 to 21.7 g) |
| Stress test (hide 77 ms of good data) | physics 1.59 g error, curve fitting 7.36 g, centripetal-only 28.27 g |
| Noise test (5 deg/s noise + 3 deg/s bias) | 1.04 deg change at contact (worst 5%: 1.71 deg) |
| Forward swing vs clicked video points, front | median 2.2 deg, mean 4.4 deg, max 27.4 deg |
| Forward swing vs clicked video points, rear | median 2.1 deg, mean 6.0 deg, max 38.7 deg |
| Forward swing vs SAM 2 axis (independent), front | median 5.0 deg, mean 6.9 deg, max 24.1 deg |
| Forward swing vs SAM 2 axis (independent), rear | median 4.3 deg, mean 6.0 deg, max 32.0 deg |
| Clicked points vs SAM 2 axis, forward swing | median 3.2 deg front, 2.6 deg rear |
| Follow-through vs video | 83.8 / 74.1 deg mean vs clicks; 29.7 / 38.9 deg median vs SAM 2 (modulo 180 deg): low confidence |
| Impact | sample 200; shock disturbs the gyro for 43.3 ms (detected from the vibration level and bridged) |

Coach numbers: peak rotation 1720 deg/s (2.4 ms before contact),
net rotation into contact 163 deg, roll about the handle -46 deg,
forward swing 224 ms, head speed from racket rotation about the hand 37 km/h
(the hand's own forward speed adds on top). The rotation centre sits at the grip.

### Independent check with SAM 2 (figure 4)

The camera fit is tuned on the clicked points, so "vs clicked points" is the optimistic number. SAM 2 (small model)
segments the racket in both videos, prompted with the clicks on the first frame of each view only, and the long axis
of each mask is an independent 2D racket axis. Masks with elongation below 1.6 are not trusted (front 29/29 trusted,
rear 23/29). Against SAM 2 the sensor-only reconstruction is 4 to 5 deg median off over the forward swing,
rising to 24 to 32 deg in the last ~70 ms before contact, the same pattern as against the clicks.
The clicks themselves agree with SAM 2 to about 3 deg. The frames where they differ by more than 8 deg
(front 126; rear 172, 173, 176, 177) are SAM errors on inspection: around contact the rear mask covers only the blurred head,
and the front mask includes the hand. Run on one GH200 on CSC Roihu in 12 s (`hpc/roihu/README.md`); numbers in
`out/sam2/sam_validation.json`, masks in `out/sam2/overlays/`.

## Known limits (say these out loud, judges respect it)

1. Absolute tilt and heading are not observable here: the recording starts mid-swing, so there is no moment of rest
   to read gravity from, and there is no magnetometer. Face tilt and swing-path angle are therefore withheld
   (they are in results.json under `gravity_dependent_metrics_NOT_RELIABLE`). The viewer shows the racket from the
   front camera's angle for display only.
2. After contact the gyro is disturbed by the impact shock, and the follow-through matches the video poorly.
3. The best video sync offset sits at the one-frame limit in both views, so the impact frames may be off by a frame.
   This may explain part of the error rise in the last ~70 ms before contact, which shows against both the clicks and SAM 2.
4. Impact location (0.4 cm from the centre line) is experimental only.
5. The SAM 2 check is 2D and modulo 180 deg: it cannot detect a butt/tip flip, and its axis is biased when the
   racket head is seen face-on or blurred (rear view around contact).

## Open question for the organizers

Is positive gyro rotation right-handed in the accelerometer frame?
- Accelerometer physics says yes: with the opposite sign the model puts the clipped samples at 9.4 g, below the 16 g rail, which is impossible.
- The gyro axis photo appears to show all three arrows the other way, and the follow-through fits the video better flipped
  (31.7 / 37.7 deg instead of 83.8 / 74.1 deg).
- Results for the flipped convention are in `out_gyro_sign_flipped/results.json` . Rerun with `GYRO_SIGN=-1` to rebuild everything.


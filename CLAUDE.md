# Project context for Claude Code

Hackathon project: reconstruct a tennis racket's 3D orientation through one forehand from the SWINQ dampener IMU.
The challenge rule is "relying solely on the sensor data", so:
- Orientation must come from the gyroscope (pipeline/core.py `integrate`). Never feed video data into the orientation.
- The videos are only a referee (pipeline/fastval.py fits camera angle, scale and a sync offset of at most 14 samples).

Key facts
- data/raw_data.csv: 400 samples, 416 Hz, accel in g (clips at 16 g on samples 182 to 199), gyro in deg/s, contact at sample 200.
- Sensor frame: X toward the butt, Y across the face, Z out of the face. Sensor sits on the strings near the throat.
- GYRO_SIGN env var: +1 (default, right-handed, backed by accelerometer physics) or -1 (what the organizer's axis photo seems to show).
- Gravity tilt and heading are not observable in this recording; do not report face tilt or swing-path angle as reliable.
- After contact the gyro is disturbed for about 43 ms (bridged in `robust_gyro`); the follow-through is low confidence.

Commands
- Full rebuild: `./run_all.sh` (writes out/ and viewer/swing_viewer.html)
- Blender: `blender -b -P blender/render_racket.py -- --preview`, then without `--preview` for the mp4
- Paths are repo-relative through pipeline/paths.py; run scripts from anywhere.

Style: keep results honest, keep numbers in out/results.json the single source of truth, do not hand-edit generated files.

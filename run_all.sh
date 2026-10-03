#!/usr/bin/env bash
# Rebuild every output from the raw data.
#   ./run_all.sh                 right-handed gyro (default)
#   GYRO_SIGN=-1 ./run_all.sh    flipped gyro convention
#   PYTHON=python ./run_all.sh   if your interpreter is called python, not python3
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p out
"$PY" pipeline/swing_pipeline.py > out/pipeline_log.txt
"$PY" pipeline/figures.py
"$PY" pipeline/overlay.py
"$PY" pipeline/build_viewer.py
echo "Done. Results in out/, viewer in viewer/swing_viewer.html"
echo "Blender preview: blender -b -P blender/render_racket.py -- --preview"
echo "Blender video:   blender -b -P blender/render_racket.py"

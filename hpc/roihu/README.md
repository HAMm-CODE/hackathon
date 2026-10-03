# SAM 2 referee on CSC Roihu

`pipeline/sam2_axis.py` segments the racket in both videos with SAM 2 (prompted with the
clicked points on the first frame of each view only) and measures its 2D axis per frame.
`pipeline/validate_sam.py` then compares that axis with the clicks and with the sensor-only
reconstruction. This is a referee only: nothing from the video goes into the orientation.

Roihu's GPU nodes are NVIDIA GH200 with ARM (aarch64) Grace CPUs; the login nodes are x86.
So the venv is built on a GPU node, never on the login node.

Placeholders below: `project_XXXXXXX` = your CSC project, `<username>` = your CSC username,
`<roihu-login-host>` = the Roihu login address.
**TODO(Roihu):** fill in the login hostname from docs.csc.fi.

## 0. On the laptop: check the frame numbering (no GPU, no torch)

```bash
python pipeline/sam2_axis.py --check-only
```

Open `out/sam2/first_frame_check_front.png` and `out/sam2/first_frame_check_rear.png`: the red
circles must sit on the racket. If they do not, try `--frame-offset 1` (or `-1`) until they do,
and use that value as `FRAME_OFFSET` in step 3. Checked 2026-10-03: **offset 0** is correct in
both views (±1 visibly misses the racket on the fast frames just before contact).

## 1. On the laptop: copy the repo to Roihu

Run from the repo root in Git Bash (or WSL).

```bash
ssh <username>@<roihu-login-host> "mkdir -p /scratch/project_XXXXXXX/<username>/hackathon"
rsync -av --exclude .git --exclude __pycache__ --exclude out_gyro_sign_flipped \
    ./ <username>@<roihu-login-host>:/scratch/project_XXXXXXX/<username>/hackathon/
```

Git Bash on Windows has no `rsync`. This does the same with `tar` over `ssh` (about 9 MB):

```bash
tar czf - --exclude=.git --exclude=__pycache__ --exclude=out_gyro_sign_flipped . \
  | ssh <username>@<roihu-login-host> "mkdir -p /scratch/project_XXXXXXX/<username>/hackathon && tar xzf - -C /scratch/project_XXXXXXX/<username>/hackathon"
```

## 2. On Roihu, once: build the SAM 2 environment

```bash
ssh <username>@<roihu-login-host>
cd /scratch/project_XXXXXXX/<username>/hackathon
export PROJECT=project_XXXXXXX
mkdir -p out/sam2
sbatch --account=$PROJECT hpc/roihu/setup_env.sh
```

This is a short gputest job (15 min max) that installs into `/projappl/$PROJECT/sam2-venv` and
`/projappl/$PROJECT/sam2`, and downloads only `sam2.1_hiera_small.pt`. Its log is
`hpc/roihu/setup-<jobid>.out`; it should end with `cuda available: True` and `Setup done.`

Alternative, in an interactive GPU session:

```bash
srun --account=$PROJECT --partition=gputest --gres=gpu:gh200:1 --cpus-per-task=16 --time=00:15:00 --pty bash
bash hpc/roihu/setup_env.sh
exit
```

**TODO(Roihu):** confirm that interactive `srun --pty` is allowed on gputest (or whether CSC
provides `sinteractive` for the GH200 nodes). The batch version above does not depend on it.

**TODO(Roihu):** setup needs outbound internet from the GPU node (git clone, PyPI, checkpoint).
If compute nodes are offline, run the `git clone` and the checkpoint `curl` from `setup_env.sh`
on the login node first (both are architecture independent and the script keeps existing
copies); `pip install` must still run on the ARM node.

## 3. On Roihu: run SAM 2

```bash
cd /scratch/project_XXXXXXX/<username>/hackathon
export PROJECT=project_XXXXXXX
sbatch --account=$PROJECT --export=ALL,FRAME_OFFSET=0 hpc/roihu/sam2_job.sh
```

Wait for the setup job to show `COMPLETED` before submitting this. The job prints `nvidia-smi`
and `torch.cuda.is_available()`, then runs `sam2_axis.py --device cuda` (it stops rather than
falling back to the CPU).

## 4. On Roihu: monitor

```bash
squeue -u $USER                                                  # PD = waiting, R = running
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode   # after it finishes
tail -f out/sam2/slurm-<jobid>.out                               # live output
cat out/sam2/run_log.txt                                         # device, timings, trusted frames per view
```

## 5. On the laptop: copy the results back

From the repo root:

```bash
rsync -av <username>@<roihu-login-host>:/scratch/project_XXXXXXX/<username>/hackathon/out/sam2/ out/sam2/
```

Without rsync:

```bash
ssh <username>@<roihu-login-host> "tar czf - -C /scratch/project_XXXXXXX/<username>/hackathon/out sam2" | tar xzf - -C out
```

## 6. On the laptop: validate (CPU only)

```bash
python pipeline/validate_sam.py
```

This writes `out/sam2/sam_validation.json` and `out/fig4_sam_validation.png`. Only frames whose
SAM mask has elongation >= 1.6 are used. Look at `out/sam2/overlays/<view>/<frame>.jpg`
(green = SAM mask and axis, red = clicks) for any frame flagged in `sam_vs_clicks.csv`.

## What `sam2_axis.py` writes to `out/sam2/`

| file | content |
|---|---|
| `run_log.txt` | host, device, model load and tracking times, leakage-guard retries, trusted frames per view |
| `sam_axes.json` | per view and frame: axis angle, elongation, centroid, mask area, trusted flag |
| `sam_vs_clicks.csv` | per frame: clicked angle, SAM angle, difference, flag |
| `overlays/<view>/<frame>.jpg` | mask and both axes drawn on the frame |
| `first_frame_check_<view>.png` | clicks on the prompt frame (same as step 0) |
| `slurm-<jobid>.out` | the job's full output |

Leakage guard: if the mask area jumps by more than 50% between frames, or elongation drops below
1.6, the view is rerun once with an extra negative point at the wrist (0.12 racket lengths beyond
the butt click on the first frame). The attempt with fewer suspect frames is kept; both
attempts are logged.

## Troubleshooting

- `cuda available: False`: the job did not get a GPU. Check `--gres=gpu:gh200:1`, and that
  no `cuda` module is loaded (python-pytorch brings its own).
- pip reports a torch conflict during setup: sam2 needs a newer torch than the loaded
  python-pytorch module. Setup pins the module's torch on purpose so pip cannot silently swap
  in a different build. Load a newer python-pytorch version (`module avail python-pytorch`)
  and rerun setup.
- `checkpoint not found`: the setup job did not finish; check `hpc/roihu/setup-<jobid>.out`.
- Job log missing: `out/sam2/` must exist before `sbatch` (Slurm opens the log first).

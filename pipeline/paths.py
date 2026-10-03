"""Repo-relative paths, so every script works no matter where it is launched from."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
OUT = ROOT / "out"
VIEWER = ROOT / "viewer"

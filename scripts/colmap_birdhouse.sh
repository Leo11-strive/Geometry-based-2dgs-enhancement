#!/usr/bin/env bash
set -euo pipefail

# Wrapper around `convert.py` to run COLMAP on the birdhouse images located in:
#   data/birdhouse/rgb
#
# It prepares `data/birdhouse/input` (expected by convert.py) as a symlink to `rgb`,
# then runs the full COLMAP pipeline + undistortion.
#
# Usage:
#   bash scripts/colmap_birdhouse.sh
#
# Common options (forwarded to convert.py):
#   --no_gpu
#   --skip_matching
#   --camera OPENCV
#   --colmap_executable /path/to/colmap
#   --resize
#   --magick_executable /path/to/magick
#
# After completion, your dataset will be in:
#   data/birdhouse/images/        (undistorted images)
#   data/birdhouse/sparse/0/      (cameras.bin/images.bin/points3D.bin...)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_DIR="${SCENE_DIR:-$REPO_ROOT/data/suitcase}"
RGB_DIR="${RGB_DIR:-$SCENE_DIR/rgb}"
INPUT_DIR="$SCENE_DIR/input"

if [[ ! -d "$RGB_DIR" ]]; then
  echo "ERROR: RGB_DIR not found: $RGB_DIR" >&2
  exit 1
fi

mkdir -p "$SCENE_DIR"

# Prepare input/ expected by convert.py.
if [[ -e "$INPUT_DIR" ]]; then
  if [[ -L "$INPUT_DIR" ]]; then
    # If it's already a symlink, keep it unless it points elsewhere.
    TARGET="$(readlink "$INPUT_DIR")"
    if [[ "$TARGET" != "$RGB_DIR" ]]; then
      echo "ERROR: $INPUT_DIR is a symlink to '$TARGET' (expected '$RGB_DIR')." >&2
      echo "       Remove it or set INPUT_DIR/RGB_DIR accordingly." >&2
      exit 1
    fi
  else
    echo "ERROR: $INPUT_DIR exists but is not a symlink." >&2
    echo "       Please remove it, or manually copy/link your images into $INPUT_DIR." >&2
    exit 1
  fi
else
  ln -s "$RGB_DIR" "$INPUT_DIR"
fi

cd "$REPO_ROOT"
python convert.py -s "$SCENE_DIR" "$@"


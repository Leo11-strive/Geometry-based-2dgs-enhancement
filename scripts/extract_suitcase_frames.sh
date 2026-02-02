#!/usr/bin/env bash
set -euo pipefail

VIDEO="/mnt/nas_9/group/lixingxuan/SRTP/2d-gaussian-splatting_modified/data/suitcase/20251220_101514.mp4"
OUT_DIR="data/suitcase/rgb"
COUNT=100

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe not found; please install ffmpeg." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; please install ffmpeg." >&2
  exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "Video not found: $VIDEO" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

duration="$(ffprobe -v error -select_streams v:0 -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 "$VIDEO")"

if [[ -z "$duration" ]]; then
  echo "Failed to read video duration from: $VIDEO" >&2
  exit 1
fi

fps="$(awk -v count="$COUNT" -v dur="$duration" 'BEGIN { printf "%.10f", count/dur }')"

ffmpeg -v error -i "$VIDEO" \
  -vf "fps=${fps}" -frames:v "$COUNT" \
  -q:v 2 "$OUT_DIR/frame_%04d.jpg"

echo "Extracted frames to $OUT_DIR"

#!/usr/bin/env bash
# Style-transfer a trained Gaussian splat using SGSST (arXiv:2412.03371).
# Scoped to run on a 6GB-class consumer GPU (the paper used an 80GB A100) --
# see README.md for the full rationale behind each setting.
#
# Usage:
#   ./scripts/stylize_splat.sh <splat_project_dir> <style_image> <output_name> [mode] [iterations]
#
# <splat_project_dir> is a scene trained with the companion
# gaussian-splatting-pipeline repo (or any compatible setup): must contain
# <splat_project_dir>/undistorted/sparse/0/ and
# <splat_project_dir>/output/point_cloud/iteration_7000/point_cloud.ply
#
# [mode] is one of:
#   fast     (default) -- downscaled resolution, single-scale optimization
#            only (never reaches SGSST's multi-scale SOS loss). Minutes,
#            not hours. Good for quick previews / iterating on style choice.
#   quality  -- FULL resolution, lazy per-image disk loading (bounded
#            memory regardless of dataset size), small VGG tiles for VRAM
#            safety, AND genuinely reaches the multi-scale SOS loss phase
#            (SGSST's actual paper contribution). Confirmed real runtime
#            on a 699-camera/101k-gaussian scene on an RTX 3050: ~7.5 HOURS.
#            Use this only when you want the best achievable result and can
#            let it run unattended.
#
# [iterations] overrides the mode's default total iteration count if given.
#
# Environment variables (all optional):
#   SGSST_DIR   -- path to your SGSST checkout (default: ./SGSST relative
#                  to this script's repo root)
#   SGSST_ENV   -- conda/micromamba/venv environment name or activation
#                  prefix to run Python in (default: none, uses `python3`
#                  on PATH directly)
#   REGISTER_CMD -- optional command to run on the final stylized .ply to
#                  register it somewhere (e.g. a viewer gallery's
#                  add_splat.sh). Called as:
#                    $REGISTER_CMD <ply_path> <output_name> "<label>"
#                  Left empty by default -- the script just reports the
#                  output .ply path and you decide what to do with it.
#
# Examples:
#   ./scripts/stylize_splat.sh ~/scenes/my-scene \
#       examples/styles/starry_night.jpg my-scene-starry fast
#
#   SGSST_ENV=gs ./scripts/stylize_splat.sh ~/scenes/my-scene \
#       examples/styles/starry_night.jpg my-scene-starry-hq quality

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SPLAT_PROJECT="$1"
STYLE_IMG="$(realpath "$2")"  # resolve to absolute path -- stylize.py changes
                                # its own working directory internally, so a
                                # relative path here breaks (confirmed bug
                                # while developing this script).
OUT_NAME="$3"
MODE="${4:-fast}"

case "$MODE" in
  fast)
    # Downscaled resolution, single-scale only. Fast preview quality.
    RESOLUTION=2
    LAZY_FLAG=""
    VGG_BLOCK_SIZE=512
    DEFAULT_ITERATIONS=9000
    OPTIMIZE_ITERATION=10000  # intentionally never crosses into multi-scale
    ;;
  quality)
    # Full resolution, lazy disk loading (bounded memory), small VGG
    # tiles (VRAM safety), genuinely reaches multi-scale SOS loss.
    # Confirmed working end-to-end -- real ~7.5hr runtime on a dense
    # (699-camera) scene on a 6GB GPU. Lighter scenes will be faster;
    # GPUs with more VRAM can raise VGG_BLOCK_SIZE for a speedup.
    RESOLUTION=1
    LAZY_FLAG="--lazy_images"
    VGG_BLOCK_SIZE=256
    DEFAULT_ITERATIONS=15000
    OPTIMIZE_ITERATION=2000   # ~2000 iters pre-opt, rest in real multi-scale
    ;;
  *)
    echo "ERROR: unknown mode '$MODE' -- expected 'fast' or 'quality'." >&2
    exit 1
    ;;
esac

ITERATIONS="${5:-$DEFAULT_ITERATIONS}"

SGSST_DIR="${SGSST_DIR:-$REPO_DIR/SGSST}"
PLY_IN="$SPLAT_PROJECT/output/point_cloud/iteration_7000/point_cloud.ply"
SOURCE_PATH="$SPLAT_PROJECT/undistorted"
CHKPNT_OUT="$(mktemp -t "${OUT_NAME}_chkpnt7000.XXXXXX.pth")"
MODEL_OUT="$SGSST_DIR/output/${OUT_NAME}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduces fragmentation OOMs on dense scenes

if [ -n "${SGSST_ENV:-}" ] && command -v micromamba >/dev/null 2>&1; then
  RUN_PY="micromamba run -n $SGSST_ENV python3"
elif [ -n "${SGSST_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  RUN_PY="conda run -n $SGSST_ENV python3"
else
  RUN_PY="python3"
fi

if [ ! -f "$PLY_IN" ]; then
  echo "ERROR: no trained point_cloud.ply at $PLY_IN -- train this splat first (see the companion gaussian-splatting-pipeline repo)." >&2
  exit 1
fi
if [ ! -d "$SOURCE_PATH/sparse/0" ]; then
  echo "ERROR: no undistorted/sparse/0 at $SOURCE_PATH -- expected COLMAP's undistorted output layout." >&2
  exit 1
fi
if [ ! -f "$SGSST_DIR/stylize.py" ]; then
  echo "ERROR: SGSST checkout not found at $SGSST_DIR -- see README.md 'Setup' section." >&2
  exit 1
fi

echo "== [1/3] Building SGSST checkpoint from existing ply =="
cd "$SGSST_DIR"
$RUN_PY "$REPO_DIR/scripts/make_checkpoint_from_ply.py" "$PLY_IN" "$CHKPNT_OUT"

echo "== [2/3] Running style transfer: mode=$MODE, $ITERATIONS total iterations, resolution=$RESOLUTION, vgg_block_size=$VGG_BLOCK_SIZE, optimize_iteration=$OPTIMIZE_ITERATION =="
rm -rf "$MODEL_OUT"
$RUN_PY stylize.py \
  --source_path "$SOURCE_PATH" \
  --model_path "$MODEL_OUT" \
  --start_checkpoint "$CHKPNT_OUT" \
  --style_img "$STYLE_IMG" \
  --iterations "$ITERATIONS" \
  --optimize_iteration "$OPTIMIZE_ITERATION" \
  --resolution "$RESOLUTION" \
  $LAZY_FLAG \
  --vgg_block_size "$VGG_BLOCK_SIZE" \
  --content_weights 0.5

rm -f "$CHKPNT_OUT"  # checkpoint is a regenerable intermediate -- don't leave it on disk

STYLIZED_PLY="$MODEL_OUT/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
if [ ! -f "$STYLIZED_PLY" ]; then
  echo "ERROR: expected stylized ply not found at $STYLIZED_PLY" >&2
  exit 1
fi

echo "== [3/3] Done =="
echo "Stylized point cloud: $STYLIZED_PLY"

if [ -n "${REGISTER_CMD:-}" ]; then
  LABEL="$(basename "$SPLAT_PROJECT") x $(basename "$STYLE_IMG" | sed 's/\.[^.]*$//') (SGSST $MODE)"
  echo "Registering via REGISTER_CMD..."
  $REGISTER_CMD "$STYLIZED_PLY" "$OUT_NAME" "$LABEL"
fi

#!/bin/bash
# Download convnext-tiny watermark weights into a host directory.
# Build first: docker compose build
# Usage: ./scripts/download_model.sh [target_dir]   (default: /data/models/watermark)

set -eu

MODEL_DIR="${1:-/data/models/watermark}"
IMAGE="${WM_IMAGE:-wmdetect-api:latest}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$MODEL_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found; build it first: docker compose build" >&2
    exit 1
fi

docker run --rm -i \
    -e HF_ENDPOINT="$HF_ENDPOINT" \
    -v "$MODEL_DIR":/out \
    --entrypoint python \
    "$IMAGE" - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="boomb0om/watermark-detectors",
    filename="convnext-tiny_watermarks_detector.pth",
    local_dir="/out",
)
print("saved", path)
PY

docker run --rm \
    -v "$MODEL_DIR":/out \
    --entrypoint sh \
    "$IMAGE" -c "chown -R $(id -u):$(id -g) /out"

ls -lh "$MODEL_DIR"

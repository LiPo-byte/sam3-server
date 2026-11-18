#!/bin/bash
# Download PP-OCRv6 det + rec into a host directory, using the built image.
# Build first: docker compose build
# Usage: ./scripts/download_model.sh [target_dir]   (default: /data/models/ppocrv6)
#   DOWNLOAD_PREPROCESS=1 also fetches doc orientation, unwarping, and textline orientation.

set -eu

MODEL_DIR="${1:-/data/models/ppocrv6}"
IMAGE="${PPOCR_IMAGE:-ppocr-api:latest}"
PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-BOS}"
DOWNLOAD_PREPROCESS="${DOWNLOAD_PREPROCESS:-0}"

mkdir -p "$MODEL_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found; build it first: docker compose build" >&2
    exit 1
fi

docker run --rm -i \
    -e PADDLE_PDX_MODEL_SOURCE="$PADDLE_PDX_MODEL_SOURCE" \
    -e PADDLE_PDX_CACHE_HOME=/cache \
    -e DOWNLOAD_PREPROCESS="$DOWNLOAD_PREPROCESS" \
    -v "$MODEL_DIR":/cache/official_models \
    --entrypoint python \
    "$IMAGE" - <<'PY'
import os
from paddleocr import PaddleOCR

common = dict(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    device="cpu",
)
print("downloading PP-OCRv6_medium_det and PP-OCRv6_medium_rec ...")
PaddleOCR(
    **common,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
if os.environ.get("DOWNLOAD_PREPROCESS") == "1":
    print("downloading doc orientation, unwarping, and textline orientation ...")
    PaddleOCR(
        **common,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
print("download done")
PY

docker run --rm \
    -v "$MODEL_DIR":/out \
    --entrypoint sh \
    "$IMAGE" -c "chown -R $(id -u):$(id -g) /out"

ls -lh "$MODEL_DIR"

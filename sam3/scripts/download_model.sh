#!/bin/bash
# Download unsloth/sam3.1 (BF16 only) from ModelScope into a host directory.
# Runs modelscope inside a throwaway container, so nothing is installed on the host.
# Usage: ./scripts/download_model.sh [target_dir]   (default: /data/models/sam3.1)

set -eu

MODEL_DIR="${1:-/data/models/sam3.1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

mkdir -p "$MODEL_DIR"

TTY_FLAG=""
[[ -t 1 ]] && TTY_FLAG="-t"

docker run --rm $TTY_FLAG \
    -v "$MODEL_DIR":/out \
    python:3.12-slim \
    sh -c "echo '[1/2] installing modelscope ...' \
        && pip install -q --progress-bar on modelscope -i $PIP_INDEX_URL \
        && echo '[2/2] downloading unsloth/sam3.1 (~1.7GB) ...' \
        && modelscope download --model unsloth/sam3.1 --local_dir /out \
           --exclude 'sam3.1-*.safetensors' 'samples/*' \
        && chown -R $(id -u):$(id -g) /out"

ls -lh "$MODEL_DIR"

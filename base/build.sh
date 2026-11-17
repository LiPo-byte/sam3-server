#!/bin/bash
# Build sam3-base from torch already installed in a local venv.
# Usage: ./build.sh [venv_dir] [image_tag]
#   default venv_dir:  ~/venvs/sam2
#   default image_tag: sam3-base:torch2.11-cu128

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VENV_DIR="${1:-$HOME/venvs/sam2}"
IMAGE_TAG="${2:-sam3-base:torch2.11-cu128}"
STAGING="$SCRIPT_DIR/_site"

PY_VER=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [[ "$PY_VER" != "3.12" ]]; then
    echo "source venv is Python $PY_VER, base/Dockerfile expects 3.12" >&2
    exit 1
fi

"$VENV_DIR/bin/python" "$SCRIPT_DIR/collect_torch.py" "$STAGING"
du -sh "$STAGING"

docker build -t "$IMAGE_TAG" -f "$SCRIPT_DIR/Dockerfile" "$STAGING"

rm -rf "$STAGING"
docker image ls "$IMAGE_TAG"

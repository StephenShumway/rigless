#!/usr/bin/env bash
# Sets up a RunPod pod for Rigless and starts it: ComfyUI with the H3 Relay nodes, the
# MiniMax H3 character-swap models, the app's environment, and comfy-api-proxy, all
# under /workspace (the pod's disk). A new pod takes about 25 minutes, mostly
# downloads (~67GB); a restarted one skips what's there and is up in a few minutes.
# Safe to rerun: interrupted downloads resume.
#
# Usage, on the pod from a checkout of this repo:
#   bash deploy/runpod/setup.sh
#
# Needs a GPU with 48GB or more, a driver for CUDA 13 (comfy-kitchen's fast int8
# kernels need it), the pod environment variable PROXY_TOKEN, and port 8189 exposed
# over HTTP. HF_TOKEN, if set, downloads the Rigless Swap LoRA from a private repo.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
COMFY=/workspace/ComfyUI
VENV=/workspace/venv
COMFY_VERSION=v0.41.2  # the version the swap graph was validated against
H3_RELAY=e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220  # akatz-ai/h3-relay v2.0.1, GPL-3.0
# The Pythons uv installs live on /workspace with the venvs, so both survive a
# restart; its download cache stays off it, on the container disk.
export UV_PYTHON_INSTALL_DIR=/workspace/uv/python UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

CUDA_MAJOR=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | grep -oE "[0-9]+$")
if [ "$CUDA_MAJOR" -lt 13 ]; then
  echo "This pod's driver supports CUDA $CUDA_MAJOR; Rigless needs CUDA 13." >&2
  exit 1
fi

[ -d "$COMFY" ] || git clone --depth 1 --branch "$COMFY_VERSION" https://github.com/Comfy-Org/ComfyUI "$COMFY"
if [ ! -d "$COMFY/custom_nodes/h3-relay" ]; then
  git clone --quiet https://github.com/akatz-ai/h3-relay "$COMFY/custom_nodes/h3-relay"
  git -C "$COMFY/custom_nodes/h3-relay" checkout --quiet "$H3_RELAY"
fi
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$VENV/bin/python" -r "$COMFY/requirements.txt" scipy comfy-api-proxy==0.1.6

export UV_PROJECT_ENVIRONMENT=/workspace/rigless-venv
uv sync --project "$REPO" --no-dev
uv run --project "$REPO" --no-dev python "$REPO/scripts/download_models.py" --models-dir "$COMFY/models"
bash "$REPO/deploy/runpod/start.sh"

#!/usr/bin/env bash
# Sets up a RunPod pod for Rigless and starts it: ComfyUI with the H3 Relay nodes, the
# MiniMax H3 character-swap models, the app's environment, and comfy-api-proxy.
# ComfyUI and the models live on /workspace (the pod's disk), the Python environments
# on the container disk. A new pod takes about 5 minutes on a fast data center (Oct
# 28), mostly the ~67GB download; a restarted one reinstalls the environments, skips
# the rest, and is up in a few minutes. Safe to rerun: interrupted downloads resume.
#
# Usage, on the pod from a checkout of this repo:
#   bash deploy/runpod/setup.sh
#
# Needs a GPU with 48GB or more, a driver for CUDA 13 (comfy-kitchen's fast int8
# kernels need it), the pod environment variable PROXY_TOKEN, and port 8189 exposed
# over HTTP. HF_TOKEN, if set, downloads the Rigless LoRA from a private repo.
set -euo pipefail

# RunPod writes pod environment variables here; SSH sessions don't inherit them.
[ -f /etc/rp_environment ] && source /etc/rp_environment
: "${PROXY_TOKEN:?Set the pod environment variable PROXY_TOKEN.}"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
COMFY=/workspace/ComfyUI
VENV=/opt/venv
COMFY_VERSION=v0.41.2  # the version the swap graph was validated against
H3_RELAY=e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220  # akatz-ai/h3-relay v2.0.1, GPL-3.0
# Where /workspace is network storage it can write small files ~100x slower than the
# container disk (72 a second against 7,000 on a us-mo-1 pod, Oct 28), and the
# environments are ~30,000 of them: an hour to install there, and a ComfyUI start
# longer than start.sh waited. So uv's Pythons, cache and venvs stay on the
# container disk, overriding the image's UV_CACHE_DIR on /workspace.
export UV_CACHE_DIR=/root/.cache/uv UV_PYTHON_PREFERENCE=only-managed

# Until comfy-api-proxy takes port 8189, the app reads the setup's progress there.
PROGRESS=/workspace/logs/progress
STARTED=$(date +%s)
mkdir -p "$PROGRESS"
stage() {
  echo "{\"stage\": \"$1\", \"since\": $(date +%s), \"started\": $STARTED}" > "$PROGRESS/progress.json"
}
stage comfyui
# EXIT, not ERR: ERR misses an explicit `exit 1`.
trap '[ $? -eq 0 ] || stage failed' EXIT
pkill -f "[h]ttp.server 8189" || true
nohup python3 -m http.server 8189 --directory "$PROGRESS" > /dev/null 2>&1 < /dev/null &

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

CUDA_MAJOR=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | grep -oE "[0-9]+$")
if [ "$CUDA_MAJOR" -lt 13 ]; then
  echo "This pod's driver supports CUDA $CUDA_MAJOR; Rigless needs CUDA 13." >&2
  exit 1
fi

[ -d "$COMFY" ] || git clone --depth 1 --branch "$COMFY_VERSION" https://github.com/Comfy-Org/ComfyUI "$COMFY"
[ -d "$COMFY/custom_nodes/h3-relay" ] || git clone --quiet https://github.com/akatz-ai/h3-relay "$COMFY/custom_nodes/h3-relay"
# Every run, so a checkout that failed once never leaves the nodes at their latest.
git -C "$COMFY/custom_nodes/h3-relay" checkout --quiet "$H3_RELAY"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
stage torch
uv pip install --python "$VENV/bin/python" torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
stage packages
uv pip install --python "$VENV/bin/python" -r "$COMFY/requirements.txt" scipy comfy-api-proxy==0.1.6

stage rigless
export UV_PROJECT_ENVIRONMENT=/opt/rigless-venv
uv sync --project "$REPO" --no-dev --locked
uv run --project "$REPO" --no-dev python "$REPO/scripts/download_models.py" \
  --models-dir "$COMFY/models" --progress "$PROGRESS/progress.json" --setup-started "$STARTED"
stage start
bash "$REPO/deploy/runpod/start.sh"

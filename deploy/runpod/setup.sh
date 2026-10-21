#!/usr/bin/env bash
# One-time setup of a RunPod pod for rigless: ComfyUI, the Wan Animate 2, Qwen-Image
# 2.1, and Qwen-Image-Edit 2511 models, and comfy-api-proxy, all under /workspace so they survive
# pod stops. Safe to rerun: finished steps are skipped, interrupted downloads resume.
#
# Usage, on the pod from a checkout of this repo:
#   bash deploy/runpod/setup.sh
#
# Needs the pod environment variable PROXY_TOKEN, set from a RunPod secret. The web app
# also needs RIGLESS_PASSCODE (another secret) and port 8190 exposed over HTTP.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
COMFY=/workspace/ComfyUI
VENV=/workspace/venv
COMFY_VERSION=v0.41.2  # the version our workflow was validated against

# ComfyUI turns off comfy-kitchen's fast int8/fp8 kernels unless PyTorch is built for
# CUDA 13+, which needs a driver for CUDA 13+. Use cu130 where the driver allows it.
CUDA_MAJOR=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | grep -oE "[0-9]+$")
if [ "$CUDA_MAJOR" -ge 13 ]; then CUDA_TAG=cu130; else CUDA_TAG=cu128; fi
TORCH_INDEX=https://download.pytorch.org/whl/$CUDA_TAG

if [ ! -d "$COMFY" ]; then
  git clone --depth 1 --branch "$COMFY_VERSION" https://github.com/Comfy-Org/ComfyUI "$COMFY"
fi
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision torchaudio --index-url "$TORCH_INDEX"
uv pip install --python "$VENV/bin/python" -r "$COMFY/requirements.txt" comfy-api-proxy==0.1.6
"$VENV/bin/python" "$REPO/scripts/download_models.py" --models-dir "$COMFY/models" \
  --bf16 --qwen-edit --qwen-image-21
bash "$REPO/deploy/runpod/start.sh"

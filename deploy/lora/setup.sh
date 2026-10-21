#!/usr/bin/env bash
# One-time setup of a GPU pod for training the real-footage LoRA with musubi-tuner:
# the trainer in its own venv and the Wan 2.1 I2V 14B weights it trains on. Safe to
# rerun. Needs uv and git; an 80GB GPU trains in bf16 without fp8 or block swap.
#
# Usage, on the pod from a copy of this folder:
#   bash setup.sh
set -euo pipefail

TRAINER=/workspace/musubi-tuner
VENV=/root/musubi-venv  # container disk: fast imports; rebuilt in minutes after a stop
MODELS=/workspace/models

[ -d "$TRAINER" ] || git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git "$TRAINER"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# The trainer's own torch, not the image's: mixing them breaks torchvision/torchaudio.
uv pip install --python "$VENV/bin/python" torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$VENV/bin/python" -e "$TRAINER" tensorboard

HF="$VENV/bin/hf"
"$HF" download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir "$MODELS"
"$HF" download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors \
  split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors \
  split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors --local-dir "$MODELS"

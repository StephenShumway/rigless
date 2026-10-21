#!/usr/bin/env bash
# Trains the real-footage LoRA on Wan 2.1 I2V 14B 720p. Wan Animate 2 was fine-tuned
# from that model (only its self-attention changed; checked Oct 19), so the LoRA loads
# into our Animate 2 workflow like the LightX2V LoRA does. Caches the clips' latents
# and captions first; checkpoints land in /workspace/lora/output every 250 steps.
#
# Usage, on the pod after setup.sh, with the clips in /workspace/lora/clips:
#   bash train.sh 20     # smoke test: speed and memory
#   bash train.sh 1500   # full run
set -euo pipefail

STEPS=${1:?Give the number of training steps.}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/root/musubi-venv/bin
MODELS=/workspace/models
OUT=/workspace/lora

cd /workspace/musubi-tuner
"$PY/python" src/musubi_tuner/wan_cache_latents.py --dataset_config "$HERE/dataset.toml" \
  --vae "$MODELS/split_files/vae/wan_2.1_vae.safetensors" --i2v \
  --clip "$MODELS/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --skip_existing
"$PY/python" src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config "$HERE/dataset.toml" \
  --t5 "$MODELS/models_t5_umt5-xxl-enc-bf16.pth" --batch_size 16 --skip_existing

# musubi's documented Wan settings (rank 32, alpha 1, lr 2e-4), in bf16 on 80GB.
"$PY/accelerate" launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/wan_train_network.py \
  --task i2v-14B \
  --dit "$MODELS/split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors" \
  --dataset_config "$HERE/dataset.toml" --sdpa --mixed_precision bf16 \
  --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_wan --network_dim 32 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --max_train_steps "$STEPS" --save_every_n_steps 250 --seed 42 \
  --output_dir "$OUT/output" --output_name rigless_phone_look \
  --log_with tensorboard --logging_dir "$OUT/logs"

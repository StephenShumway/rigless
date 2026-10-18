#!/usr/bin/env bash
# Starts ComfyUI, reachable only inside the pod, and the token-protected
# comfy-api-proxy on port 8189, which RunPod exposes over HTTPS. Run after every
# pod start: processes don't survive a stop.
#
# Usage, on the pod:
#   bash deploy/runpod/start.sh
set -euo pipefail

# RunPod writes pod environment variables here; SSH sessions don't inherit them.
[ -f /etc/rp_environment ] && source /etc/rp_environment
: "${PROXY_TOKEN:?Set the pod environment variable PROXY_TOKEN from a RunPod secret.}"

COMFY=/workspace/ComfyUI
VENV=/workspace/venv
LOGS=/workspace/logs
mkdir -p "$LOGS"

if ! curl -sf http://127.0.0.1:8188/system_stats > /dev/null; then
  nohup "$VENV/bin/python" "$COMFY/main.py" --listen 127.0.0.1 --port 8188 \
    > "$LOGS/comfyui.log" 2>&1 &
  for _ in $(seq 90); do
    curl -sf http://127.0.0.1:8188/system_stats > /dev/null && break
    sleep 2
  done
  curl -sf http://127.0.0.1:8188/system_stats > /dev/null \
    || { echo "ComfyUI didn't start; see $LOGS/comfyui.log" >&2; exit 1; }
fi

"$VENV/bin/comfy-api-proxy" stop > /dev/null 2>&1 || true
"$VENV/bin/comfy-api-proxy" start --comfyui http://127.0.0.1:8188 \
  --host 0.0.0.0 --port 8189 --token "$PROXY_TOKEN"
echo "Ready. Set COMFY_BASE_URL=https://${RUNPOD_POD_ID:-<pod id>}-8189.proxy.runpod.net"

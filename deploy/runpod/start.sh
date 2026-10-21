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
  # ComfyUI sizes its RAM caches from the host's memory, not the pod's limit, so
  # pinned model copies piled up until the container was OOM-killed (Oct 19).
  # Unpinned, the weights stay memory-mapped and reclaimable.
  nohup "$VENV/bin/python" "$COMFY/main.py" --listen 127.0.0.1 --port 8188 \
    --disable-pinned-memory > "$LOGS/comfyui.log" 2>&1 &
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

# The web app on port 8190, only with a passcode (pod environment variable
# RIGLESS_PASSCODE, from a RunPod secret). It reaches ComfyUI through the proxy.
# Its environment lives on the container disk: uv links it to a Python there, which
# a pod restart wipes, and the network volume can't always delete the stale copy.
if [ -n "${RIGLESS_PASSCODE:-}" ]; then
  REPO=$(cd "$(dirname "$0")/../.." && pwd)
  pkill -f "[r]igless --host" || true
  COMFY_BASE_URL=http://127.0.0.1:8189 COMFY_API_KEY="$PROXY_TOKEN" \
    RIGLESS_PROFILE=cloud-48gb UV_PROJECT_ENVIRONMENT=/root/rigless-venv \
    nohup uv run --project "$REPO" --no-dev rigless \
    --host 0.0.0.0 --port 8190 --no-browser > "$LOGS/app.log" 2>&1 &
  echo "App: https://${RUNPOD_POD_ID:-<pod id>}-8190.proxy.runpod.net"
fi

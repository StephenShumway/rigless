#!/usr/bin/env bash
# Starts ComfyUI, reachable only inside the pod, and the token-protected
# comfy-api-proxy on port 8189, which RunPod exposes over HTTPS. setup.sh ends
# by running this; run it again after restarting anything by hand.
#
# Usage, on the pod:
#   bash deploy/runpod/start.sh
set -euo pipefail

# RunPod writes pod environment variables here; SSH sessions don't inherit them.
[ -f /etc/rp_environment ] && source /etc/rp_environment
: "${PROXY_TOKEN:?Set the pod environment variable PROXY_TOKEN.}"

COMFY=/workspace/ComfyUI
VENV=/workspace/venv
LOGS=/workspace/logs
mkdir -p "$LOGS"

if ! curl -sf http://127.0.0.1:8188/system_stats > /dev/null; then
  # ComfyUI sizes its RAM caches from the host's memory, not the pod's limit, so
  # pinned model copies piled up until the container was OOM-killed (Oct 19).
  # Unpinned, the weights stay memory-mapped and reclaimable.
  [ -f "$LOGS/comfyui.log" ] && mv "$LOGS/comfyui.log" "$LOGS/comfyui.previous.log"
  nohup "$VENV/bin/python" "$COMFY/main.py" --listen 127.0.0.1 --port 8188 \
    --disable-pinned-memory > "$LOGS/comfyui.log" 2>&1 &
  # Up to 10 minutes: a first start on a network disk took 4.5 (Oct 28).
  for _ in $(seq 300); do
    curl -sf http://127.0.0.1:8188/system_stats > /dev/null && break
    sleep 2
  done
  curl -sf http://127.0.0.1:8188/system_stats > /dev/null \
    || { echo "ComfyUI didn't start; see $LOGS/comfyui.log" >&2; exit 1; }
fi

# `stop` can leave the old proxy holding the port, still serving for a ComfyUI
# that has since restarted, and the new one then dies on "address in use".
"$VENV/bin/comfy-api-proxy" stop > /dev/null 2>&1 || true
pkill -f "[c]omfy-api-proxy start" || true
for _ in $(seq 15); do
  ss -ltn | grep ":8189 " > /dev/null || break
  sleep 1
done
"$VENV/bin/comfy-api-proxy" start --comfyui http://127.0.0.1:8188 \
  --host 0.0.0.0 --port 8189 --token "$PROXY_TOKEN"
echo "Ready. Set COMFY_BASE_URL=https://${RUNPOD_POD_ID:-<pod id>}-8189.proxy.runpod.net"

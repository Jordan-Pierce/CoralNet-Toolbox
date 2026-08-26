#!/usr/bin/env bash
#
# Convenience launcher. Mounts ./data only when it actually exists, and asks
# for the GPU only when the daemon can provide one.
#
#   ./docker/run.sh
#   CORALNET_DATA=/mnt/imagery ./docker/run.sh
#
set -euo pipefail

IMAGE="${IMAGE:-coralnet-toolbox:local}"
DATA="${CORALNET_DATA:-$PWD/data}"
PORT="${PORT:-6901}"

# Preflight. Docker's own errors for these two cases name an endpoint hash
# rather than the container in the way, which is not much help.
existing=$(docker ps -a --filter "name=^coralnet$" --format '{{.Names}}' 2>/dev/null || true)
if [ -n "$existing" ]; then
    echo "error: a container named 'coralnet' already exists." >&2
    echo "       docker rm -f coralnet" >&2
    exit 1
fi

holder=$(docker ps --filter "publish=${PORT}" --format '{{.Names}}' 2>/dev/null | head -1)
if [ -n "$holder" ]; then
    echo "error: port ${PORT} is already published by container '${holder}'." >&2
    echo "       docker rm -f ${holder}      # or: PORT=6902 ./docker/run.sh" >&2
    exit 1
fi

args=(--rm -it --name coralnet --shm-size=2g -p "${PORT}:6901"
      -e VNC_USER="${VNC_USER:-user}"
      -e VNC_PW="${VNC_PW:-password}"
      -e LOCKOUT_LEVEL="${LOCKOUT_LEVEL:-2}")

# A plain `-v` would silently CREATE an empty, root-owned ./data on the host if
# the path were missing. Only mount what is really there.
if [ -d "$DATA" ]; then
    args+=(-v "$DATA:/home/kasm-user/data")
    echo "data:  $DATA -> /home/kasm-user/data"
else
    echo "data:  no directory at $DATA (skipping mount)"
fi

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    args+=(--gpus all)
    echo "gpu:   enabled"
else
    echo "gpu:   no nvidia runtime detected, running on CPU"
fi

echo "lock:  LOCKOUT_LEVEL=${LOCKOUT_LEVEL:-2} (1=desktop, 2=kiosk, 3=unimplemented)"
echo "open:  https://localhost:${PORT}  (user: ${VNC_USER:-user})"
exec docker run "${args[@]}" "$IMAGE"

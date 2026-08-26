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

args=(--rm -it --name coralnet --shm-size=2g -p "${PORT}:6901"
      -e VNC_USER="${VNC_USER:-user}"
      -e VNC_PW="${VNC_PW:-password}")

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

echo "open:  https://localhost:${PORT}  (user: ${VNC_USER:-user})"
exec docker run "${args[@]}" "$IMAGE"

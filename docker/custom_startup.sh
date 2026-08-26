#!/usr/bin/env bash
set -u

# Kasm runs this once the session is up. The session lives exactly as long as
# this script does, so it must block.

# Everything this script and the app emit goes to a log we can read later with
# `docker exec <container> cat /tmp/coralnet-startup.log`. Without this, a
# startup failure is invisible -- the app's stderr does not reach `docker logs`.
LOG=/tmp/coralnet-startup.log
exec > >(tee -a "$LOG") 2>&1
echo "[custom_startup] $(date -Is) starting"

# KasmVNC provides a real X display, so use the normal xcb platform plugin.
# (The old start-coralnet.sh set QT_QPA_PLATFORM=vnc, which starts Qt's *own*
# VNC server on :5900 and ignores DISPLAY entirely -- that was why nothing
# ever showed up in the browser.)
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1

# Belt and braces alongside the opencv-python removal in the Dockerfile: pin
# the plugin path to PyQt5's own, so no dependency that bundles a stray
# libqxcb.so can win the search order.
export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins/platforms

export XDG_RUNTIME_DIR="/tmp/runtime-$(id -un)"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

export PATH="/opt/venv/bin:$PATH"

# desktop_ready only waits for the xfce4-session PID to appear, which happens
# well before the display will accept clients. Qt does not retry -- it aborts
# with SIGABRT and a misleading "could not load the Qt platform plugin xcb".
# So wait for a connection that actually succeeds.
echo "[custom_startup] waiting for desktop session..."
/usr/bin/desktop_ready
for i in $(seq 1 120); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        echo "[custom_startup] display $DISPLAY ready after ${i}s"
        break
    fi
    sleep 1
done
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "[custom_startup] FATAL: display $DISPLAY never became ready"
    exit 1
fi

# Restart on exit so a stray window-close doesn't tear down the whole session.
while true; do
    echo "[custom_startup] launching coralnet-toolbox"
    coralnet-toolbox
    echo "[custom_startup] coralnet-toolbox exited ($?); restarting in 3s..."
    sleep 3
done

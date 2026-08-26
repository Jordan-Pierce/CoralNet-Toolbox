# syntax=docker/dockerfile:1
#
# CoralNet-Toolbox, streamed to a browser.
#
# Uses KasmVNC rather than TightVNC + websockify + noVNC: one component instead
# of three, better compression for a pan/zoom annotation canvas, built-in auth,
# a working clipboard, and file upload/download in the web toolbar.
#
#   docker build -t coralnet-toolbox .
#   docker run --rm -it --gpus all --shm-size=2g -p 6901:6901 \
#       -e VNC_PW=coralnet -v "$PWD/data:/home/kasm-user/data" coralnet-toolbox
#
#   -> https://localhost:6901   (user: kasm_user, self-signed cert)

ARG KASM_VERSION=1.19.0
FROM kasmweb/core-ubuntu-jammy:${KASM_VERSION}

# jammy is Ubuntu 22.04, whose system Python is 3.10 -- exactly the range
# pyproject.toml requires (>=3.10, <3.11). No conda needed.

USER root

ENV HOME=/home/kasm-default-profile
ENV STARTUPDIR=/dockerstartup
ENV INST_SCRIPTS=$STARTUPDIR/install
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR $HOME

# ---------------------------------------------------------------------------
# System packages: Python 3.10 and the runtime libs PyQt5's xcb plugin needs
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential curl ca-certificates gnupg \
        libgl1 libgl1-mesa-dri libglib2.0-0 libegl1 libdbus-1-3 \
        libfontconfig1 libxkbcommon-x11-0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 \
        libxcb-util1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Layer 1: torch, pinned to a CUDA build with Blackwell (sm_120) kernels.
#
# An RTX 5090 needs cu128 or newer. Letting ultralytics pull whatever torch
# PyPI defaults to is how you end up with "no kernel image is available for
# execution on the device" at runtime. Freeze the exact versions afterwards so
# the dependency install below cannot quietly swap them for another build.
# ---------------------------------------------------------------------------
ARG TORCH_CUDA=cu128
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
 && pip freeze | grep -iE '^(torch|torchvision)==' > /tmp/torch-constraints.txt \
 && cat /tmp/torch-constraints.txt

# ---------------------------------------------------------------------------
# Layer 2: application dependencies. Cached until requirements.txt changes,
# which keeps rebuilds cheap while iterating on the source below.
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/torch-constraints.txt

# ultralytics pulls opencv-python; other deps pull opencv-python-headless. Both
# unpack into the SAME cv2/ directory, and the GUI build additionally drops its
# own Qt5 plugins at cv2/qt/plugins. PyQt5 then finds cv2's libqxcb.so before
# its own, tries to load it against a mismatched Qt build, and the app dies:
#
#   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
#   ".../cv2/qt/plugins" even though it was found     -> SIGABRT (exit 134)
#
# cv2 is used here for array work, never for windows, so keep only headless.
RUN pip uninstall -y opencv-python opencv-python-headless  && pip install --no-cache-dir opencv-python-headless

# ---------------------------------------------------------------------------
# Layer 3: the toolbox itself, installed from THIS working tree.
#
# The previous Dockerfile ran `uv pip install coralnet-toolbox`, which pulls the
# published PyPI release -- so the image could never test local changes.
# ---------------------------------------------------------------------------
COPY . /opt/coralnet-toolbox
WORKDIR /opt/coralnet-toolbox
RUN pip install --no-cache-dir --no-deps -e .

# ---------------------------------------------------------------------------
# Optional: Chrome, required by the CoralNet download feature (selenium +
# webdriver_manager). Build with --build-arg INSTALL_CHROME=false to skip.
# ---------------------------------------------------------------------------
ARG INSTALL_CHROME=true
RUN if [ "$INSTALL_CHROME" = "true" ]; then \
        curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
          | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg && \
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
          > /etc/apt/sources.list.d/google-chrome.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends google-chrome-stable && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# ---------------------------------------------------------------------------
# Session startup
# ---------------------------------------------------------------------------
COPY docker/custom_startup.sh $STARTUPDIR/custom_startup.sh
RUN chmod +x $STARTUPDIR/custom_startup.sh

# ---------------------------------------------------------------------------
# Login credentials
#
# Kasm hardcodes the literal username `kasm_user` in five places: the VNC
# password entry, and the auth tokens for the audio, upload and gamepad
# sidecars. Only renaming the VNC one would let you log in but break the
# upload/download toolbar, so rewrite all five to honour $VNC_USER.
#
# The sed patterns are single-quoted so the build shell leaves them alone;
# the variables are expanded by vnc_startup.sh at container start.
# ---------------------------------------------------------------------------
ENV VNC_USER=user
ENV VNC_PW=password
RUN sed -i       -e 's/kasm_user:\$VNC_PW/${VNC_USER}:$VNC_PW/g'       -e 's/-u kasm_user -wo/-u ${VNC_USER} -wo/'       $STARTUPDIR/vnc_startup.sh  && echo "rewrote $(grep -c '\${VNC_USER}' $STARTUPDIR/vnc_startup.sh) references"  && ! grep -qE 'kasm_user[^_]' $STARTUPDIR/vnc_startup.sh

ENV QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1

# Hand the customized profile over to the runtime user (Kasm convention).
#
# Deliberately NOT chowning /opt/venv or /opt/coralnet-toolbox. `chown -R`
# rewrites metadata on every file it touches, and Docker stores a changed file
# as a whole new copy -- so recursing over the venv duplicated all ~9 GB of
# torch into an extra layer. Neither path needs to be writable at runtime;
# read+execute for other, which they already have, is enough.
RUN chown -R 1000:0 $HOME

# The editable install lives in a root-owned tree, so the runtime user cannot
# drop .pyc files beside the sources. Skip bytecode rather than fail quietly.
ENV PYTHONDONTWRITEBYTECODE=1

ENV HOME=/home/kasm-user
WORKDIR $HOME
RUN mkdir -p $HOME/data && chown -R 1000:0 $HOME

USER 1000
EXPOSE 6901

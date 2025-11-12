#!/bin/bash

# Start VNC server
export USER=root
export HOME=/root
vncserver :1 -geometry 1280x800 -depth 24

# Wait for VNC to start
sleep 2

# Start noVNC
/usr/bin/websockify --web /usr/share/novnc/ 6901 localhost:5901 &

# Activate conda environment and start the PyQt5 application
source /opt/conda/bin/activate coralnet && \
export QT_QPA_PLATFORM=vnc && \
export DISPLAY=:1 && \
python -c "from coralnet_toolbox.main import run; run()"
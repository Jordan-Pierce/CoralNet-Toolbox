#!/bin/bash

# Start VNC server
vncserver :1 -geometry 1280x800 -depth 24

# Start noVNC
/usr/bin/websockify --web /usr/share/novnc/ 6901 localhost:5901 &

# Activate conda environment and start the PyQt5 application
source /opt/conda/bin/activate coralnet && \
python -c "from coralnet_toolbox.gui import main; main()"
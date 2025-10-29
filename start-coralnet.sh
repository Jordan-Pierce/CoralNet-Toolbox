#!/bin/bash
# Start VNC server with geometry and depth
/usr/bin/vncserver -geometry 1920x1080 -depth 24
# Set display variable
export DISPLAY=:1
# Start the CoralNet-Toolbox GUI in the background
coralnet-toolbox &
# Keep the container running
tail -f /dev/null

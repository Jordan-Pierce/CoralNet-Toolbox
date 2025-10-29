# Use a pre-configured image with Ubuntu, XFCE, and VNC
FROM consol/ubuntu-xfce-vnc

# Set a VNC password (recommended for security)
ENV VNC_PW=coralnet

# Install uv and the CoralNet-Toolbox with uv
USER 0
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install uv
RUN uv pip install coralnet-toolbox

# Copy the startup script into the container
COPY start-coralnet.sh /usr/local/bin/start-coralnet.sh

# Change ownership of the script and make it executable
RUN chown 1000:1000 /usr/local/bin/start-coralnet.sh
RUN chmod +x /usr/local/bin/start-coralnet.sh

# Switch to the default user (id 1000) for security
USER 1000

# Set the command to run the startup script
CMD ["/usr/local/bin/start-coralnet.sh"]

# Expose the ports for noVNC (6901) and VNC (5901)
EXPOSE 6901
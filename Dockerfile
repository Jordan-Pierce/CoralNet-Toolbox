# Use a modern Ubuntu base
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV VNC_PW=coralnet
ENV DISPLAY=:1

# Install VNC, XFCE, and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    xfce4 \
    xfce4-goodies \
    tightvncserver \
    websockify \
    net-tools \
    libgl1-mesa-glx \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create and activate Python 3.10 environment
RUN conda create -n coralnet python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "coralnet", "/bin/bash", "-c"]

# Install coralnet-toolbox and PyQt5 dependencies in the conda environment
RUN pip install coralnet-toolbox

# Alternatively, if you need to use uv instead of pip:
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# RUN uv pip install coralnet-toolbox

# Set up VNC
RUN mkdir -p /root/.vnc
RUN echo "$VNC_PW" | vncpasswd -f > /root/.vnc/passwd
RUN chmod 600 /root/.vnc/passwd

# Copy and set up startup script
COPY start-coralnet.sh /usr/local/bin/start-coralnet.sh
RUN chmod +x /usr/local/bin/start-coralnet.sh

# Expose VNC and noVNC ports
EXPOSE 5901 6901

CMD ["/usr/local/bin/start-coralnet.sh"]
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
    novnc \
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

ENV PATH=/opt/conda/bin:$PATH

# Accept Conda Terms of Service
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

# Configure conda to use conda-forge as primary channel
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Create Python 3.10 environment with pip
RUN conda create -n coralnet python=3.10 pip -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "coralnet", "/bin/bash", "-c"]

# Install coralnet-toolbox and PyQt5 dependencies in the conda environment
RUN pip install uv
RUN uv pip install coralnet-toolbox

# Set up VNC
RUN mkdir -p /root/.vnc
RUN touch /root/.vnc/passwd
RUN chmod 600 /root/.vnc/passwd

# Copy and set up startup script
COPY start-coralnet.sh /usr/local/bin/start-coralnet.sh
RUN chmod +x /usr/local/bin/start-coralnet.sh
RUN apt-get update && apt-get install -y dos2unix && dos2unix /usr/local/bin/start-coralnet.sh && apt-get remove -y dos2unix && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Expose VNC and noVNC ports
EXPOSE 5901 6901

CMD ["/usr/local/bin/start-coralnet.sh"]
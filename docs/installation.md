## 💾 **How to Install**

### 🐍 Create Conda Environment (Recommended)

It's recommended to use `Anaconda` to create an environment for the `toolbox`:
```bash
# Create and activate an environment
conda create --name coralnet10 python=3.10 -y
conda activate coralnet10

# Install uv first
pip install uv
```

### ⚡ GPU Acceleration (Optional)

If you have an **NVIDIA GPU with CUDA**, you can install the corresponding versions of `CUDA` and `PyTorch` for full GPU acceleration.

Below is an example for CUDA 12.8:
```bash
# Install CUDA toolkit and compiler
conda install nvidia/label/cuda-12.8.0::cuda-nvcc -y
conda install nvidia/label/cuda-12.8.0::cuda-toolkit -y

# Install PyTorch with CUDA 12.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

For other CUDA versions and detailed installation, see:
- [`cuda-nvcc`](https://anaconda.org/nvidia/cuda-nvcc)
- [`cudatoolkit`](https://anaconda.org/nvidia/cuda-toolkit)
- [`torch`](https://pytorch.org/get-started/locally/)

### 📦 Install

Once this has finished, install the `toolbox` using `uv`:

```bash
# Install with uv (fastest)
uv pip install coralnet-toolbox
```

> **Fallback**: If `uv` fails, simply fall back to using `pip`:

```bash
# Install with pip
pip install coralnet-toolbox
```

### ▶️ Run

Finally, you can run the `toolbox` from the command line:

```bash
coralnet-toolbox
```

### 🎯 GPU Status Indicators

If `CUDA` is installed and `PyTorch` was built with it properly, you'll see a device indicator in the bottom-left corner of the toolbox:
- **🐢** CPU only
- **🐇** Single GPU (CUDA)
- **🚀** Multiple GPUs (CUDA)
- **🍎** Mac Metal (Apple Silicon)

*Click the icon to see available device details*

### 🔄 Upgrade

When opening the `toolbox`, you will be notified if there is an update available. To upgrade to a specific version, run:

```bash
# Upgrade to latest version
uv pip install -U coralnet-toolbox
```

Or upgrade to a specific version:
```bash
uv pip install -U coralnet-toolbox==[version_number]
```

> **Fallback**: If `uv` fails, use `pip` instead: `pip install -U coralnet-toolbox`

> **Note**: If you have `torch` installed with `CUDA`, adding `-U` may trigger a regression to the CPU version. If this occurs, uninstall `torch` and `torchvision`, and reinstall the CUDA versions.

## 🐍 Install from Source (GitHub Repository)

If you prefer to clone the repository and run the `toolbox` from the source code:

```bash
# Create and activate an environment
conda create --name coralnet10 python=3.10 -y
conda activate coralnet10

# Install git via conda (if not already installed)
conda install git -y

# Change to your desired directory
cd Documents

# Clone and enter the repository
git clone https://github.com/Jordan-Pierce/CoralNet-Toolbox.git
cd CoralNet-Toolbox

# Install CUDA requirements (if applicable)
conda install nvidia/label/cuda-12.8.0::cuda-nvcc -y
conda install nvidia/label/cuda-12.8.0::cuda-toolkit -y

# Install PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the toolbox in development mode
pip install -e .

# Run
coralnet-toolbox
```

### Updating Your Repository

To update your repository to match the current version on `main`:

```bash
# Navigate to repository directory
cd CoralNet-Toolbox

# Fetch latest changes
git fetch

# Pull updates from main
git pull

# Update your environment
uv pip install -e . -U
```

### Install from GitHub Directly

You can also install the `toolbox` from the GitHub repo without cloning:

```bash
# Install from main branch
uv pip install git+https://github.com/Jordan-Pierce/CoralNet-Toolbox.git@main -U

# Or install from a different branch (e.g., for testing experimental features)
uv pip install git+https://github.com/Jordan-Pierce/CoralNet-Toolbox.git@branch-name -U
```

## 🧹 Cleanup

### Remove a Package

To remove a problematic package:

```bash
uv pip uninstall package-name-here
```

### Delete and Reset Environment

To delete an old environment and start fresh:

```bash
# Deactivate the environment first
conda deactivate

# Delete the environment by name
conda env remove --name coralnet10

# Confirm when prompted
y
```

## ⚠️ MacOS Users

> **Version 1.0.0 and later** rely heavily on `PyQtADS`, which cannot be installed on macOS. **Do not upgrade from version 0.0.105** until this is resolved.

An optional workaround is to run `toolbox` through docker (see below).

## 🐳 Docker (run in a browser)

The `toolbox` can run inside a container and stream its interface to a web
browser over [KasmVNC](https://github.com/kasmtech/KasmVNC) - no local install,
no X server on your machine.

```bash
# Build (first build is large: torch + CUDA is several GB)
docker compose build

# Run, then open https://localhost:6901
docker compose up
```

Or without compose:

```bash
docker build -t coralnet-toolbox .

# Launcher: mounts ./data only if it exists, adds --gpus only if available
./docker/run.sh
```

Or fully by hand:

```bash
docker run --rm -it --gpus all --shm-size=2g -p 6901:6901     -e VNC_USER=user -e VNC_PW=password     -v "$PWD/data:/home/kasm-user/data"     coralnet-toolbox
```

Then browse to **https://localhost:6901**:

| | |
|---|---|
| Username | value of `VNC_USER` (default `user`) |
| Password | value of `VNC_PW` (default `password`) |

The certificate is self-signed, so the browser will warn on first visit.

**Notes**

- **Your images.** The app's file dialogs see the *container's* filesystem, not
  your machine's. `docker/run.sh` mounts `./data` at `~/data` when that folder
  exists; override the source with `CORALNET_DATA=/some/path`. You can also use
  the upload/download buttons in the KasmVNC toolbar.
- **GPU.** `--gpus all` requires the NVIDIA Container Toolkit. Without it the
  container still runs, on CPU. The image pins a CUDA build of torch via the
  `TORCH_CUDA` build arg (default `cu128`); Blackwell cards (RTX 50-series)
  require `cu128` or newer.
- **The app is the session.** Closing the toolbox window restarts it rather than
  ending the session. Stop the container to end it.
### Interface lockout

`LOCKOUT_LEVEL` controls how much of the container the user can reach. It is
read at container start, so one image serves every level.

| Level | Behaviour |
|---|---|
| `1` | Full XFCE desktop: panel, file manager, right-click menu, window decorations. For development. |
| `2` | **Default.** Kiosk: openbox only. No panel, no desktop, no file manager, no menus, no window decorations, no window-management keybindings. The toolbox fills the screen and cannot be closed or minimised. |
| `3` | Reserved for OS-level isolation. **Not implemented** - the container refuses to start rather than silently giving you level 2. |

```bash
docker run -e LOCKOUT_LEVEL=1 ...      # desktop, for debugging
LOCKOUT_LEVEL=1 ./docker/run.sh        # same via the launcher
```

Levels 1 and 2 lock the *interface*. They stop a user wandering off into the
desktop; they do not stop a determined one reaching the filesystem through the
application's own file dialogs. That is what level 3 is for.

- **One user per container.** Everyone pointed at the same port shares one
  screen and one mouse. Serving multiple people means one container each, behind
  a router - not covered here.
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
# Upgrade to latest version (coralnet-toolbox only)
uv pip install --upgrade coralnet-toolbox
```

Or upgrade to a specific version:
```bash
uv pip install coralnet-toolbox==[version_number]
```

> **Fallback**: If `uv` fails, use `pip` instead: `pip install --upgrade coralnet-toolbox`

> **Note**: Using `-U` or `--upgrade-all` upgrades **all packages**, which may trigger a regression to the CPU version of `torch`. To avoid this, use the commands above to upgrade only coralnet-toolbox. If you do experience a regression, uninstall `torch` and `torchvision`, and reinstall the CUDA versions.


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

## 🧊 Alternative: One-Command Install with `pixi`

> The `conda` + `uv` steps above are the supported default and what most users should follow.
> `pixi` is an alternative for those already comfortable with that tool.

[`pixi`](https://pixi.sh) builds the whole environment -- Python, `Qt`, `GDAL`, and a `CUDA` build of
`PyTorch` -- from a committed lock file, so every machine resolves to identical dependencies. No
`conda create`, no separate `PyTorch` index URL, no `CUDA` toolkit install.

```bash
# Clone and enter the repository
git clone https://github.com/Jordan-Pierce/CoralNet-Toolbox.git
cd CoralNet-Toolbox

# Build the environment from pixi.lock
pixi install

# Launch
pixi run start
```

GPU support is automatic: `pixi` selects a `CUDA` build of `PyTorch` when your driver supports it,
including Blackwell (RTX 50-series).

> **Platforms**: Windows and Linux only. macOS is not supported because `conda-forge` has no `qt5-advanced-docking-system` build for Apple Silicon. **macOS users should use Docker** (see below).


## ⚠️ macOS Users

> **Version 1.0.0 and later** rely heavily on `PyQtADS`, which cannot be installed on macOS. **Do not upgrade from version 0.0.105** until this is resolved.

The recommended workaround is to run the `toolbox` through Docker (see below). This sidesteps the problem entirely: `PyQtADS` and `Qt` run inside the Linux container and only the rendered interface reaches your browser, so nothing Qt related is installed on macOS at all.

### Docker on macOS

Two macOS caveats:

- **Build for `amd64`.** There is no `arm64` build of `PyQtADS` from any source,
  and Google ships no `arm64` Chrome `.deb`, so an Apple Silicon Mac must build
  and run the image emulated: add `--platform linux/amd64` to `docker build` /
  `docker run`, or set `platform: linux/amd64` on the service. Expect it to be
  slow.
- **CPU only.** No Mac has CUDA, so use the plain `docker compose up` shown
  below and never the GPU overlay. Passing `--build-arg TORCH_CUDA=cpu` also
  saves several GB of image that would go unused.


## 🐳 Docker (run in a browser)

The `toolbox` can run inside a container and stream its interface to a web
browser over [KasmVNC](https://github.com/kasmtech/KasmVNC) - no local install,
no X server on your machine.

**Recommended - works the same on every shell:**

```bash
# Build (first build is large: torch + CUDA is several GB)
docker compose build

# Run, then open https://localhost:6901
docker compose up
```

Compose resolves relative paths itself, so `./data` is mounted correctly on
Linux, macOS and Windows alike.

`docker-compose.yml` requests no GPU, so the command above starts on any host.
On a machine with an NVIDIA GPU and the NVIDIA Container Toolkit, add the
overlay to pass the card through:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

The toolbox shows the device it actually got in its bottom-left corner
(turtle = CPU, rabbit/rocket = CUDA), so it is easy to confirm.

**Or via a launcher.** Both mount `./data` only when that folder exists, and
add `--gpus` only when the NVIDIA runtime is available:

```bash
docker build -t coralnet-toolbox:local .

./docker/run.sh          # Linux, macOS, Git Bash
```

```bat
docker build -t coralnet-toolbox:local .

docker\run.cmd           :: Windows Command Prompt / PowerShell
```

`run.sh` is a bash script and will not run in `cmd.exe`; `run.cmd` is the
equivalent for Windows. Both honour `CORALNET_DATA`, `LOCKOUT_LEVEL`,
`VNC_USER`, `VNC_PW` and `PORT` as environment variables.

**Or fully by hand.** Note the image tag is `coralnet-toolbox:local`, and that
the current-directory variable differs per shell - `$(pwd)` is not defined in
`cmd.exe`, and passing it literally makes Docker reject the path:

```bash
# Linux / macOS / Git Bash
docker run --rm -it --gpus all --shm-size=2g -p 6901:6901 \
    -e VNC_USER=user -e VNC_PW=password \
    -v "$(pwd)/data:/home/kasm-user/data" \
    coralnet-toolbox:local
```

```powershell
# Windows PowerShell
docker run --rm -it --gpus all --shm-size=2g -p 6901:6901 `
    -e VNC_USER=user -e VNC_PW=password `
    -v "${PWD}/data:/home/kasm-user/data" `
    coralnet-toolbox:local
```

```bat
:: Windows Command Prompt
docker run --rm -it --gpus all --shm-size=2g -p 6901:6901 ^
    -e VNC_USER=user -e VNC_PW=password ^
    -v "%cd%\data:/home/kasm-user/data" ^
    coralnet-toolbox:local
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
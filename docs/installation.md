## **How to Install**

### Anaconda

It's recommended to use `Anaconda` to create an environment for the `toolbox`:
```bash
# cmd

# Create and activate an environment
conda create --name coralnet10 python=3.10 -y
conda activate coralnet10
```

### Install

Once this has finished, install the `toolbox` using `uv`: 

```bash
# cmd

# Install uv first
pip install uv

# Install with uv
uv pip install coralnet-toolbox
```

Although fast, `uv` is still relatively new; if this fails, simply fall back to using `pip`:

```bash
# cmd

# Install
pip install coralnet-toolbox
```

### CUDA

If you have `CUDA`, you should install the versions of `cuda-nvcc` and `cudatoolkit` that you
need, and then install the corresponding versions of `torch` and `torchvision`. Below is an example of how that can be
done using `CUDA` version 11.8:

```bash
# cmd

# Example for CUDA 11.8
conda install nvidia/label/cuda-11.8.0::cuda-nvcc -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Example for torch w/ CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

If `CUDA` is installed on your computer, and `torch` was built with it properly, you should see a `🐇` icon in the
`toolbox` instead of a `🐢`; if you have multiple `CUDA` devices available, you should see a `🚀` icon,
and if you're using a Mac with `Metal`, you should see an `🍎` icon (click on the icon to see the device information).

See here for more details on versions for the following:
- [`cuda-nvcc`](https://anaconda.org/nvidia/cuda-nvcc)
- [`cudatoolkit`](https://anaconda.org/nvidia/cuda-toolkit)
- [`torch`](https://pytorch.org/get-started/locally/)


### Run
Finally, you can run the `toolbox` from the command line:

```bash
# cmd

# Run
coralnet-toolbox
```

## GitHub Repository

If you prefer to clone the repository and run the `toolbox` from the source code, you can do so with the following:

```bash
# cmd

# Create and activate an environment
conda create --name coralnet10 python=3.10 -y
conda activate coralnet10

# Install git via conda, if not already installed
conda install git -y

# Change to the desired directory (e.g., Documents)
cd Documents

# Clone and enter the repository
git clone https://github.com/Jordan-Pierce/CoralNet-Toolbox.git
cd CoralNet-Toolbox

# Install the latest
pip install -e .

# Install CUDA requirements (if applicable)
conda install nvidia/label/cuda-11.8.0::cuda-nvcc -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Example for torch w/ CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade

# Run
coralnet-toolbox
```

To update your repository to match the current version on `main`, run `fetch` and `pull` commands:

```bash
# cmd

# Change to the proper directory
cd Coralnet-Toolbox

# Ask for the updates on main
git fetch

# Pull the updates from main
git pull

# Update your conda environment 
pip install -e . -U
```

Or, if you want to simply install the `toolbox` from the GitHub repo directly you can also do the following:

```bash
# cmd

pip install git+https://github.com/Jordan-Pierce/CoralNet-Toolbox.git@main -U

# replace @main with a different branch if you want to test out experimental code
```

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

Once this has finished, install the `toolbox`:
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
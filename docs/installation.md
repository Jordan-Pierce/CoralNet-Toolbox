## 💾 **How to Install**

### 🐍 Anaconda

It's recommended to use `Anaconda` to create an environment for the `toolbox`:
```bash
# cmd

# Create and activate an environment
conda create --name coralnet10 python=3.10 -y
conda activate coralnet10
```

### 📦 Install

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

### ⚡ CUDA

If you have `CUDA`, you should install the versions of `cuda-nvcc` and `cudatoolkit` that you
need, and then install the corresponding versions of `torch` and `torchvision`. Below is an example of how that can be
done using `CUDA` version 11.8:
```bash
# cmd

# Example for CUDA 11.8
conda install nvidia/label/cuda-11.8.0::cuda-nvcc -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Example for torch w/ CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

If `CUDA` is installed on your computer, and `torch` was built with it properly, you should see a `🐇` icon in the
`toolbox` instead of a `🐢`; if you have multiple `CUDA` devices available, you should see a `🚀` icon,
and if you're using a Mac with `Metal`, you should see an `🍎` icon (click on the icon to see the device information).

See here for more details on versions for the following:
- [`cuda-nvcc`](https://anaconda.org/nvidia/cuda-nvcc)
- [`cudatoolkit`](https://anaconda.org/nvidia/cuda-toolkit)
- [`torch`](https://pytorch.org/get-started/locally/)

### ▶️ Run

Finally, you can run the `toolbox` from the command line:

```bash
# cmd

# Run
coralnet-toolbox
```

### **How to Upgrade**

When opening the `toolbox`, you will be notified if there is an update available, and you have the _option_ to do so, 
if you so choose. To upgrade, run the following command from your terminal:

```bash
# cmd

uv pip install -U coralnet-toolbox==[enter_newest_version_here]
```

Again, fall back to using just `pip` and not `uv` if this fails.
# CoralNet-Toolbox

<div align="center">
  <p>
    <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/CoralNet_Toolbox.png" alt="CoralNet-Toolbox">
  </p>
</div>


<div align="center">

[![python-version](https://img.shields.io/pypi/pyversions/CoralNet-Toolbox.svg)](https://pypi.org/project/CoralNet-Toolbox)
[![version](https://img.shields.io/pypi/v/CoralNet-Toolbox.svg)](https://pypi.python.org/pypi/CoralNet-Toolbox)
[![pypi-passing](https://github.com/Jordan-Pierce/CoralNet-Toolbox/actions/workflows/pypi.yml/badge.svg)](https://pypi.org/project/CoralNet-Toolbox)
[![windows](https://github.com/Jordan-Pierce/CoralNet-Toolbox/actions/workflows/windows.yml/badge.svg)](https://pypi.org/project/CoralNet-Toolbox)
[![macos](https://github.com/Jordan-Pierce/CoralNet-Toolbox/actions/workflows/macos.yml/badge.svg)](https://pypi.org/project/CoralNet-Toolbox)
[![ubuntu](https://github.com/Jordan-Pierce/CoralNet-Toolbox/actions/workflows/ubuntu.yml/badge.svg)](https://pypi.org/project/CoralNet-Toolbox)
</div>


## Quick Start

Running the following command will install the `coralnet-toolbox`, which you can then run from the command line:
```bash
# cmd

# Install
pip install coralnet-toolbox

# Run
coralnet-toolbox
```

For further instructions please see the following:
- [Installation](https://jordan-pierce.github.io/CoralNet-Toolbox/installation)
- [Usage](https://jordan-pierce.github.io/CoralNet-Toolbox/usage)

<details open>
  <summary><h2><b>Watch the Video Demos</b></h2></summary>
  <p align="center">
    <a href="https://youtube.com/playlist?list=PLG5z9IbwhS5NQT3B2jrg3hxQgilDeZak9&feature=shared">
      <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/toolbox_qt.PNG" alt="Video Title" width="75%">
    </a>
  </p>
</details>

## TL;Dr

The `CoralNet-Toolbox` is an unofficial codebase that can be used to augment processes associated with those on
[CoralNet](https://coralnet.ucsd.edu/).

It uses✨[`Ultralytics`](https://github.com/ultralytics/ultralytics)🚀 as a  base, which is an open-source library for
computer vision and deep learning built in `PyTorch`. For more information on their `AGPL-3.0` license, see
[here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

The `toolbox` also uses the following to create rectangle and polygon annotations:
- [`Fast-SAM`](https://github.com/CASIA-IVA-Lab/FastSAM)
- [`RepViT-SAM`](https://github.com/THU-MIG/RepViT)
- [`EdgeSAM`](https://github.com/chongzhou96/EdgeSAM)
- [`MobileSAM`](https://github.com/ChaoningZhang/MobileSAM)
- [`CoralSCOP`](https://github.com/zhengziqiang/CoralSCOP)
- [`SAM`](https://github.com/facebookresearch/segment-anything)
- [`AutoDistill`](https://github.com/autodistill)
  - [`GroundingDino`](https://github.com/IDEA-Research/GroundingDINO)


## Tools

Enhance your CoralNet experience with these tools:
- ✏️ Annotate: Create annotations freely
- 👁️ Visualize: See CoralNet and CPCe annotations superimposed on images
- 🔬 Sample: Sample patches using various methods (Uniform, Random, Stratified)
- 🧩 Patches: Create patches (points)
- 🔳 Rectangles: Create rectangles (bounding boxes)
- 🟣 Polygons: Create polygons (instance masks)
- 🦾 SAM: Use `FastSAM`, `CoralSCOP`, `RepViT-SAM`, `EdgeSAM`, `MobileSAM`, and `SAM` to create polygons
- 🧪 AutoDistill: Use `AutoDistill` to access `GroundingDINO` for creating rectangles
- 🧠 Train: Build local patch-based classifiers, object detection, and instance segmentation models
- 🔮 Deploy: Use trained models for predictions
- 📊 Evaluation: Evaluate model performance
- 🚀 Optimize: Productionize models for faster inferencing
- ⚙️ Batch Inference: Perform predictions on multiple images, automatically
- ↔️ I/O: Import and Export annotations from / to CoralNet, Viscore, and TagLab
- 📸 YOLO: Import and Export YOLO datasets for machine learning

### TODO
- 🔍 API: Get predictions from any CoralNet source model
- 📥 Download: Retrieve source data from CoralNet
- 📤 Upload: Add images and annotations to CoralNet
- 📦 Toolshed: Access tools from the old repository


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

## [**About CoralNet**](https://coralnet.ucsd.edu/source/)
Coral reefs are vital ecosystems that support a wide range of marine life and provide numerous
benefits to humans. However, they are under threat due to climate change, pollution, overfishing,
and other factors. CoralNet is a platform designed to aid researchers and scientists in studying
these important ecosystems and their inhabitants.

CoralNet allows users to upload photos of coral reefs and annotate them with detailed information
about the coral species and other features present in the images. The platform also provides tools
for analyzing the annotated images, and create patch-based image classifiers.

The CoralNet-Toolbox is an unofficial tool developed to augment processes associated with analyses that
use CoralNet and Coral Point Count (CPCe).


## **Conclusion**
In summary, this repository provides a range of tools that can assist with interacting with
CoralNet and performing various tasks related to analyzing annotated images. These tools can be
useful for researchers and scientists working with coral reefs, as well as for students and
hobbyists interested in learning more about these important ecosystems.


## Citation

If used in project or publication, please attribute your use of this repository with the following:

```
@misc{CoralNet-Toolbox,
  author = {Pierce, Jordan and Edwards, Clinton and Rojano, Sarah and Cook, Sophie and Sweeney, Edward and Costa, Bryan and Vieham, Shay and Battista, Tim},
  title = {CoralNet-Toolbox},
  year = {2023},
  howpublished = {\url{https://github.com/Jordan-Pierce/CoralNet-Toolbox}},
  note = {GitHub repository}
}
```

---

## Disclaimer

This repository is a scientific product and is not official communication of the National
Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA
GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its
use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from
the use of this GitHub project will be governed by all applicable Federal law. Any reference to
specific commercial products, processes, or services by service mark, trademark, manufacturer, or
otherwise, does not constitute or imply their endorsement, recommendation or favoring by the
Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC
bureau, shall not be used in any manner to imply endorsement of any commercial product or activity
by DOC or the United States Government.


## License

Software code created by U.S. Government employees is not subject to copyright in the United States
(17 U.S.C. §105). The United States/Department of Commerce reserve all rights to seek and obtain
copyright protection in countries other than the United States for Software authored in its
entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to
Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the
Software outside of the United States.

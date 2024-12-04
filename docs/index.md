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
[![downloads](https://static.pepy.tech/badge/coralnet-toolbox)](https://pepy.tech/projects/coralnet-toolbox)

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
- 🦾 SAM: Use `FastSAM`, `RepViT-SAM`, `EdgeSAM`, `MobileSAM`, and `SAM` to create polygons
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

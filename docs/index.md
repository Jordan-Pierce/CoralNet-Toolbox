# CoralNet-Toolbox 🪸🧰

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

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <h3>🔍 Annotation</h3>
        <p>Create patches, rectangles, and polygons with AI assistance</p>
      </td>
      <td align="center" width="33%">
        <h3>🧠 AI-Powered</h3>
        <p>Leverage SAM, YOLOE, and various foundation models</p>
      </td>
      <td align="center" width="33%">
        <h3>🚀 Complete Workflow</h3>
        <p>From data collection to model training and deployment</p>
      </td>
    </tr>
  </table>
</div>

## 🚦 Quick Start

Running the following command will install the `coralnet-toolbox`, which you can then run from the command line:
```bash
# cmd

# Install
pip install coralnet-toolbox

# Run
coralnet-toolbox
```

## 📚 Guides

For further instructions please see the following guides:
- [Installation](https://jordan-pierce.github.io/CoralNet-Toolbox/installation)
- [Usage](https://jordan-pierce.github.io/CoralNet-Toolbox/usage)
- [Patch-based Image Classifier](https://jordan-pierce.github.io/CoralNet-Toolbox/classify)

<details open>
  <summary><h2><b>🎥 Watch the Video Demos</b></h2></summary>
  <p align="center">
    <a href="https://youtube.com/playlist?list=PLG5z9IbwhS5NQT3B2jrg3hxQgilDeZak9&feature=shared">
      <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/toolbox_qt.PNG" alt="Video Title" width="90%">
    </a>
  </p>
</details>

## ⏩ TL;Dr

The `CoralNet-Toolbox` is an unofficial codebase that can be used to augment processes associated with those on
[CoralNet](https://coralnet.ucsd.edu/).

It uses✨[`Ultralytics`](https://github.com/ultralytics/ultralytics)🚀 as a  base, which is an open-source library for
computer vision and deep learning built in `PyTorch`. For more information on their `AGPL-3.0` license, see
[here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

### 🚀 Supported Models

The `toolbox` integrates a variety of state-of-the-art models to help you create rectangle and polygon annotations efficiently. Below is a categorized overview of the supported models and frameworks:

<div align="center">

| Category                | Models                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------|
| **Trainable**           | - 🦾 [YOLOv3](https://docs.ultralytics.com/models/) <br> - 🦈 [YOLOv4](https://docs.ultralytics.com/models/) <br> - 🦅 [YOLOv5](https://docs.ultralytics.com/models/) <br> - 🐬 [YOLOv6](https://docs.ultralytics.com/models/) <br> - 🐢 [YOLOv7](https://docs.ultralytics.com/models/) <br> - 🐙 [YOLOv8](https://docs.ultralytics.com/models/) <br> - 🐠 [YOLOv9](https://docs.ultralytics.com/models/) <br> - 🦑 [YOLOv10](https://docs.ultralytics.com/models/) <br> - 🚀 [YOLO11](https://docs.ultralytics.com/models/) <br> - 🐳 [YOLO12](https://docs.ultralytics.com/models/) |
| **Segment Anything**    | - 🪸 [SAM](https://github.com/facebookresearch/segment-anything) <br> - 🌊 [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP) <br> - ⚡ [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) <br> - 🔁 [RepViT-SAM](https://github.com/THU-MIG/RepViT) <br> - ✂️ [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) <br> - 📱 [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |
| **Visual Prompting**    | - 👁️ [YOLOE](https://github.com/THU-MIG/yoloe) <br> - 🤖 [AutoDistill](https://github.com/autodistill): <br> &nbsp;&nbsp;&nbsp;• 🦒 Grounding DINO <br> &nbsp;&nbsp;&nbsp;• 🦉 OWLViT <br> &nbsp;&nbsp;&nbsp;• ⚡ OmDetTurbo |

</div>

These models enable fast, accurate, and flexible annotation workflows for a wide range of use cases for patch-based image classification, object detection, instance segmentation.

## 🛠️ Toolbox Features

<div align="center">

| ![Patch Annotation Tool](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Patches.gif)<br><sub>**Patch Annotation**</sub> | ![Rectangle Annotation Tool](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Rectangles.gif)<br><sub>**Rectangle Annotation**</sub> | ![Polygon Annotation Tool](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Polygons.gif)<br><sub>**Polygon Annotation**</sub> |
|:--:|:--:|:--:|
| ![Patch-based Image Classification](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classification.gif)<br><sub>**Image Classification**</sub> | ![Object Detection](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Object_Detection.gif)<br><sub>**Object Detection**</sub> | ![Instance Segmentation](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Instance_Segmentation.gif)<br><sub>**Instance Segmentation**</sub> |
| ![Segment Anything Model (SAM)](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Segment_Anything.gif)<br><sub>**Segment Anything (SAM)**</sub> | ![Polygon Classification](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classifying_Polygons.gif)<br><sub>**Polygon Classification**</sub> | ![Patch-based LAI Classification](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classifying_Orthomosaics.gif)<br><sub>**Patch-based LAI Classification**</sub> |
| ![Cut](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Cut.gif)<br><sub>**Cut**</sub> | ![Combine](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Combine.gif)<br><sub>**Combine**</sub> | ![See Anything (YOLOE)](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/See_Anything.gif)<br><sub>**See Anything (YOLOE)**</sub> |
|  | ![Region-based Detection](https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Work_Areas.gif)<br><sub>**Region-based Detection**</sub> |  |

</div>

Enhance your CoralNet experience with these tools:
- 📥 [Download](https://www.youtube.com/watch?v=Ds9JZATmCmw): Retrieve Source data (images and annotations) from CoralNet
- 🎬 Rasters: Import images, or extract frames directly from video files
- ✏️ Annotate: Create annotations freely
- 👁️ Visualize: See CoralNet and CPCe annotations superimposed on images
- 🔬 Sample: Sample patches using various methods (Uniform, Random, Stratified)
- 🧩 Patches: Create patches (points)
- 🔳 Rectangles: Create rectangles (bounding boxes)
- 🟣 Polygons: Create polygons (instance masks)
- ✍️ Edit: Cut and Combine polygons and rectangles
- 🦾 SAM: Use `FastSAM`, `CoralSCOP`, `RepViT-SAM`, `EdgeSAM`, `MobileSAM`, and `SAM` to create polygons
  - Uses [`xSAM`](https://github.com/Jordan-Pierce/xSAM)
- 👀 YOLOE (See Anything): Detect similar appearing objects using visual prompts automatically
- 🧪 AutoDistill: Use `AutoDistill` to access the following for creating rectangles and polygons:
  - Uses `Grounding DINO`, `OWLViT`, `OmDetTurbo`
- 🧠 Train: Build local patch-based classifiers, object detection, and instance segmentation models
- 🔮 Deploy: Use trained models for predictions
- 📊 Evaluation: Evaluate model performance
- 🚀 Optimize: Productionize models for faster inferencing
- ⚙️ Batch Inference: Perform predictions on multiple images, automatically
- ↔️ I/O: Import and Export annotations from / to CoralNet, Viscore, and TagLab
  - Export annotations as [GeoJSONs](https://datatracker.ietf.org/doc/html/rfc7946), segmentation masks
- 📸 YOLO: Import and Export YOLO datasets for machine learning
- 🧱 Tile Dataset: Tile existing Detection / Segmentation datasets
  - Uses [`yolo-tiling`](https://github.com/Jordan-Pierce/yolo-tiling)

### 📝 TODO
- 🤗 Model Zoo: Download `Ultralytics` models from `HuggingFace` for use in `toolbox`
- 🦊 BioCLIP, MobileCLIP (AutoDistill): Automatically classify annotations
- 📦 [Toolshed: Access tools from the old repository](https://github.com/Jordan-Pierce/CoralNet-Toolshed)
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


## Quick Start

Running the following command will install the `coralnet-toolbox`, which you can then run from the command line:
```bash
# cmd

# Install
pip install coralnet-toolbox

# Run
coralnet-toolbox
```

For further instructions please see the following guides:
- [Installation](https://jordan-pierce.github.io/CoralNet-Toolbox/installation)
- [Usage](https://jordan-pierce.github.io/CoralNet-Toolbox/usage)
- [Patch-based Image Classifier](https://jordan-pierce.github.io/CoralNet-Toolbox/classify)

<details open>
  <summary><h2><b>Watch the Video Demos</b></h2></summary>
  <p align="center">
    <a href="https://youtube.com/playlist?list=PLG5z9IbwhS5NQT3B2jrg3hxQgilDeZak9&feature=shared">
      <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/toolbox_qt.PNG" alt="Video Title" width="90%">
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
- [`YOLOE`](https://github.com/THU-MIG/yoloe)
- [`AutoDistill`](https://github.com/autodistill)
  - [`Grounding Dino`](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)
  - [`OWLViT`](https://huggingface.co/docs/transformers/en/model_doc/owlvit)
  - [`OmDetTurbo`](https://huggingface.co/docs/transformers/en/model_doc/omdet-turbo)


## Tools

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Patches.gif" height="200"/>
        <br>
        <em>Patch Annotation Tool</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Rectangles.gif" height="200"/>
        <br>
        <em>Rectangle Annotation Tool</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Polygons.gif" height="200"/>
        <br>
        <em>Polygon Annotation Tool</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classification.gif" height="200"/>
        <br>
        <em>Patch-based Image Classification</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Object_Detection.gif" height="200"/>
        <br>
        <em>Object Detection</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Instance_Segmentation.gif" height="200"/>
        <br>
        <em>Instance Segmentation</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Segment_Anything.gif" height="200"/>
        <br>
        <em>Segment Anything Model (SAM)</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classifying_Polygons.gif" height="200"/>
        <br>
        <em>Polygon Classification</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classifying_Orthomosaics.gif" height="200"/>
        <br>
        <em>Patch-based LAI Classification</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Cut.gif" height="200"/>
        <br>
        <em>Cut</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Combine.gif" height="200"/>
        <br>
        <em>Combine</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/See_Anything.gif" height="200"/>
        <br>
        <em>See Anything (YOLOE)</em>
      </td>
    </tr>
  </table>
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

### TODO
- 🤗 Model Zoo: Download `Ultralytics` models from `HuggingFace` for use in `toolbox`
- 🦊 BioCLIP, MobileCLIP (AutoDistill): Automatically classify annotations
- 📦 [Toolshed: Access tools from the old repository](https://github.com/Jordan-Pierce/CoralNet-Toolshed)

# CoralNet-Toolbox 🪸🧰

<div align="center">
<img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/coralnet_toolbox.png" alt="CoralNet-Toolbox" width="300">
<p><strong>AI-Powered Annotation for Coral Reef Analysis</strong>. An unofficial toolkit to supercharge your <a href="https://coralnet.ucsd.edu/">CoralNet</a> workflows.</p>
</div>

<div align="center">

[![Python Version](https://img.shields.io/pypi/pyversions/CoralNet-Toolbox.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/CoralNet-Toolbox)
[![Version](https://img.shields.io/pypi/v/CoralNet-Toolbox.svg?style=for-the-badge&color=blue)](https://pypi.python.org/pypi/CoralNet-Toolbox)
[![GitHub last commit](https://img.shields.io/github/last-commit/Jordan-Pierce/CoralNet-Toolbox?style=for-the-badge)](https://github.com/Jordan-Pierce/CoralNet-Toolbox)
[![Downloads](https://img.shields.io/pepy/dt/coralnet-toolbox.svg?style=for-the-badge&color=brightgreen)](https://pepy.tech/project/coralnet-toolbox)

[![PyPI Passing](https://img.shields.io/github/actions/workflow/status/Jordan-Pierce/CoralNet-Toolbox/pypi.yml?style=for-the-badge&label=PyPI%20Build&logo=github)](https://pypi.org/project/CoralNet-Toolbox)
[![Windows](https://img.shields.io/github/actions/workflow/status/Jordan-Pierce/CoralNet-Toolbox/windows.yml?style=for-the-badge&label=Windows&logo=windows&logoColor=white)](https://pypi.org/project/CoralNet-Toolbox)
[![macOS](https://img.shields.io/github/actions/workflow/status/Jordan-Pierce/CoralNet-Toolbox/macos.yml?style=for-the-badge&label=macOS&logo=apple&logoColor=white)](https://pypi.org/project/CoralNet-Toolbox)
[![Ubuntu](https://img.shields.io/github/actions/workflow/status/Jordan-Pierce/CoralNet-Toolbox/ubuntu.yml?style=for-the-badge&label=Ubuntu&logo=ubuntu&logoColor=white)](https://pypi.org/project/CoralNet-Toolbox)

</div>

## ⚡ Get Started

**1. Create Conda Environment (Recommended)**

```bash
# Create and activate custom environment
conda create --name coralnet10 python=3.10 -y
conda activate coralnet10

# Install uv
pip install uv
```

**2. (Optional) GPU Acceleration**
If you have an NVIDIA GPU with CUDA, install PyTorch with CUDA support for full acceleration.

```bash
# Example for CUDA 12.9; use your version of CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

**3. Install**

```bash
# Use UV for the fastest installation
uv pip install coralnet-toolbox
```
> **Fallback**: If UV fails, use regular pip: `pip install coralnet-toolbox`

**4. Launch**

```bash
coralnet-toolbox
```

*See the [Installation Guide](https://jordan-pierce.github.io/CoralNet-Toolbox/installation) for details on other versions.*


### 🎯 **GPU Status Indicators**
- **🐢** CPU only
- **🐇** Single GPU
- **🚀** Multiple GPUs  
- **🍎** Mac Metal (Apple Silicon)

*Click the icon in the bottom-left to see available devices*

### 🔄 **Upgrading**
```bash
# When updates are available
uv pip install -U coralnet-toolbox==[latest_version]
```

-----

## 📚 Resources & Advanced Details


* [**Installation Guide**](https://jordan-pierce.github.io/CoralNet-Toolbox/installation): Detailed setup for CUDA, Windows, and Mac.
* [**User Manual**](https://jordan-pierce.github.io/CoralNet-Toolbox/usage): A complete guide to all tools and features.
* [**Hot Keys**](https://jordan-pierce.github.io/CoralNet-Toolbox/hot-keys): Keyboard shortcuts to accelerate your workflow.
* [**AI Tutorial**](https://jordan-pierce.github.io/CoralNet-Toolbox/classify): Learn to train your own classification models.

<details open>
<summary><h3>📺 <strong>Watch the Demo Videos</strong></h3></summary>

<div align="center">
  <a href="https://youtube.com/playlist?list=PLG5z9IbwhS5NQT3B2jrg3hxQgilDeZak9&feature=shared">
    <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/toolbox_qt.PNG" alt="Video Tutorial Series" width="80%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  </a>
  
  <p><strong>🎬 Complete playlist covering all major features and workflows</strong></p>
</div>
</details>

## From Bottleneck to Pipeline

Traditional benthic imagery analysis is time-consuming. Manual annotation, data management, and model training are often separate, complex tasks. CoralNet-Toolbox unifies this process, turning a research bottleneck into an integrated, AI-accelerated pipeline.

<div align="center">

### 📝 **Core Annotation Tools**

| <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Patches.gif" alt="Patch Annotation" width="250" style="border-radius: 8px;"/><br>**🎯 Patch Annotation** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Rectangles.gif" alt="Rectangle Annotation" width="250" style="border-radius: 8px;"/><br>**📐 Rectangle Annotation** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Polygons.gif" alt="Polygon Annotation" width="250" style="border-radius: 8px;"/><br>**🔷 Multi-Polygon Annotation** |
|:---:|:---:|:---:|

### 🤖 **AI-Powered Analysis**

| <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classification.gif" alt="Classification" width="250" style="border-radius: 8px;"/><br>**🧠 Image Classification** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Object_Detection.gif" alt="Object Detection" width="250" style="border-radius: 8px;"/><br>**🎯 Object Detection** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Instance_Segmentation.gif" alt="Instance Segmentation" width="250" style="border-radius: 8px;"/><br>**🎭 Instance Segmentation** |
|:---:|:---:|:---:|

### 🔬 **Advanced Capabilities**

| <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Segment_Anything.gif" alt="SAM" width="250" style="border-radius: 8px;"/><br>**🪸 Segment Anything (SAM)** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Classifying_Polygons.gif" alt="Polygon Classification" width="250" style="border-radius: 8px;"/><br>**🔍 Polygon Classification** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Work_Areas.gif" alt="Work Areas" width="250" style="border-radius: 8px;"/><br>**📍 Region-based Detection** |
|:---:|:---:|:---:|

### ✂️ **Editing & Processing Tools**

| <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Cut.gif" alt="Cut Tool" width="250" style="border-radius: 8px;"/><br>**✂️ Cut** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Combine.gif" alt="Combine Tool" width="250" style="border-radius: 8px;"/><br>**🔗 Combine** | <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/figures/tools/Simplify.gif" alt="Simplify Tool" width="250" style="border-radius: 8px;"/><br>**🎨 Simplify** |
|:---:|:---:|:---:|

</div>

## 🌊 Success Stories

> **Using CoralNet-Toolbox in your research?** 
> 
> We'd love to feature your work! Share your success stories to help others learn and get inspired.

---

### 🏗️ **Repository Structure**

<div align="center">
  <a href="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/diagram.svg">
    <img src="https://raw.githubusercontent.com/Jordan-Pierce/CoralNet-Toolbox/refs/heads/main/diagram.svg" alt="Visualization of the codebase" width="80%">
  </a>
</div>

---

## 🌍 About CoralNet

Coral reefs are among Earth's most biodiverse ecosystems, supporting marine life and coastal communities worldwide. However, they face unprecedented threats from climate change, pollution, and human activities.

**[CoralNet](https://coralnet.ucsd.edu/)** is a revolutionary platform enabling researchers to:
- Upload and analyze coral reef photographs
- Create detailed species annotations
- Build AI-powered classification models
- Collaborate with the global research community

The **CoralNet-Toolbox** extends this mission by providing advanced AI tools that accelerate research and improve annotation quality.

---

## 📄 Citation

If you use CoralNet-Toolbox in your research, please cite:

```bibtex
@misc{CoralNet-Toolbox,
  author = {Pierce, Jordan and Battista, Tim and Kuester, Falko},
  title = {CoralNet-Toolbox: Tools for Annotating and Developing Machine Learning Models for Benthic Imagery},
  year = {2025},
  howpublished = {\url{https://github.com/Jordan-Pierce/CoralNet-Toolbox}},
  note = {GitHub repository}
}
```

---

## ⚖️ Legal & Licensing

<div align="center">

### ⚠️ **Disclaimer**
*This is a scientific product and not official communication of NOAA or the US Department of Commerce. All code is provided 'as is' - users assume responsibility for its use.*

### 📋 **License**
*Software created by US Government employees is not subject to copyright in the United States (17 U.S.C. §105). The Department of Commerce reserves rights to seek copyright protection in other countries.*

</div>

---

<div align="center">
  <p><em>Empowering researchers • Protecting ecosystems • Advancing science</em></p>
</div>
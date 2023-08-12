# CoralNet Toolbox  


<p align="center">
  <img src="./Figures/CoralNet-Toolbox.png" alt="CoralNet-Toolbox">
</p>

---

### [**About CoralNet**](https://coralnet.ucsd.edu/source/)
Coral reefs are vital ecosystems that support a wide range of marine life and provide numerous 
benefits to humans. However, they are under threat due to climate change, pollution, overfishing, 
and other factors. CoralNet is a platform designed to aid researchers and scientists in studying 
these important ecosystems and their inhabitants.

CoralNet allows users to upload photos of coral reefs and annotate them with detailed information 
about the coral species and other features present in the images. The platform also provides tools 
for analyzing the annotated images, and create patch-based image classifiers. 

### [**Tools**](./Toolbox/README.md#coralnet-toolbox)

The `CoralNet-Toolbox` is an **unofficial** codebase that can be used to augment processes associated
with those on CoralNet. The following scripts allow a user to run processes programmatically, or through 
a GUI; these scripts currently include:

- [`API`](./Toolbox/README.md#api): Use the CoralNet API to get predictions from any source model
- [`Download`](./Toolbox/README.md#download): Download all data associated with a source
- [`Upload`](./Toolbox/README.md#upload): Upload images, annotations, and labelsets to a source
- [`Labelset`](./Toolbox/README.md#labelset): Create a custom labelset on CoralNet
- [`Viscore`](./Toolbox/README.md#viscore): Upload annotations made in Viscore's VPI to CoralNet
- [`Classifier`](./Toolbox/README.md#classifier): Create your own patch-based image classifier, locally
- [`Annotate`](./Toolbox/README.md#annotation): Create your own patches from annotations, locally
- [`Visualize`](./Toolbox/README.md#visualize): Visualize points/patches superimposed on images
- [`Patches`](./Toolbox/README.md#patches): Extract patches from images given an annotation file
- [`Points`](./Toolbox/README.md#points): Sample points from images (Uniform, Random, Stratified)
- [`Inference`](./Toolbox/README.md#inference): Perform inference using a locally trained model
- [`SAM`](./Toolbox/README.md#sam): Create segmentation masks for each image using `MSS` w/ `SAM`
- [`SfM`](./Toolbox/README.md#sfm): Use Metashape to create 3D models (sure, why not)
- [`3DSeg`](./Toolbox/README.md#3dseg): Use masks and SfM to create classified 3D models


<p align="center">
  <img src="Figures/toolbox_api.PNG" alt="CoralNet-Toolbox-Features">
</p>

#### **Future Features**
- `Analysis`: Calculate CPCe statistics from locally trained model's predictions
- `Seg`: Create a FCN to perform semantic segmentation 
- `Clean`: Use `CleanLab.ai` to identify potentially incorrectly labeled patches
- `GAN`: Synthesize samples using generative AI
- `GPT`: LLMs for QA of annotation dataframes, plot visualizations

#### **How to use**
To use these tools, you should have access to the CoralNet platform. Once you have an account, 
you can use the `CoralNet-Toolbox` codebase to programmatically interact with the platform and perform 
various tasks.

To install, use the `coralnet_toolbox.yml` file using anaconda:
```python
# cmd
conda env create -f coralnet_toolbox.yml
conda activate coralnet_toolbox
python Toolbox\toolbox.py
```

Note that the `CoralNet-Toolbox` has ***only*** been tested on the following:
- `Windows 11`
- `Python 3.8`
- `Torch 2.0.1 + CUDA 11.8`
- `Tensorflow 2.10.1 + CUDA 11.8`
- `Keras 2.10.0 + CUDA 11.8`
- `Metashape Professional 2.0.X`
- `Google Chrome 114`

### **Conclusion**
In summary, this repository provides a range of tools that can assist with interacting with 
CoralNet and performing various tasks related to analyzing annotated images. These tools can be 
useful for researchers and scientists working with coral reefs, as well as for students and
hobbyists interested in learning more about these important ecosystems.

### Citation

If used in project or publication, please attribute your use of this repository with the following:
    
```
@misc{CoralNet-Toolbox,
  author = {Pierce, Jordan and Edwards, Clint and Vieham, Shay and Rojano, Sarah and Cook, Sophie and Battista, Tim},
  title = {CoralNet Tools},
  year = {2022},
  howpublished = {\url{https://github.com/Jordan-Pierce/CoralNet_Tools}},
  note = {GitHub repository}
}
```

### References  

The following papers inspired this repository:
```python
Pierce, J., Butler, M. J., Rzhanov, Y., Lowell, K., &amp; Dijkstra, J. A. (2021).
Classifying 3-D models of coral reefs using structure-from-motion and multi-view semantic segmentation.
Frontiers in Marine Science, 8. https://doi.org/10.3389/fmars.2021.706674

Pierce, J. P., Rzhanov, Y., Lowell, K., &amp; Dijkstra, J. A. (2020).
Reducing annotation times: Semantic Segmentation of coral reef survey images.
Global Oceans 2020: Singapore – U.S. Gulf Coast. https://doi.org/10.1109/ieeeconf38699.2020.9389163

Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I., Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J., Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., &amp; Kriegman, D. (2015).
Towards automated annotation of benthic survey images: Variability of human experts and operational modes of automation.
PLOS ONE, 10(7). https://doi.org/10.1371/journal.pone.0130312
```
---

### Disclaimer

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


### License 

Software code created by U.S. Government employees is not subject to copyright in the United States 
(17 U.S.C. §105). The United States/Department of Commerce reserve all rights to seek and obtain 
copyright protection in countries other than the United States for Software authored in its 
entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to 
Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the 
Software outside of the United States.

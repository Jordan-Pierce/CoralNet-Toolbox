# CoralNet_Tools  


<p align="center">
  <img src="./Figures/CoralNet_Tools_Logo.png" alt="CoralNet_Tools_Logo">
</p>

---

### **About CoralNet**
Coral reefs are vital ecosystems that support a wide range of marine life and provide numerous 
benefits to humans. However, they are under threat due to climate change, pollution, overfishing, 
and other factors. CoralNet is a platform designed to aid researchers and scientists in studying 
these important ecosystems and their inhabitants.

CoralNet allows users to upload photos of coral reefs and annotate them with detailed information 
about the coral species and other features present in the images. The platform also provides tools 
for analyzing the annotated images, such as image classification and semantic segmentation.

### **Tools Available in this Repository**
This repository contains a collection of tools that can be used to interact with CoralNet and 
perform various tasks related to analyzing the annotated images. The following is a list of the 
tools currently available:

#### **CoralNet_3D**
CoralNet_3D is a set of tools for representing single images as 3D point clouds. Images are 
provided depth maps using MiDAS and represented as a point cloud with RGB color component values 
coming from the original images. This tool can be useful for visualizing the annotated images in 3D 
and analyzing their structure.

#### **CoralNet_API**
CoralNet_API is a Python library for accessing the CoralNet API, which allows users to 
programmatically interact with CoralNet and perform tasks such as uploading and downloading data, 
and annotating images. This library can be used to automate tasks and integrate CoralNet 
functionality into other applications.

#### **Image_Classifier**
Image_Classifier is a set of tools for training your own patch-based image classifier using the 
images and annotations downloaded from CoralNet, or those that you create using the Patch_Extractor 
tool. This tool can be used to classify images based on the presence of specific features or coral
species.

#### **Patch_Extractor**
Patch_Extractor is a tool for dividing annotated images into smaller patches, which can be useful 
for training machine learning models on the annotated data. This tool can be used to generate 
training datasets for image classification or semantic segmentation.

#### **Semantic_Segmentation**
Semantic_Segmentation is a set of tools for training a semantic segmentation algorithm using the 
images and annotations downloaded from CoralNet, or those you create using the Patch_Extractor 
tool. This tool can be used to segment images into different regions based on the presence of 
specific features or coral species.

#### **How to use**
To use these tools, you need to have access to the CoralNet platform. Once you have an account, 
you can use the CoralNet_API library to programmatically interact with the platform and perform 
various tasks.

### **Conclusion**
In summary, this repository provides a range of tools that can assist with interacting with 
CoralNet and performing various tasks related to analyzing annotated images. These tools can be 
useful for researchers and scientists working with coral reefs, as well as for students and
hobbyists interested in learning more about these important ecosystems.

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

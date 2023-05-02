### The Patch Extractor tool

- Whole images ( .bmp, .png, .jpg) can be dragged and dropped into the GUI;
- Users can then adjust the desired patch size (defaults to 224 x 224 pixels), then press Apply;
- Use the Base name and Type to differentiate between class categories as needed;
- Left-clicking the mouse will extract a patch: this is saved on your local machine where the original image is located;
- For each patch, the following information will be concatenated to a .txt file: original image name, top-left X, Y, and image patch name.  
- Resolution of the local machine needs to be adjusted to be 100% for the entire GUI to be within screen.
- Use the `Converting_CSV` script to convert the output .txt file into a format that can be uploaded to CoralNet.

**This tool was developed by Dr. Yuri Rzhanov of the Center for Coastal and Ocean Mapping/Joint Hydrographic Center.**

### Converting

This notebook can be used to convert and merge the .txt files output from 
the Patch Exaction Tool. The .txt files have an xy value that represents 
the top-left corner (i.e., origin) of the patch extracted, though for 
CoralNet, these need to be the center. The notebook expects all .txt files 
to placed in the folder `TXT_Files`, and the `patch_size` variable to be 
set the size the patch was when extracted.

Alternatively, you can use this script in Google Colab, which you can find 
[here](https://colab.research.google.com/drive/19hfFQL3B-J3xQDe-UnpWs1YX3hfb3PFA?usp=sharing).
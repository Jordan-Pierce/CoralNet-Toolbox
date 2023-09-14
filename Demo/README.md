# Demo

This showcases the steps to using the CoralNet-Toolbox. In this example, we use the 
[D3](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html) dataset, which contains 68 images of a 
small 3D scene. The images were downloaded and converted into `.jpg` format before this demonstration.

### Setup

Before starting, the Toolbox must be downloaded or pulled from GitHub. Once you have the codebase on your local machine,
proceed to installing all the dependencies required. This can be done using the conda environment file like so:

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

It's important to note that some tools **require** a NVIDIA GPU with CUDA enabled. If you do not have this type
of GPU, these tools will not be available, and installing will likely be cumbersome. 

If you run into issues installing, please leave a ticket in the `Issues` tab of the repo on GitHub.

### Creating a Source

Once everything is installed and you can open `toolbox.py`, we move on to setting up a project on CoralNet. Projects are
referred to as `Sources`; in this demo we create a source on CoralNet and are provided with the source ID `4349`. This
is a standard process, and more information on creating a source can be found on CoralNet.

Image

After creating a source, we create a labelset. The labelset is a set of labels that represents all class categories 
present in our image set. Below you can see that for this source, we created a small labelset for demonstration 
purposes.

Image

After creating the labelset, we used the CoralNet-Toolbox to `Download` the labelset onto our local machine, which
created the source folder structure. We can now `Upload` our image set to CoralNet using the CoralNet-Toolbox. Below
you can see that all images were uploaded successfully. 

Image

### Annotations

This image set does not contain any labels, so we'll make some within the CoralNet-Toolbox. This is done using the
`Annotation` tool, which opens up an executable for the `Patch Extraction` tool. With this tool, we create patches
manually for the different class categories present.

Image

It's important that the `Short Codes` entered in the `Annotation` tool match those of the labelset on CoralNet! Once 
finished, the output will be stored under the annotations folder with a timestamp; opening this up, you'll see
all the center points for the patches extracted. This annotation file can then be uploaded using the `Upload` tool.

### Visualize

If you want to doublecheck your work, you can use the `Visualize` tool to view the labels or predictions made, 
superimposed on the images. Toggle between the points and patches view, and save the figure to an output folder.

Image

### API

Once images and annotations are uploaded to CoralNet, with 24 hours the model will begin to train. You need at least
20 labeled images for this to occur. After the model is trained, you'll see it on the source page

Image

Now that we have a trained model on CoralNet, we can use it to provide predictions to any images currently on CoralNet.
To showcase this, we'll have the model make ~2000 predictions on each of the images. We specify to the model where to 
annotate on each image by creating a points file using the `Points` tool. Below you can see we're creating 2048 points
for each image in the directory, in a uniform sampling method.

Image

We can then visualize these points using the `Visualize` tool as a sanity check.

Image

Given these points, we'll use the `API` tool to get the model to make the predictions on each of the images. Below you
can see that we're passing the points file, which contains `Row`, `Column` and the image `Name` as it appears on 
CoralNet. Since the images are already on CoralNet, we'll use code from the `Download` tool to pass the image URLs
directly to the model.

Image

The `API` accepts 200 points per image, 100 images per job, and 5 jobs at a time. If there are more than 200 points for
an image, the image will have predictions made on it multiple times to satisfy this requirement. As the `API` makes 
predictions, the status will be continuously logged until it's been completed. The predictions will be output as a
predictions file on your local machine within a `predictions` folder.

Image

Again, as a sanity check, we can visualize these predictions using the `Visualize` tool.

Image

### SAM

These predictions won't be wasted! They can be used for analysis, or to create segmentation masks for each image.
The CoralNet-Toolbox uses the [Segment Anything Model (SAM)]() to create superpixels of homogenous regions within an 
image, and existing point annotations to create segmentation masks. These masks can be used for analysis, or to create
classified SfM models (more on that below). Opening up the `SAM` tool, we provide the directory containing images, and
the predictions made by the CoralNet `API`.

If you indicate to `Plot Progess`, you can see how the segmentation masks form (though this does increase runtime). The
output will be stored in a `seg` folder, which contains the segmentation masks (int8), segmentation masks (colorized),
and figures showing the colored masks superimposed on each image.

### SfM

For image sets of 3D scenes (as opposed to traditional benthic quadrat habitat surveys), the CoralNet-Toolbox also
has the ability to perform a SfM workflow using the Metashape Python API. The workflow consists of:

- image alignment
- depth maps
- dense cloud
- DEM
- mesh
- orthomosaic
- export

To perform the workflow, we can use the `SfM` tool to provide the directory containing the image set, and provide
the quality of the model to create. Very few parameters are exposed to the user, but this might change over time.
It's also important to note that because we're using the Metashape API to perform to workflow, a professional license
is required.

After running through the workflow, we can check the project to see the results:

Image

If we look in the output folder, we can find the project file (`.psx`), along with the exported data products.

### Seg3D

Since we've created segmentation masks using `SAM` and created a 3D model using `SfM`, we can generate a classified
3D model using the `Seg3D` tool fairly easily. The process involves swapping the original source images used in the
`SfM` process with the masks generated using `SAM`, and then re-projecting the pixel color component values to the
3D models instead. 

With the `Seg3D` tool, we just need to pass in the directory containing the colorized masks, and our existing `SfM` 
project file (`.psx`). The tool will first create a duplicate of the indicated chunk, and then perform the image / mask
swap, following by re-projecting the new colors to create a classified point cloud. Unfortunately because of blending,
this point cloud will need to be post-processed first before creating a classified mesh, and orthomosaic. Once 
post-processed, the updated classified point cloud will be imported back into the project replacing the previous one.
From here the mesh will be classified, followed by the orthomosaic. All classified data products will be exported in
the Metashape project folder.

# Other Tools

But wait, there's more! In addition to the tools shown above, there are some others that might be of use

### Labelset

CoralNet requires a project to create a labelset of officially registered labels. It's possible to use existing labels
that are already on CoralNet, or you can use the `labelset` tool to programmatically create your own. This tool is
useful when you have a large project with many class categories that do not have an exact match already on CoralNet.

### Patches

Any dataframe file (`.csv`) containing the columns `Name`, `Row`, `Column` can be used with the `Patches` tool to
extract sub-images (i.e., patches) from the original image. The process for doing so matches that of CoralNet. Patches
are stored in directory called `patches`, which contains subdirectories for each class category. The patch names
follow a convention of: `Image_Name_Row_Column_Label.png`, which can be used for local model training.

### Classifier

CoralNet is great! But what if you want to train your own model locally? The `Classifier` tool allows you to do just
that. This is most useful in situations where you need access to a model locally for other downstream tasks, or, if
you have enough labeled data that it's more beneficial to train / fine-tune your own convolutional neural network (CNN)
as opposed to using CoralNet's existing encoder as a base. To reiterate: if you have enough labeled data, a locally
trained model might actually perform better than a model made on CoralNet, however, will smaller datasets, the locally
trained model could be worse than one made by CoralNet.

The `Classifier` tool exposes some parameters for you to experiment with, will run the training and log to Tensorboard,
and record the output results / metrics to your local machine for posterity. To use the `Classifier` tool, just provide
the `patches` dataframe file; the output will be inside the `classifier` folder.

Note that the `Classifier` tool and `Upload` tool serve similar roles: the former allows you to train a model locally,
whereas the latter allows you to upload data to CoralNet which will train a model for you.

### Inference

Of course, if you can train a model locally you should also have the ability to perform inferences locally! The
`Inference` tool allows you to use your trained model to make predictions on images locally. This can be extremely 
useful as it's significantly faster than using the `API` tool to get a CoralNet model to make predictions, and there
are no limitations on the number of predictions.

To use the `Inference` tool, provide the directory containing the imageset, the model file (`.h5`), the class map
file (`.json`), and a points dataframe file indicating where on each image you want predictions for. Like the `API`,
predictions will be output as a dataframe file (`.csv`) in the `predictions` folder.

### TODO

That's it for now, but in the future we'll be adding additional tools including: 

- `Analysis`: Calculate CPCe statistics from locally trained model's predictions
- `SDM` Species distribution modeling visualizations
- `Seg`: Create a FCN to perform semantic segmentation 
- `Clean`: Use `CleanLab.ai` to identify potentially incorrectly labeled patches
- `GAN`: Synthesize samples using generative AI
- `GPT`: LLMs for QA of annotation dataframes, plot visualizations

Happy to take on any requests. If you run into issues, please leave a ticket in the repository's `Issue` tab.
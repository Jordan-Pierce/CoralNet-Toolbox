# Usage

## Overview
The CoralNet Toolbox is a Python application built using PyQt5 for image annotation.
This guide provides instructions on how to use the application, including key functionalities and hotkeys.

## Annotations
- **PatchAnnotation**: Represents a patch annotation.
- **RectangleAnnotation**: Represents a rectangular annotation.
- **PolygonAnnotation**: Represents a polygonal annotation.

## Computer Vision Tasks
- **Classification**: Assign a label to an image (Patch).
- **Detection**: Detect objects in an image (Rectangle).
- **Segmentation**: Segment objects in an image (Polygon).

## Thresholds for Computer Vision Tasks
- **Patch Size**: Adjust the patch size in the status bar.
- **Uncertaint** Threshold: Adjust the uncertainty threshold in the status bar.
- **IoU Threshold**: Adjust the IoU threshold in the status bar.
- **Area Threshold**: Adjust the min and max area threshold in the status bar.

## Main Window
The main window consists of several components:
- **Menu Bar**: Contains import, export, and other actions.
- **Tool Bar**: Contains tools for selection and annotation.
- **Status Bar**: Displays the image size, cursor position, view extent, annotation transparency, and thresholds.
- **Annotation Window**: Displays the image and annotations.
- **Label Window**: Lists and manages labels.
- **Image Window**: Displays imported images.
- **Confidence Window**: Displays cropped images and confidence charts.

## Menu Bar Actions
- **New Project**: Reload CoralNet-Toolbox (loss of data warning).
- **Open Project**: Open an existing CoralNet-Toolbox project JSON file.
- **Save Project**: Save current CoralNet-Toolbox project to JSON file.

- **Import**:
  - **Import Images**: Load image files.
  - **Import Frames**: Load video frames (Currently not available).
  - **Import Labels (JSON)**: Load label data from a JSON file.
  - **Import CoralNet Labels (CSV)**: Load label data from a CoralNet CSV file.
  - **Import TagLab Labels (JSON)**: Load label data from a TagLab JSON file.
  - **Import Annotations (JSON)**: Load annotation data from a JSON file.
  - **Import CoralNet Annotations**: Load annotation data from a CoralNet CSV file.
  - **Import TagLab Annotations**: Load annotation data from a TagLab JSON file.
  - **Import Viscore Annotations**: Load annotation data from a Viscore CSV file.
  - **Import Dataset**: Import a YOLO dataset for machine learning (Detection, Segmentation).

- **Export**:
  - **Export Labels (JSON)**: Save label data to a JSON file.
  - **Export TagLab Labels (JSON)**: Save label data to a TagLab JSON file.
  - **Export Annotations (JSON)**: Save annotation data to a JSON file.
  - **Export GeoJSON Annotations**: Save annotations to GeoJSON file (only for GeoTIFFs with CRS and Transforms data)
  - **Export Mask Annotations (Raster)**: Save annotations as segmentation masks
  - **Export CoralNet Annotations**: Save annotation data to a CoralNet CSV file.
  - **Export TagLab Annotations**: Save annotation data to a TagLab JSON file.
  - **Export Viscore Annotations**: Save annotation data to a Viscore CSV file.
  - **Export Dataset**: Create a YOLO dataset for machine learning (Classification, Detection, Segmentation).

- **Sample**:
  - **Sample Annotations**: Automatically generate Patch annotations.

- **Tile**:
  - **Tile Dataset**: Tile existing Classification, Detection or Segmention datasets using `yolo-tiling`.

- **CoralNet**: 
  - **Authenticate**: Authenticate with CoralNet.
  - **Download**: Download data from CoralNet.

- **Ultralytics**:
  - **Merge Datasets**: Merge multiple Classification datasets.
  - **Train Model**: Train a machine learning model.
  - **Evaluate Model**: Evaluate a trained model.
  - **Optimize Model**: Convert model format.
  - **Deploy Model**: Make predictions using a trained model (Classification, Detection, Segmentation).
  - **Batch Inference**: Perform batch inferences.

- **SAM**:
  - **Deploy Predictor**: Deploy `EdgeSAM`, `MobileSAM`, `SAM`, etc, to use interactively (points, box).
  - **Deploy Generator**: Deploy `FastSAM` to automatically segment the image.
  - **Batch Inference**: Perform batch inferencing using `FastSAM`.

- **See Anything (YOLOE)**:
  - **Train Model**: Train a YOLOE Segmentation model using an existing YOLO Segmentation dataset.
  - **Deploy Predictor**: Deploy an existing `YOLOE` model to use interactively.
  - **Batch Inference**: Perform batch inferencing using `YOLOE`.

- **AutoDistill**:
  - **Deploy Model**: Deploy a foundational model
    - Models Available: `Grounding DINO`, `OWLViT`, `OmDetTurbo`
  - **Batch Inference**: Perform batch inferences.

## Tool Bar
- **Select Tool**:
  -

- **Patch Tool**:
  - 

- **Rectangle Tool**:
  -

- **Polygon Tool**:
  -

- **SAM Tool**: After a model is loaded
  - **Space Bar**: Set working area; confirm prediction; finalize predictions and exit working area.
  - **Left-Click**: Start a box; press again to end a box.
  - **Ctrl + Left-Click**: Add positive point.
  - **Ctrl + Right-Click**: Add negative point.
  - **Backspace**: Discard unfinalized predictions.

- **See Anything (YOLOE) Tool**: After a model is loaded
  - **Space Bar**: Set working area; run prediction; finalize predictions and exit working area.
  - **Left-Click**: Start a box; press again to end a box.
  - **Backspace**: Discard unfinalized predictions.

## Status Bar
- **Image Size**: Displays the image size.
- **Cursor Position**: Displays the cursor position.
- **View Extent**: Displays the view extent.
- **Annotation Transparency**: Adjust the annotation transparency.
  - **Select All Labels**: Select all labels, adjusting transparency for all labels.
  - **Unselect All Labels**: Unselect all labels, adjusting transparency for only selected labels.
- **Patch Size**: Manipulate Patch Size (only active when using Patch Tool).
- **Parameters**: Adjust parameters including uncertainty, IoU, and area thresholds.

## Annotation Window
- **Zoom**: Use the mouse wheel to zoom in and out.
- **Pan**: Hold Ctrl + Right-click the mouse button to pan the image.

## Label Window
- **Move Label**: Right-click and drag to move labels.
- **Add Label**: Click the "Add Label" button to add a new label.
- **Delete Label**: Click the "Delete Label" button to delete the selected label.
- **Edit Label**: Click the "Edit Label" button to edit the selected label.
- **Lock Label**: Click the "Lock Label" button to lock the selected label.

## Image Window
- **Load Image**: Click on a row to load the image in the annotation window.
- **Delete Image**: Right-click on a row and select "Delete Image" to remove the image.
- **Delete Annotations**: Right-click on a row and select "Delete Annotations" to remove the image's annotations.
- **Search / Filter**:
  - **By Image**: Filter for images by name or sub-string.
  - **By Label**: Filter images by labels they contain.
  - **No Annotations**: Filter images with no annotations.
  - **Has Annotations**: Filter images with annotations.
  - **Has Predictions**: Filter images with predictions.
  - **Selected**: Filter images selected.

## Confidence Window
- **Display Cropped Image**: Shows the cropped image of the selected annotation.
- **Confidence Chart**: Displays a bar chart with confidence scores.
  - **Prediction Selection**: Select a prediction from the list to change the label.

### Hotkeys
- **Ctrl + W/A/S/D**: Navigate through labels.
- **Ctrl + Left/Right**: Cycle through annotations.
- **Ctrl + Up/Down**: Cycle through images.
- **Ctrl + Shift + <**: Select all annotations.
- **Ctrl + Shift + >**: Unselect all annotations.
- **Escape**: Exit the program.

- **Machine Learning, SAM, and AutoDistill**: After a model is loaded
  - **Ctrl + 1**: Make prediction on selected Patch annotation, else all in the image with Review label.
  - **Ctrl + 2**: Make predictions using Object Detection model.
  - **Ctrl + 3**: Make predictions using Instance Segmentation model.
  - **Ctrl + 4**: Make predictions using FastSAM model.
  - **Ctrl + 5**: Make predictions using AutoDistill model.
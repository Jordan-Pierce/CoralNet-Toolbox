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

## Main Window
The main window consists of several components:
- **Menu Bar**: Contains import, export, and other actions.
- **Tool Bar**: Contains tools for selection and annotation.
- **Annotation Window**: Displays the image and annotations.
- **Label Window**: Lists and manages labels.
- **Image Window**: Displays imported images.
- **Confidence Window**: Displays cropped images and confidence charts.

## Menu Bar Actions
- **Import**:
  - **Import Images**: Load image files.
  - **Import Frames**: Load video frames (Currently not available).
  - **Import Orthomosaic**: Load orthomosaic images (Currently not available).
  - **Import Labels (JSON)**: Load label data from a JSON file.
  - **Import Annotations (JSON)**: Load annotation data from a JSON file.
  - **Import Annotations (CoralNet)**: Load annotation data from a CoralNet CSV file.
  - **Import Annotations (Viscore)**: Load annotation data from a Viscore CSV file.
  - **Import Annotations (TagLab)**: Load annotation data from a TagLab JSON file.
  - **Import Dataset**: Import a YOLO dataset for machine learning (Detection, Segmentation).

- **Export**:
  - **Export Labels (JSON)**: Save label data to a JSON file.
  - **Export Annotations (JSON)**: Save annotation data to a JSON file.
  - **Export Annotations (CoralNet)**: Save annotation data to a CoralNet CSV file.
  - **Export Annotations (Viscore)**: Save annotation data to a Viscore CSV file.
  - **Export Annotations (TagLab)**: Save annotation data to a TagLab JSON file.
  - **Export Dataset**: Create a YOLO dataset for machine learning (Classification, Detection, Segmentation).

- **Sample**:
  - **Sample Annotations**: Automatically generate Patch annotations.

- **CoralNet**: (Currently not available)
  - **Authenticate**: Authenticate with CoralNet.
  - **Upload**: Upload data to CoralNet.
  - **Download**: Download data from CoralNet.
  - **Model API**: Access CoralNet model API.

- **Machine Learning**:
  - **Merge Datasets**: Merge multiple Classification datasets.
  - **Train Model**: Train a machine learning model.
  - **Evaluate Model**: Evaluate a trained model.
  - **Optimize Model**: Convert model format.
  - **Deploy Model**: Make predictions using a trained model (Classification, Detection, Segmentation).
  - **Batch Inference**: Perform batch inferences.

- **SAM**:
  - **Deploy Predictor**: Deploy EdgeSAM, MobileSAM, or SAM to use interactively (points, box); Segment Anything
  - **Deploy Generator**: Deploy FastSAM to automatically segment the image; Segment Everything
    - Recommendation: Use the "Use Predictor to create Polygons", as the results are significantly better
  - **Batch Inference**: Perform batch inferences.

- **AutoDistill**:
  - **Deploy Model**: Deploy a foundational model
    - Models Available: GroundingDino,
  - **Batch Inference**: Perform batch inferences.

## Tool Bar
- **Select Tool**: Select multiple annotations; move and change the size of annotations.
- **Patch Tool**: Add new PatchAnnotations.
- **Polygon Tool**: Add new PolygonAnnotations.
- **Rectangle Tool**: Add new RectangleAnnotations.
- **SAM Tool**: Use SAM model for automatic segmentation (points, box).

## Annotation Window
- **Zoom**: Use the mouse wheel to zoom in and out.
- **Pan**: Hold Ctrl + Right-click the mouse button to pan the image.
- **Add Annotation**: Click with the Left mouse button while using one of the annotation tools.
- **Select Annotations**:
  - Ctrl + Left-Click on multiple annotations while using the select tool.
  - Ctrl + Left-Click and drag to select multiple annotations while using the select tool.
    - **Move Annotation**: Drag a selected annotation.
    - **Modify Annotation**: Hold Shift and drag the vertices of the selected annotation (Rectangle, Polygon).
    - **Resize Annotation**: Hold Ctrl and Zoom in / out to increase / decrease a selected annotation's size.
    - **Delete Annotations**: Press Ctrl + Delete to delete the selected annotations.

## Label Window
- **Move Label**: Right-click and drag to move labels.
- **Add Label**: Click the "Add Label" button to add a new label.
- **Edit Label**: Click the "Edit Label" button to edit the selected label.
- **Delete Label**: Click the "Delete Label" button to delete the selected label.

## Image Window
- **Load Image**: Click on a row to load the image in the annotation window.
- **Delete Annotations**: Right-click on a row and select "Delete Annotations" to remove the image's annotations.
- **Search Image**: Search for an image row by typing out a search string

## Confidence Window
- **Display Cropped Image**: Shows the cropped image of the selected annotation.
- **Confidence Chart**: Displays a bar chart with confidence scores.
- **Prediction Selection**: Select a prediction from the list to change the label.

### Hotkeys
- **Ctrl + Delete**: Delete the selected annotations.
- **Ctrl + W/A/S/D**: Navigate through labels.
- **Ctrl + Mouse Wheel**: Adjust annotation size.
- **Ctrl + Left/Right**: Cycle through annotations.
- **Ctrl + Up/Down**: Cycle through images.
- **End**: Unselect annotation.
- **Home**: Untoggle all tools.
- **Escape**: Exit the program.

- **Machine Learning, SAM, and AutoDistill**: After a model is loaded
  - **Ctrl + 1**: Make prediction on selected Patch annotation, else all in the image with Review label.
  - **Ctrl + 2**: Make predictions using Object Detection model.
  - **Ctrl + 3**: Make predictions using Instance Segmentation model.
  - **Ctrl + 4**: Make predictions using FastSAM model.
  - **Ctrl + 5**: Make predictions using AutoDistill model.

- **SAM**: After a model is loaded
  - **Space Bar**: Set working area; finalize prediction.
  - **Left-Click**: Start a box; press again to end a box.
  - **Ctrl + Left-Click**: Add positive point.
  - **Ctrl + Right-Click**: Add negative point.

## Additional Tips
- **Annotation Sampling**: Use the "Sample Annotations" action in the menu bar to automatically generate annotations.
- **Transparency Control**: Adjust the transparency slider in the status bar to change annotation transparency.
- **Uncertainty Threshold**: Modify the uncertainty threshold in the status bar to restrict predicts with low confidence.
- **IoU Threshold**: Adjust the IoU threshold in the status bar to filter predictions based on Intersection over Union.
- **Use SAM**: Use SAM with a deployed Detection or Segmentation model to create Polygons for each detected object.

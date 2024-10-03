# CoralNet Toolbox Instructions

## Overview
The CoralNet Toolbox is a Python application built using PyQt5 for image annotation and analysis. 
This guide provides instructions on how to use the application, including key functionalities and hotkeys.

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
  - **Import Labels (JSON)**: Load label data from a JSON file.
  - **Import Annotations (JSON)**: Load annotation data from a JSON file.
  - **Import Annotations (CoralNet)**: Load annotation data from a CoralNet CSV file.

- **Export**:
  - **Export Labels (JSON)**: Save label data to a JSON file.
  - **Export Annotations (JSON)**: Save annotation data to a JSON file.
  - **Export Annotations (CoralNet)**: Save annotation data to a CoralNet CSV file.

- **Sample**:
  - **Sample Annotations**: Automatically generate annotations.

- **CoralNet**: (Currently not available)
  - **Authenticate**: Authenticate with CoralNet.
  - **Upload**: Upload data to CoralNet.
  - **Download**: Download data from CoralNet.
  - **Model API**: Access CoralNet model API.

- **Machine Learning**:
  - **Create Dataset**: Create a dataset for machine learning.
  - **Merge Datasets**: Merge multiple datasets.
  - **Train Model**: Train a machine learning model.
  - **Evaluate Model**: Evaluate a trained model.
  - **Optimize Model**: Convert model format.
  - **Deploy Model**: Make predictions using a trained model.
  - **Batch Inference**: Perform batch inferences.

- **SAM**:
  - **Deploy Model**: Deploy SAM or ModbileSAM.
  - **Batch Inference**: Perform batch inferences.

## Tool Bar
- **Select Tool**: Select and move annotations.
- **Patch Tool**: Add new PatchAnnotations.
- **Polygon Tool**: Add new PolygonAnnotations.
- **SAM Tool**: Use SAM model for automatic segmentation.

## Annotation Window
- **Zoom**: Use the mouse wheel to zoom in and out.
- **Pan**: Hold Ctrl + Right-click the mouse button to pan the image.
- **Add Annotation**: Click with the left mouse button while using the annotate tool.
- **Select Annotation**: Click on an annotation while using the select tool.

### Hotkeys
- **Ctrl + Delete**: Delete the selected annotation, or image.
- **Ctrl + W/A/S/D**: Navigate through labels.
- **Ctrl + Mouse Wheel**: Adjust annotation size.
- **Ctrl + Left/Right**: Cycle through annotations.
- **Ctrl + Up/Down**: Cycle through images.
- **End**: Unselect annotation.
- **Home**: Untoggle all tools.
- **Escape**: Exit the program.

- **Machine Learning**: After a model is loaded
  - **Ctrl + Z**: Make prediction on selected annotation, else all in the image.

- **SAM**: After a model is loaded
  - **Space Bar**: Set working area; finalize prediction.
  - **Ctrl + Left-Click**: Add positive point.
  - **Ctrl + Right-Click**: Add negative point.

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

## Additional Tips
- **Annotation Sampling**: Use the "Sample Annotations" action in the menu bar to automatically generate annotations.
- **Transparency Control**: Adjust the transparency slider in the status bar to change annotation transparency.
- **Uncertainty Threshold**: Modify the uncertainty threshold in the status bar to restrict predicts with low confidence.

## Annotations
- **PatchAnnotation**: Represents a patch annotation.
- **PolygonAnnotation**: Represents a polygonal annotation.

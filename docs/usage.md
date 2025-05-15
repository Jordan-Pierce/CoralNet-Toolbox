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
  - **Export GeoJSON Annotations**: Save annotations to GeoJSON file (only for GeoTIFFs with CRS and Transforms data).
  - **Export Mask Annotations (Raster)**: Save annotations as segmentation masks.
  - **Export CoralNet Annotations**: Save annotation data to a CoralNet CSV file.
  - **Export TagLab Annotations**: Save annotation data to a TagLab JSON file.
  - **Export Viscore Annotations**: Save annotation data to a Viscore CSV file.
  - **Export Dataset**: Create a YOLO dataset for machine learning (Classification, Detection, Segmentation).

- **Sample**:
  - **Sample Annotations**: Automatically generate Patch annotations.
    - **Sampling Method**: Choose between Random, Stratified Random, or Uniform distribution.
    - **Number of Annotations**: Specify how many annotations to generate.
    - **Annotation Size**: Set the size of the generated patch annotations.
    - **Label Selection**: Choose which label to assign to generated annotations.
    - **Exclude Regions**: Option to prevent sampling in areas with existing annotations.
    - **Margins**: Define image boundary constraints for sampling:
      - Set margins in pixels or percentage
      - Configure different values for top, right, bottom, and left edges
      - Annotations will only be placed within these margins
    - **Apply Options**: Apply sampling to current image, filtered images, previous/next images, or all images.

- **Tile**:
  - **Tile Dataset**: Tile existing Classification, Detection or Segmention datasets using `yolo-tiling`.
  - **Tile Inference**: Pre-compute multiple work areas for the current image.

- **CoralNet**: 
  - **Authenticate**: Authenticate with CoralNet.
    - Enter your CoralNet username and password to access your sources.
    - Authentication is required before downloading any CoralNet data.
  - **Download**: Download data from CoralNet.
    - **Source ID**: Enter the Source ID (or multiple IDs separated by commas).
    - **Output Directory**: Select where to save downloaded files.
    - **Download Options**: Choose what to download:
      - Metadata: Source information and settings
      - Labelset: All available labels from the source
      - Annotations: Point annotations with their labels
      - Images: Original images from the source
    - **Parameters**: Configure download settings:
      - Image Fetch Rate: Time between image downloads (seconds)
      - Image Fetch Break Time: Pause duration between batches (seconds)
    - **Debug Mode**: Toggle headless browser mode for troubleshooting.

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

## Tool Bar Tools
- **Select Tool**: After selecting the tool
  - **Left-Click**: Select an annotation; drag to move it.
  - **Ctrl + Left-Click**: Add/remove annotation to current selection.
  - **Ctrl + Drag**: Create rectangle selection to select multiple annotations.
  - **Delete / Backspace**: Remove selected annotation(s).
  - **Ctrl + Shift**: Show resize handles for the selected annotation.
  - **Ctrl + Mouse Wheel**: Change size of the selected annotation.
  - **Ctrl + Space**: Confirm prediction for selected annotation with top machine confidence.
  - **Ctrl + C**: Combine multiple selected annotations (if same type and label) or enter cutting mode for single annotation.
    - **Combining Rules**: 
      - All selected annotations must have the same label
      - All selected annotations must be verified (not machine predictions)
      - Patch annotations can be combined with other patches or polygons
      - Rectangle annotations can only be combined with other rectangles
      - Polygon annotations can be combined with other polygons
    - **Cutting Mode**: Left-click to start drawing a cut line, click again to complete the cut.
  - **Backspace/Ctrl + C**: Cancel cutting mode.

- **Patch Tool**: After selecting the tool
  - **Left-Click**: Add a patch annotation at the clicked position.
  - **Ctrl + Mouse Wheel**: Adjust the patch size up or down.
  - **Mouse Movement**: Shows a semi-transparent preview of the patch at the cursor position.

- **Rectangle Tool**: After selecting the tool
  - **Left-Click**: Start drawing a rectangle; click again to finish.
  - **Mouse Movement**: Shows a preview of the rectangle while drawing.
  - **Backspace**: Cancel the current rectangle annotation.

- **Polygon Tool**: After selecting the tool
  - **Left-Click (first)**: Start drawing a polygon.
  - **Left-Click (subsequent)**: Add points to the polygon; click near the first point to close.
  - **Ctrl + Left-Click**: Enable straight line mode; click to add straight line segments.
  - **Mouse Movement**: Shows a preview of the polygon as you draw.
  - **Backspace**: Cancel the current polygon annotation.

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

- **Work Area Tool**: For creating restricted areas for model prediction
  - **Left-Click**: Start drawing a work area; click again to finish drawing.
  - **Backspace**: Cancel drawing the current work area.
  - **Ctrl + Space**: Create a work area from the current view.
  - **Ctrl + Alt**: Create temporary work area from current view (disappears when keys released / pressed again).
  - **Ctrl + Shift**: Show removal buttons on existing work areas (click the "X" to remove).
  - **Ctrl + Shift + Backspace**: Remove all work areas in the current image.
  - **Practical Use**:
    - Define specific regions where models should make predictions.
    - Useful for processing only relevant parts of large images.
    - Work areas persist between tool changes and sessions.

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
- **Filter Labels**: Use the filter text box to search for specific labels.
- **Label Count**: Displays the total number of labels in the project.
- **Annotation Count**: Shows information about the current annotations:
  - When no annotation is selected: Shows the total count of annotations.
  - When a single annotation is selected: Shows the selected annotation's index.
  - When multiple annotations are selected: Shows how many annotations are selected.
  - Can be edited (when in select mode) to navigate to a specific annotation by index.

## Image Window
- **Select Image**: Double-click on a row to select and load the image in the annotation window.
- **Highlight Image**: Single-click on a row to highlight one or more rows in the image window.
  - **Ctrl + Left-click**: Select multiple, non-adjacent rows.
  - **Shift + Left-click**: Select multiple, adjacent rows.
- **Open Context Menu**:
  - **Right-click on a single highlighted row**: Delete images / annotations for the highlighted row.
  - **Shift + Right-click on multiple highlighted rows**: Delete images / annotations for highlighted rows.
- **Search / Filter**:
  - **By Image**: Filter for images by name or sub-string.
  - **By Label**: Filter images by labels they contain.
  - **No Annotations**: Filter images with no annotations.
  - **Has Annotations**: Filter images with annotations.
  - **Has Predictions**: Filter images with predictions.
  - **Highlighted**: Filter highlighted images.
- **Navigation**:
  - **Home Button**: Click to center the table on the currently selected image.
  - **Highlight All**: Highlight all images in the current filtered view.
  - **Unhighlight All**: Unhighlight all images in the current filtered view.
- **Image Preview**:
  - **Tool Tip**: Hover over a row to show image metadata.
  - **Thumbnail**: Hold Ctrl while hovering over a row to show a thumbnail.

## Confidence Window
- **Display Cropped Image**: Shows the cropped image of the selected annotation.
  - The dimensions shown include both original and scaled sizes when applicable.
  - The border of the image is highlighted with the color of the top confident label.
- **Confidence Chart**: Displays a bar chart with confidence scores.
  - **Top 5 Predictions**: Shows up to 5 predictions with their confidence scores.
  - **Prediction Selection**: Click on any confidence bar to change the annotation's label.
  - **Numerical Keys**: Press keys 1-5 to quickly select from the top 5 predictions.
- **Confidence Mode Toggle**: 
  - Click the icon button next to the dimensions to toggle between user and machine confidence views.
  - User icon shows user-assigned confidence scores.
  - Machine icon shows model-predicted confidence scores.
  - The toggle is only enabled when both user and machine confidences are available.
- **Visual Indicators**:
  - Each confidence bar shows the label color and confidence percentage.
  - Numbered indicators (1-5) show the rank of each prediction.
  - Hover over confidence bars to see a pointing hand cursor when selection is possible.

### Secret Hotkeys
- **Alt + Up/Down**: Cycle through images.
- **Ctrl + W/A/S/D**: Cycle through labels.
- **Ctrl + Left/Right**: Cycle through annotations.

- **Ctrl + Shift + <**: Select all annotations.
- **Ctrl + Shift + >**: Unselect all annotations.

- **Escape**: Exit the program.

- **Machine Learning, SAM, and AutoDistill**: After a model is loaded
  - **Ctrl + 1**: Make prediction on selected Patch annotation, else all in the image with Review label.
  - **Ctrl + 2**: Make predictions using Object Detection model.
  - **Ctrl + 3**: Make predictions using Instance Segmentation model.
  - **Ctrl + 4**: Make predictions using FastSAM model.
  - **Ctrl + 5**: Make predictions using AutoDistill model.
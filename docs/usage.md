# Usage

## Overview
The CoralNet Toolbox is a Python application built using PyQt5 for image annotation
This guide provides instructions on how to use the application, including key functionalities and hotkeys

## Annotations
- **PatchAnnotation**: Represents a patch annotation
- **RectangleAnnotation**: Represents a rectangular annotation
- **PolygonAnnotation**: Represents a polygonal annotation
  - **MultiPolygonAnnotation**: Represents multiple, non-overlapping polygonal annotations
- **MaskAnnotation**: Represents a segmentation mask (one mask per image)

## Computer Vision Tasks
- **Classification**: Assign a label to an image (Patch)
- **Detection**: Detect objects in an image (Rectangle)
- **Instance Segmentation**: Segment objects in an image (Polygon)
- **Semantic Segmentation**: Segment the entire image (Mask)

## Thresholds for Computer Vision Tasks
- **Patch Size**: Adjust the patch size in the status bar
- **Uncertainty** Threshold: Adjust the uncertainty threshold in the status bar
- **IoU Threshold**: Adjust the IoU threshold in the status bar
- **Area Threshold**: Adjust the min and max area threshold in the status bar

## Main Window
The main window consists of several components:
- **Menu Bar**: Contains import, export, and other actions
- **Tool Bar**: Contains tools for selection and annotation
- **Status Bar**: Displays the image size, cursor position, view extent, annotation transparency, and thresholds
- **Annotation Window**: Displays the image, Z-channel, and annotations
- **Label Window**: Lists and manages labels
- **Image Window**: Displays imported images
- **Confidence Window**: Displays cropped images and confidence charts

## Menu Bar Actions
- **New Project**: Reload CoralNet-Toolbox (loss of data warning)
- **Open Project**: Open an existing CoralNet-Toolbox project JSON file
- **Save Project**: Save current CoralNet-Toolbox project to JSON file

- **Import**:
  - **Import Images**: Load image files
  - **Import Frames**: Load video frames (Currently not available)
  - **Import Labels (JSON)**: Load label data from a JSON file
  - **Import CoralNet Labels (CSV)**: Load label data from a CoralNet CSV file
  - **Import TagLab Labels (JSON)**: Load label data from a TagLab JSON file
  - **Import Annotations (JSON)**: Load annotation data from a JSON file
  - **Import CoralNet Annotations (CSV)**: Load annotation data from a CoralNet CSV file
  - **Import TagLab Annotations (JSON)**: Load annotation data from a TagLab JSON file
  - **Import Squidle+ Annotations (JSON)**: Load annotation data from a Squidle+ JSON file
  - **Import Viscore Annotations(CSV)**: Load annotation data from a Viscore CSV file
  - **Import Dataset**: Import a YOLO dataset for machine learning (Detection, Instance Segmentation)

- **Export**:
  - **Export Labels (JSON)**: Save label data to a JSON file
  - **Export TagLab Labels (JSON)**: Save label data to a TagLab JSON file
  - **Export Annotations (JSON)**: Save annotation data to a JSON file
  - **Export GeoJSON Annotations**: Save annotations to GeoJSON file (only for GeoTIFFs with CRS and Transforms data)
  - **Export Mask Annotations (Raster)**: Save annotations as segmentation masks
  - **Export CoralNet Annotations**: Save annotation data to a CoralNet CSV file
  - **Export TagLab Annotations**: Save annotation data to a TagLab JSON file
  - **Export Viscore Annotations**: Save annotation data to a Viscore CSV file
  - **Export Dataset**: Create a YOLO dataset for machine learning (Classification, Detection, Instance, Semantic)

- **Explorer**:
  - **Annotation Settings**: Select images, annotation types, and labels to include / filter press apply
  - **Model Settings**: Use Color Features, a pre-trained, or existing classification model
    - **Feature Mode**: Embeddings (before classification), or Predictions (after classification)
  - **Embedding Settings**: Map high-dimensional features to 2D space using PCA, TSNE, or UMAP
  - **Annotation Viewer**: View, select, modify labels of annotations
    - **Controls**:
      - <kbd>Left-Click</kbd>: Select an annotation
      - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Select multiple annotations
      - <kbd>Shift</kbd> + <kbd>Left-Click</kbd>: Select in-between annotations
      - <kbd>Double Left Click</kbd>: Unselect all annotations, exit from Isolation View
      - <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Update Annotation Window view, zoomed and centered on selected annotation
    - **Toolbar**:
      - **Isolate Selection**: Subset view
      - **Sort By**: Sort annotations by image name, label, confidence
      - **Find Similar**: Selects and isolates N nearest annotations to currently selected
      - **Size**: Slider-bar to control annotation size in Annotation Viewer
  - **Embedding Viewer**: Selected, modify labels of annotations
    - **Controls**:
      - <kbd>Left-Click</kbd>: Select an annotation
      - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Select multiple annotations
      - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd> + <kbd>Drag</kbd>: Select multiple annotations within a draw rectangle
      - <kbd>Double Left Click</kbd>: Unselect all annotations, exit from Isolation View
      - <kbd>Right-Click</kbd> + <kbd>Drag</kbd>: Pan around Embedding Viewer
      - <kbd>Mouse Wheel</kbd>: Zoom in and out of Embedding Viewer
    - **Toolbar**:
      - **Isolate Selection**: Subset view
      - **Find Potential Mislabels**: Select potentially incorrectly labeled annotations, based on location
      - **Review Uncertain**: Select annotations with lower Top-1 confidence scores (requires Predictions)
      - **Find Duplicates**: Select annotations that are likely duplicates of another (only selects the duplicates) 
      - **Home**: Resets the Embedding Viewer zoom level
  **Tip**: 
    - Use dual monitors to assess selected annotations in Annotation Viewer, in the Annotation Window

- **Sample**:
  - **Sample Annotations**: Automatically generate Patch annotations
    - **Sampling Method**: Choose between Random, Stratified Random, or Uniform distribution
    - **Number of Annotations**: Specify how many annotations to generate
    - **Annotation Size**: Set the size of the generated patch annotations
    - **Label As**: Choose which label to assign to generated annotations
    - **Apply Options**: Apply sampling to current image, filtered images, previous/next images, or all images
    - **Exclude Regions**: Option to prevent sampling in areas with existing annotations
    - **Margins**: Define image boundary constraints for sampling:
      - Set margins in pixels or percentage
      - Configure different values for top, right, bottom, and left edges
      - Annotations will only be placed within these margins

- **Tile**:
  - **Tile Manager**: Pre-compute multiple tiles / work areas for selected images
  - **Tile Batch Inferece**: Apply inference on all imagues with tiles / work areas
  - **Tile Dataset**: Tile existing Classification, Detection or Segmention datasets using `yolo-tiling`

- **CoralNet**: 
  - **Authenticate**: Authenticate with CoralNet
    - Enter your CoralNet username and password to access your sources
    - Authentication is required before downloading any CoralNet data
  - **Download**: Download data from CoralNet
    - **Source ID**: Enter the Source ID (or multiple IDs separated by commas)
    - **Output Directory**: Select where to save downloaded files
    - **Download Options**: Choose what to download:
      - Metadata: Source information and settings
      - Labelset: All available labels from the source
      - Annotations: Point annotations with their labels
      - Images: Original images from the source
    - **Parameters**: Configure download settings:
      - Image Fetch Rate: Time between image downloads (seconds)
      - Image Fetch Break Time: Pause duration between batches (seconds)
    - **Debug Mode**: Toggle headless browser mode for troubleshooting

- **Machine Learning**:
  - **Merge Datasets**: Merge multiple Classification datasets
  - **Tune Model**: Identify ideal hyperparameter values before fully training a model
  - **Train Model**: Train a machine learning model
  - **Evaluate Model**: Evaluate a trained model
  - **Optimize Model**: Convert model format
  - **Deploy Model**: Make predictions using a trained model
  - **Batch Inference**: Perform batch inferences
  - **Video Inference**: Perform inferencing on videos in real-time, view analytics

- **SAM**:
  - **Deploy Predictor**: Deploy `EdgeSAM`, `MobileSAM`, `SAM`, etc, to use interactively (points, box)
  - **Deploy Generator**: Deploy `FastSAM` to automatically segment the image
  - **Batch Inference**: Perform batch inferencing using `FastSAM`

- **See Anything (YOLOE)**:
  - **Deploy Predictor**: Deploy a `YOLOE` model to use interactively within the same image
  - **Deploy Generator**: Deploy a `YOLOE` model to use like a detector / segmentor, referencing other images' annotations
    - Select the `YOLOE` model, parameters, and load it
    - Choose a reference label, then select the image(s) containing reference annotations (must be rectangles or polygons)
    - Generate visual prompt encodings (VPEs) from reference images / annotations (save and show if needed)
    - Use the loaded model w/ VPEs on new images, or work areas, and with batch inferencing
  - **Batch Inference**: Perform batch inferencing using loaded `YOLOE` generator

- **Transformers**:
  - **Deploy Model**: Deploy a foundational model
    - Models Available: `Grounding DINO`, `OWLViT`
  - **Batch Inference**: Perform batch inferences

## Tool Bar Tools
- **Select Tool**: After selecting the tool
  - <kbd>Left-Click</kbd>: Select an annotation drag to move it
  - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add/remove annotation to current selection
  - <kbd>Ctrl</kbd> + <kbd>Delete</kbd> / <kbd>Backspace</kbd>: Remove selected annotation(s)
  - <kbd>Ctrl</kbd> + <kbd>Drag</kbd>: Create rectangle selection to select multiple annotations
  - <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd>: Change size of the selected annotation
  - <kbd>Ctrl</kbd> + <kbd>Shift</kbd>: Show resize handles for the selected annotation
  - <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Mouse Wheel</kbd>: Change the number of vertices for a polygon annotation
  - <kbd>Ctrl</kbd> + <kbd>Space</kbd>: Confirm prediction for selected annotation with top machine confidence
  - <kbd>Ctrl</kbd> + <kbd>X</kbd>: Cut a polygon annotation, explode a multi-polygon annotation, or subtract polygon annotations
     - **Cutting Rules**:
         - Only a single annotation can be selected
         - Press <kbd>Ctrl</kbd>-<kbd>X</kbd> to enter cutting mode, <kbd>Left-Click</kbd> to start, draw line, <kbd>Left-Click</kbd> to end
          - Hold <kbd>Ctrl</kbd> to create straight line segments
         - Press <kbd>Backspace</kbd> or <kbd>Delete</kbd> to cancel the current cutting line without making changes
     - **Exploding Rules**:
         - Select a MultiPolygonAnnotation, press <kbd>Ctrl</kbd> + <kbd>X</kbd> to explode it into multiple PolygonAnnotations
     - **Subtraction Rules**:
         - Multiple overlapping annotations must be selected one-by-one
         - The first annotations will be used as the cutters, the last polygon will be used as the base   
  - <kbd>Ctrl</kbd> + <kbd>C</kbd>: Combine multiple selected annotations (if same type and label)
    - **Combining Rules**: 
      - All selected annotations must have the same label
      - All selected annotations must be verified (not machine predictions)
      - RectangleAnnotations can only be combined with other rectangles
      - PatchAnnotations can be combined with other patches or polygons to create polygons
      - PolygonAnnotations can be combined with other overlapping polygons to create a polygon
      - MultiPolygonAnnotations can be made with multiple non-overlapping polygons

- **Scale Tool**: Provide scale to the image(s), and measure distances on the current image.
  - <kbd>Left-Click</kbd> to set the starting point.
  - Drag to draw a line, then <kbd>Left-Click</kbd> again to set the endpoint.
  - Press <kbd>Backspace</kbd> to cancel drawing the scale line.
  - The scale will be calculated based on the known provided length and pixel length.
  - Area and Perimeter for an annotation can be viewed when hovering over the Confidence Window.
  - Preferred units can be set in the Status Bar.

- **Patch Tool**: After selecting the tool
  - <kbd>Left-Click</kbd>: Add a patch annotation at the clicked position
  - <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd>: Adjust the patch size up or down
  - <kbd>Mouse Movement</kbd>: Shows a semi-transparent preview of the patch at the cursor position

- **Rectangle Tool**: After selecting the tool
  - <kbd>Left-Click</kbd>: Start drawing a rectangle click again to finish
  - <kbd>Mouse Movement</kbd>: Shows a preview of the rectangle while drawing
  - <kbd>Backspace</kbd>: Cancel the current rectangle annotation

- **Polygon Tool**: After selecting the tool
  - <kbd>Left-Click</kbd> (first): Start drawing a polygon
  - <kbd>Left-Click</kbd> (subsequent): Add points to the polygon click near the first point to close
  - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Enable straight line mode click to add straight line segments
  - <kbd>Mouse Movement</kbd>: Shows a preview of the polygon as you draw
  - <kbd>Backspace</kbd>: Cancel the current polygon annotation

- **Brush Tool**: After selecting the tool
  - <kbd>Left-click</kbd> and drag to paint brush strokes on the canvas.
  - Hold <kbd>Ctrl</kbd> and use the <kbd>Mouse Wheel</kbd> to adjust brush size.
  - Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> to switch between a circle and square brush shape.
  - A semi-transparent preview shows the brush stroke while drawing.

- **Erase Tool**: After selecting the tool
  - <kbd>Left-click</kbd> and drag to erase pixels.
  - Hold <kbd>Ctrl</kbd> and use the <kbd>Mouse Wheel</kbd> to adjust eraser size.
  - Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> to switch between a circle and square eraser shape.
  - Press <kbd>Ctrl</kbd> + (<kbd>Backspace</kbd> or <kbd>Delete</kbd>) to clear the mask annotation on the current image.
  - A semi-transparent preview shows the eraser while drawing.

- **Dropper Tool**: After selecting the tool
  - <kbd>Left-click</kbd> on a mask annotation region to select the associated label.

- **Fill Tool**: After selecting the tool
  - <kbd>Left-click</kbd> to fill the region under the cursor with the selected label.

- **SAM Tool**: After a model is loaded
  - <kbd>Left-Click</kbd>: Start drawing a work area click again to finish drawing
  - <kbd>Backspace</kbd>: Cancel drawing the current work area
  - <kbd>Space</kbd>: Create a work area from the current view
    - <kbd>Space</kbd>: Set working area confirm prediction finalize predictions and exit working area
    - <kbd>Left-Click</kbd>: Start a box press again to end a box
    - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add positive point
    - <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Add negative point
    - <kbd>Backspace</kbd>: Discard unfinalized predictions

- **See Anything (YOLOE) Tool**: After a model is loaded
  - <kbd>Left-Click</kbd>: Start drawing a work area click again to finish drawing
  - <kbd>Backspace</kbd>: Cancel drawing the current work area
  - <kbd>Space</kbd>: Create a work area from the current view
    - <kbd>Space</kbd>: Set working area run prediction finalize predictions and exit working area
    - <kbd>Left-Click</kbd>: Start a box press again to end a box
    - <kbd>Backspace</kbd>: Discard unfinalized predictions

- **Work Area Tool**: For creating restricted areas for model prediction
  - <kbd>Left-Click</kbd>: Start drawing a work area click again to finish drawing
  - <kbd>Backspace</kbd>: Cancel drawing the current work area
  - <kbd>Space</kbd>: Create a work area from the current view
  - <kbd>Ctrl</kbd> + <kbd>Alt</kbd>: Create temporary work area from current view (disappears when keys released / pressed again)
  - <kbd>Ctrl</kbd> + <kbd>Shift</kbd>: Show removal buttons on existing work areas (click the "X" to remove)
  - <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Backspace</kbd>: Remove all work areas in the current image
  - **Practical Use**:
    - Define specific regions where models should make predictions
    - Useful for processing only relevant parts of large images
    - Work areas persist between tool changes and sessions

## Status Bar
- **Image Size**: Displays the image size
- **Cursor Position**: Displays the cursor position
- **View Extent**: Displays the view extent
- **Annotation Visibility**: Show / Hide all existing annotations
- **Annotation Transparency**: Adjust the annotation transparency
- **Scale**: Displays the scale dimensions
  - Provides a dropdown to select preferred units (mm, cm, m, km, etc.,)
  - Enabled when a scale is set using the Scale Tool or imported from an image
- **Z**: Displays the Z-dimension
  - Provides a dropdown to select preferred units (mm, cm, m, km etc.,)
  - Enabled when a Z-channel for the image is imported
  - Select a color map in the dropdown to overlay Z-channel 
  - Click dynamic range button to enable dynamic recoloring of Z-channel 
- **Patch Size**: Manipulate Patch Size (only active when using Patch Tool)
- **Parameters**: Adjust parameters including uncertainty, IoU, and area thresholds

## Annotation Window
- **Zoom**: Use the <kbd>Mouse Wheel</kbd> to zoom in and out
- **Pan**: <kbd>Right-Click</kbd> and hold the <kbd>Mouse Button</kbd> to pan the image

## Label Window
- **Move Label**: <kbd>Right-Click</kbd> and drag to move labels
- **Add Label**: Click the "Add Label" button to add a new label
- **Delete Label**: Click the "Delete Label" button to delete the selected label
- **Edit Label**: Click the "Edit Label" button to edit the selected label
- **Lock Label**: Click the "Lock Label" button to lock the selected label
- **Enable / Disable Labels**: Enable / disable labels by their checkbox to control visibility
  - **Toggle All Button** (asterisk icon): Click to toggle all label checkboxes at once
    - If all labels are visible, clicking will hide all labels
    - If any labels are hidden, clicking will show all labels
    - Operates as a batch operation for improved performance with many labels and annotations
  - Individual label checkboxes control whether annotations of that label are shown or hidden
  - Hidden labels retain their data and can be shown again at any time
  - Transparency changes apply to all labels, visible or hidden
- **Filter Labels**: Use the filter text box to search for specific labels
- **Label Count**: Displays the total number of labels in the project
- **Annotation Count**: Shows information about the current annotations:
  - When no annotation is selected: Shows the total count of annotations
  - When a single annotation is selected: Shows the selected annotation's index
  - When multiple annotations are selected: Shows how many annotations are selected
  - Can be edited (when in select mode) to navigate to a specific annotation by index

## Image Window
- **Select Image**: <kbd>Double-Click</kbd> on a row to select and load the image in the annotation window
- **Highlight Image**: <kbd>Single-Click</kbd> on a row to highlight one or more rows in the image window
  - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Select multiple, non-adjacent rows
  - <kbd>Shift</kbd> + <kbd>Left-Click</kbd>: Select multiple, adjacent rows
- **Open Context Menu**:
  - <kbd>Right-Click</kbd> on a single / multiple highlighted row: 
    - Check / uncheck highlighted rows
    - Import Z-channel for highlighted rows (opens Z-channel import dialog)
    - Remove Z-channel for highlighted rows
    - Delete annotations for highlighted rows
    - Delete images and annotations for highlighted rows
- **Search / Filter**:
  - **By Image**: Filter for images by name or sub-string
  - **By Label**: Filter images by labels they contain
  - **No Annotations**: Filter images with no annotations
  - **Has Annotations**: Filter images with annotations
  - **Has Predictions**: Filter images with predictions
  - **Highlighted**: Filter highlighted images
- **Navigation**:
  - **Home Button**: Click to center the table on the currently selected image
  - **Highlight All**: Highlight all images in the current filtered view
  - **Unhighlight All**: Unhighlight all images in the current filtered view
- **Image Preview**:
  - **Tool Tip**: Hover over a row to show image metadata
  - **Thumbnail**: Hold <kbd>Ctrl</kbd> while hovering over a row to show a thumbnail

## Confidence Window
- **Display Cropped Image**: Shows the cropped image of the selected annotation
  - The dimensions shown include both original and scaled sizes when applicable
  - The border of the image is highlighted with the color of the top confident label
- **Confidence Chart**: Displays a bar chart with confidence scores
  - **Top 5 Predictions**: Shows up to 5 predictions with their confidence scores
  - **Prediction Selection**: Click on any confidence bar to change the annotation's label, verifying it
  - **Numerical Keys**: Press keys <kbd>1</kbd>-<kbd>5</kbd> to quickly select from the top 5 predictions
  - **Prev / Next buttons**: Cycle through annotations in order of their creation
- **Confidence Mode Toggle**: 
    - Click the icon button next to the dimensions to toggle between user and machine confidence views
    - User icon shows user-assigned confidence scores
    - Machine icon shows model-predicted confidence scores
    - The toggle is only enabled when both user and machine confidences are available
- **Visual Indicators**:
  - Each confidence bar shows the label color and confidence percentage
  - Numbered indicators (1-5) show the rank of each prediction
  - Hover over confidence bars to see a pointing hand cursor when selection is possible
- **Tool Tip**: Hover over the window while an annotation is selected to see its metadata

### [Hotkeys](https://jordan-pierce.github.io/CoralNet-Toolbox/hot-keys)

- <kbd>Escape</kbd>: Exit the program
- <kbd>Ctrl</kbd> + <kbd>S</kbd>: Save the project
- <kbd>Ctrl + Z</kbd>: Undo the last addition or deletion of an annotation
- <kbd>Ctrl + Shift + Z</kbd>: Redo the previously undone addition or deletion of an annotation
- <kbd>Alt</kbd> + <kbd>Up</kbd>/<kbd>Down</kbd>: Cycle through images
- <kbd>Ctrl</kbd> + <kbd>Up</kbd>/<kbd>Down</kbd>: Cycle through labels
- <kbd>Ctrl</kbd> + <kbd>Left</kbd>/<kbd>Right</kbd>: Cycle through annotations
- <kbd>Ctrl</kbd> + <kbd>A</kbd>: Select all annotations, unselect all annotations (when pressed twice)
- <kbd>Ctrl</kbd> + <kbd>Alt</kbd>: Switch between tools within the existing tool group, example:
  - When a PatchAnnotation is selected, this switches back to the PatchTool
  - When the PatchTool is active, this switches back to the SelectTool
  - When the BrushTool is active, this switches to the EraseTool, and vice versa

- **Machine Learning, SAM, See Anything (YOLOE), and Transformers**: After a model is loaded
  - <kbd>Ctrl</kbd> + <kbd>1</kbd>: Make prediction on selected Patch annotation, else all in the image with Review label using Classification model
  - <kbd>Ctrl</kbd> + <kbd>2</kbd>: Make predictions using Object Detection model
  - <kbd>Ctrl</kbd> + <kbd>3</kbd>: Make predictions using Instance Segmentation model
  - <kbd>Ctrl</kbd> + <kbd>4</kbd>: Make predictions using Semantic Segmentation model
  - <kbd>Ctrl</kbd> + <kbd>5</kbd>: Make predictions using FastSAM model
  - <kbd>Ctrl</kbd> + <kbd>6</kbd>: Make predictions using YOLOE model
  - <kbd>Ctrl</kbd> + <kbd>7</kbd>: Make predictions using Transformers model

- **Tooltips**: Hover over tool buttons, image / annotation rows, and the Confidence Window for additional information



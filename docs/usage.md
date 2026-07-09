# Usage

## Overview
The CoralNet Toolbox is a Python application built using PyQt5 for image annotation.  
This guide provides instructions on how to use the application, including key functionalities and hotkeys

## Annotations
- **PatchAnnotation**: Represents a patch annotation
- **RectangleAnnotation**: Represents a rectangular annotation
- **PolygonAnnotation**: Represents a polygonal annotation
  - **MultiPolygonAnnotation**: Represents multiple, non-overlapping polygonal annotations
- **MaskAnnotation**: Represents a segmentation mask (one mask per image)

## Rasters

### Images
- Standard image formats (PNG, JPG, etc.) can be imported directly
- Each image becomes a canvas for annotations and model predictions

### Orthomosaics
- GeoTIFF files with geospatial metadata (CRS and transforms)
- Enable spatial analysis and coordinate-based measurements
- Support for GeoJSON export of annotation coordinates
- Useful for drone surveys and mapping workflows

### Videos
- Video files can be imported and played frame-by-frame in the Annotation Window
- Each frame is treated as an individual image for annotation purposes
- Frames can be extracted and imported as individual images

### Frames from Video
- Extract specific frames from a video file for annotation
- Useful for selective processing without importing entire video
- Extracted frames are added to the project as regular images

### Z-Channels (Depth Maps & DEMs)
Z-Channels overlay depth or elevation data on top of images for spatial analysis:

**Supported Data Types**
- **Depth Maps**: 2D array representing depth/distance values (typically from sensors or stereo reconstruction)
- **Digital Elevation Models (DEMs)**: Georeferenced elevation data for terrain analysis
- Supports GeoTIFF, PNG, and other raster formats that can be loaded with rasterio

**Importing Z-Channel Data**
- <kbd>Right-Click</kbd> on highlighted images in Image Window
- **Import Z-Channel**: Select depth/elevation file(s) to attach to image(s)
  - Files are automatically resized to match image dimensions if needed
  - Unit of measurement can be specified (meters, feet, centimeters, etc.)
- **Batch Import**: Apply same Z-Channel to multiple selected images

**Z-Channel Visualization**
- Overlay appears in the Annotation Window alongside image and annotations
- **Colormap Selection**: Dropdown in Status Bar to choose visualization style
  - Available colormaps: viridis, plasma, inferno, magma, cividis, twilight, etc.
  - Different colormaps highlight different depth/elevation patterns
- **Dynamic Range**: Enable/disable dynamic recoloring via Status Bar button
  - When enabled: colormap automatically rescales to current viewport's min/max values
  - When disabled: colormap uses image's global min/max (better for comparing multiple images)
- **Opacity Control**: Adjust Z-Channel overlay transparency via Status Bar slider
  - 0% (transparent) shows only the image
  - 100% (opaque) shows full Z-Channel visualization
- **Value Display**: Hover over image to see Z value at cursor position in Status Bar (e.g., "Z: 2.45 m")

**Measurement & Analysis**
- **Scale Tool**: Distances and areas automatically account for Z-Channel (depth-corrected measurements)
- **Hover Tooltips**: View exact Z value when mouse hovers over image
- **Nodata Handling**: Automatically detects and masks NULL/missing values in Z data

**Removing Z-Channel Data**
- <kbd>Right-Click</kbd> on highlighted images in Image Window
- **Remove Z-Channel**: Clear depth/elevation data from selected image(s)
  - Annotation and other data remain intact

## Computer Vision Tasks
- **Classification**: Assign a label to an image (Patch)
- **Detection**: Detect objects in an image (Rectangle)
- **Instance Segmentation**: Segment objects in an image (Polygon)
- **Semantic Segmentation**: Segment the entire image (Mask)

## Thresholds for Computer Vision Tasks
- **Max Detections**: Maximum number of detections kept after non-max suppression (1-10,000)
  - Lower values reduce clutter and processing time; higher values allow more candidates
- **Uncertainty Threshold**: Minimum confidence required to accept a prediction as valid (0.00-1.00)
  - Predictions below this value are marked as Review for manual inspection
- **IoU Threshold**: Intersection-over-Union threshold for non-max suppression (0.00-1.00)
  - Higher values keep more overlapping detections; lower values remove duplicates more aggressively
- **Area Threshold**: Normalized annotation area filter (min and max as 0.00-1.00 fractions)
  - Min: Objects smaller than this fraction of image area are removed
  - Max: Objects larger than this fraction of image area are removed
- **Boundary Detections**: Keep or Ignore detections that touch a work-area edge
  - Keep: Retains cut-off objects; Ignore: Removes them to reduce seam duplicates across tiles

## Main Window
The main window consists of several dockable components:
- **Menu Bar**: Contains import, export, and other actions
- **Tool Bar**: Contains tools for selection and annotation
- **Status Bar**: Displays the image size, cursor position, view extent, annotation transparency, and thresholds
- **Annotation Window**: Displays the image, Z-channel, and annotations with interactive annotation tools
- **Label Window**: Lists and manages labels with operations for editing, merging, and organizing
- **Image Window**: Displays imported images with filtering and batch operations
- **Confidence Window**: Displays cropped images and confidence charts for annotation predictions
- **Performance Window**: Real-time hardware monitor showing CPU, Memory, and GPU usage with historical sparkline graphs
- **Timer Window**: Session timer with start, stop, and reset controls; tracks total work duration across sessions

### Advanced Docking System

All dock windows (Annotation Window, Label Window, Image Window, Confidence Window, Explorer, Performance, Timer) use an advanced docking system that allows complete layout customization:

**Moving & Rearranging Docks**
- **Grab & Drag**: Click and drag the dock title bar to move the window
  - Drag to any dock area (left, right, top, bottom) to dock it
  - Drag to create a floating window (detached from main window)
  - Drag between dock areas to rearrange layout
- **Tab Navigation**: Multiple docks in the same area appear as tabs
  - Click tabs to switch between docked windows
  - Drag tabs to reorder or move to different areas
- **Resize Docks**: Drag the borders between docks to resize
  - Works horizontally and vertically
  - Respects minimum and maximum size constraints

**Floating Windows**
- **Float a Dock**: Drag title bar away from the main window to float it
  - Floating windows can be positioned anywhere on screen
  - Each floating window has independent window controls (minimize, maximize, close)
  - Windows remember their floating state across sessions
- **Dock a Float**: Drag floating window title bar back to the main window to re-dock it

**Closing & Toggling Visibility**
- **Close a Dock**: Click the <kbd>✕</kbd> button on the dock title bar
  - Docked windows can be closed/hidden without losing their configuration
  - Closed docks can be reopened via menu or keyboard shortcut
  - Data is preserved when docks are hidden
- **Toggle Dock Visibility**: View menu contains options to show/hide all docks
  - Use keyboard shortcuts or menu items to toggle dock visibility
  - Useful for maximizing workspace for specific tasks

**Layout Persistence**
- Dock positions, sizes, and floating state are automatically saved
- Layout is restored when the application restarts
- Can reset to default layout if needed

## Menu Bar Actions

### File
- **New Project**: Reload CoralNet-Toolbox (loss of data warning)
- **Open Project**: Open an existing CoralNet-Toolbox project file
  - Supports both binary (.bin) and JSON (.json) formats
  - File dialog allows selection of either format
- **Save Project**: Save current CoralNet-Toolbox project file
  - Choose between binary (.bin) or JSON (.json) format when saving
  - **Binary Format (.bin)**: 
    - Smaller file size (significant space savings for large projects)
    - Faster load/save times
    - Not human-readable (cannot be edited with a text editor)
    - Recommended for production use and large datasets
  - **JSON Format (.json)**:
    - Human-readable text format
    - Can be edited with any text editor for manual fixes
    - Larger file size
    - Better for version control and sharing human-readable backups

- **Import**:
  - **Rasters**:
    - **Images**: Load image files (PNG, JPG, etc.)
    - **Videos**: Import video files to the project
    - **Frames from Video**: Extract and import frames from a video file
    - **Orthomosaics**: Import orthomosaic GeoTIFF files with geospatial metadata
  - **Labels**:
    - **Labels (JSON)**: Load label data from a JSON file
    - **CoralNet (CSV)**: Load label data from a CoralNet CSV file
    - **TagLab (JSON)**: Load label data from a TagLab JSON file
  - **Annotations**:
    - **Annotations (JSON)**: Load annotation data from a JSON file
    - **CoralNet (CSV)**: Load annotation data from a CoralNet CSV file
    - **TagLab (JSON)**: Load annotation data from a TagLab JSON file
    - **Squidle+ (JSON)**: Load annotation data from a Squidle+ JSON file
    - **Viscore (CSV)**: Load annotation data from a Viscore CSV file
  - **Masks**: Import segmentation mask images as MaskAnnotations
  - **Dataset**: Import a YOLO dataset for machine learning (Detection, Instance Segmentation)
    - **Image Copying**: All images from the dataset are copied into the project directory
      - Creates a copy in your project workspace (original files remain untouched)
      - Supports PNG, JPG, BMP, TIF and other common image formats
      - Images organized by split (train, val, test) during import
    - **Annotation Importing**: YOLO format annotations are converted and imported
      - Detection datasets: Bounding boxes converted to Rectangle annotations
      - Segmentation datasets: Polygon masks converted to Polygon annotations
      - Labels automatically created from dataset class names
      - Annotations linked to corresponding images by filename
    - **Dataset Organization**:
      - Select train/val/test splits to import (can choose subsets)
      - Dataset folder structure automatically recognized
      - Handles standard YOLO directory layout automatically
    - **Supported Dataset Types**:
      - Detection (bounding boxes)
      - Instance Segmentation (polygon masks)
      - Classification (organized in class subdirectories)

- **Export**:
  - **Labels**:
    - **Labels (JSON)**: Save label data to a JSON file
    - **TagLab (JSON)**: Save label data to a TagLab JSON file
  - **Annotations**:
    - **Annotations (JSON)**: Save annotation data to a JSON file
    - **CoralNet (CSV)**: Save annotation data to a CoralNet CSV file
    - **Viscore (CSV, JSON)**: Save annotation data in Viscore format
    - **TagLab (JSON)**: Save annotation data to a TagLab JSON file
    - **GeoJSON (JSON)**: Save annotations as GeoJSON for mapping software (only for GeoTIFFs with CRS and Transforms data)
    - **Masks (PNG/BMP/TIF/RLE)**: Save annotations as segmentation mask images
      - **Export Modes** (choose one):
        - **Semantic Segmentation (Integer IDs)**: Each class assigned a unique integer value (0-255)
          - Best for: Training semantic segmentation models
          - Background value configurable (0 = Background, 255 = Ignore Index)
          - Use 255 for sparse annotations (model ignores unlabeled areas), 0 for exhaustive labeling
        - **Structure from Motion (SfM) Binary Mask**: Binary foreground/background masks
          - Best for: 3D reconstruction software (Metashape, Agisoft, etc.)
          - Foreground value: 255 (objects to keep), Background: 0 (areas to ignore)
          - Preserves depth information for photogrammetry pipelines
        - **Visualization (RGB Colors)**: Human-readable color-coded masks
          - Best for: Visual inspection, reports, presentations, qualitative analysis
          - Colors automatically assigned from your label definitions
          - Directly viewable without lookup tables
      
      - **File Formats**:
        - **PNG** (recommended): Lossless compression, fast loading, widely compatible
        - **BMP**: Uncompressed, larger files, maximum compatibility
        - **TIF**: Supports georeferencing for spatial data preservation
        - **RLE (.txt)**: Run-length encoded text format, minimal storage for sparse masks
      
      - **Georeferencing** (TIF format only):
        - Preserve geographic CRS and transform from source images
        - Recommended for orthomosaics and geospatial workflows
      
      - **Annotations to Include**:
        - **Mask Annotations**: Base layer (manually painted masks)
        - **Patches**: Converted to circular regions
        - **Rectangles**: Filled rectangular regions
        - **Polygons**: Precise polygonal regions
        - **Negative Samples**: Export masks for images with NO annotations (useful for training)
      
      - **Label Configuration**:
        - **Include/Exclude Labels**: Select which labels to include in export
        - **Layer Order**: Control rendering order for overlapping annotations
          - Labels higher in list drawn first (behind), lower in list drawn last (on top)
          - Important for complex scenes with overlapping annotations
        - **Mask Values** (Semantic/SfM): Assign integer value (0-255) per label
        - **Colors** (Visualization): Automatically use label colors or customize
      
      - **Output**:
        - Creates a folder with exported masks (one per image)
        - Metadata file included (class_mapping.json or color_legend.json)
        - Handles images with no annotations based on your settings
  - **Dataset**: Create a YOLO dataset for machine learning
    - **Classify**: Export classification dataset
      - Exports Patches, Rectangles, and Polygons as classified samples
      - Each annotation becomes a single classified crop
      - Useful for training image classification models
    - **Detect**: Export object detection dataset
      - Exports Rectangles and Polygons as bounding boxes
      - Creates YOLO detection-formatted annotations
      - Includes coordinate normalization for YOLO format
    - **Segment**: Export instance segmentation dataset
      - Exports Polygons as instance masks with per-object labels
      - Creates pixel-level segmentation masks
      - Each polygon becomes a labeled instance in the mask
    - **Semantic**: Export semantic segmentation dataset
      - Exports Mask annotations for semantic segmentation
      - Handles unlabeled/background regions
      - Useful for training pixel-wise segmentation models
    
    **Export Dataset Features** (all export types):
    - **Train/Validation/Test Split**: Configurable ratios (default 70/20/10)
      - Split by image for static images
      - Split by frame for video data
    - **Label Selection**: Filter by specific labels or export all
    - **Video Frame Sampling**: Optional frame stride to sample every Nth frame
    - **Automatic Crop Generation**: Crops are automatically generated and organized
    - **Output Directory**: Specify custom export location
    - **Format Compliance**: Exports follow YOLO format standards for compatibility
  - **Spatial Metrics**: Export spatial metrics and statistics

### Utilities

- **Sample Patches**: Automatically generate Patch annotations
  - **Sampling Method**: Choose between Random, Stratified Random, or Uniform distribution
  - **Number of Annotations**: Specify how many annotations to generate
  - **Annotation Size**: Set the size of the generated patch annotations
  - **Label As**: Choose which label to assign to generated annotations
  - **Exclude Regions**: Option to prevent sampling in areas with existing annotations
  - **Margins**: Define image boundary constraints for sampling:
    - Set margins in pixels or percentage
    - Configure different values for top, right, bottom, and left edges
    - Annotations will only be placed within these margins
  - Select images by highlighting them in the ImageWindow

- **Work Areas**: Manage work areas for batch processing
  - Pre-compute and organize multiple work areas / tiles for selected images
  - Select images by highlighting them in the ImageWindow

- **Set Image Scale**: Calibrate pixel-to-distance conversion
  - Draw a reference line to establish the scale for measurements
  - Used by the Scale Tool for distance and area calculations

### AI-Assist

- **Segment Anything (SAM)**:
  - **Deploy Predictor**: Deploy SAM, EdgeSAM, MobileSAM, etc. to use interactively (points, box, bounding box)
    - Use the SAM Tool to add points or draw boxes for interactive segmentation
  - **Deploy Generator**: Deploy FastSAM to automatically segment the entire image
    - Automatically generate segmentations from loaded annotations

- **See Anything (YOLOE)**:
  - **Train Model**: Train a custom See Anything (YOLOE) model from annotations
  - **Deploy Predictor**: Deploy a YOLOE model to use interactively within the same image
  - **Deploy Generator**: Deploy a YOLOE model to use like a detector / segmentor, referencing other images' annotations
    - Select the YOLOE model and parameters
    - Choose a reference label, then select image(s) with reference annotations (rectangles or polygons)
    - Generate visual prompt encodings (VPEs) from reference images / annotations
    - Use the loaded model with VPEs on new images or work areas

- **Feature Selector**:
  - **Deploy Model**: Deploy a dense feature extraction model for semantic similarity queries
    - Use the Feature Select Tool to click-to-query semantic similarity across images
    - Supports binary mode (single object) and multi-class mode (multiple classes)

### Machine Learning

- **Machine Learning**:
  - **Merge Datasets**: Merge multiple Classification datasets together
  - **Tile Dataset**: Split datasets into tiles for large-image processing
    - **Classify**: Tile classification datasets
    - **Detect**: Tile detection datasets
    - **Segment**: Tile segmentation datasets
    - **Semantic**: Tile semantic segmentation datasets
  - **Pre-Train Model**: Pre-train model encoder using self-supervised learning
  - **Train Model**: Train a machine learning model
    - **Classify**: Train classification model
    - **Detect**: Train object detection model
    - **Segment**: Train instance segmentation model
    - **Semantic**: Train semantic segmentation model
  - **Evaluate Model**: Evaluate a trained model
    - **Classify**: Evaluate classification model performance
    - **Detect**: Evaluate detection model performance
    - **Segment**: Evaluate segmentation model performance
    - **Semantic**: Evaluate semantic segmentation model performance
  - **Optimize Model**: Convert model format for deployment (TensorRT, ONNX, etc.)
  - **Deploy Model**: Make predictions using a trained model
    - **Classify**: Deploy for classification inference
    - **Detect**: Deploy for detection inference
    - **Segment**: Deploy for segmentation inference
    - **Semantic**: Deploy for semantic segmentation inference
      - Generates pixel-wise segmentation masks for each class
      - **Convert Masks to Polygons**: Automatically trace predicted masks into polygon annotations
        - Converts raster mask output to vector polygon format
        - Creates one polygon per connected region per class
        - Polygons are editable after creation for refinement
        - Useful for precise boundary delineation
        - Configurable simplification level for polygon complexity
  - **Batch Inference**: Run inference on multiple selected images or a single video
    - **Input Options**: Select multiple images OR one video file (not both, not multiple videos)
    - **Live Mode**: Display inference results in real-time as the model processes
    - **Save Annotations**: Optional toggle to save generated annotations to the project
      - When disabled: Preview mode shows results without persisting to the project
      - When enabled: Annotations are added to the project for each image/frame

### CoralNet

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

### Explorer

The Explorer is a dual-window system for browsing and analyzing annotations using embeddings (high-dimensional feature vectors). It enables exploration of annotations through two linked views: a gallery viewer and an embedding visualization, with real-time synchronization and feature caching.

#### Annotation Gallery Viewer (Annotation Viewer Window)

The gallery displays annotation crops as a scrollable grid of thumbnail images. It includes:

**Filtering & Sorting**
- **Image Filter**: Multi-select to show annotations from specific images
- **Label Filter**: Multi-select by annotation label (e.g., coral, sand, rock)
- **Annotation Type Filter**: Filter by Patch, Rectangle, Polygon, or MultiPolygon
- **Apply Filter Button**: Explicitly apply current filter settings (gallery remains in placeholder until applied)
- **Clear Button**: Reset gallery and return to placeholder state

**Sorting Options**
- **None**: Display in current order without grouping
- **Label**: Group by label, sorted by confidence within each label
- **Image**: Group by image filename
- **Confidence**: Group by confidence buckets (0-10%, 10-20%, ... 90-100%, plus Verified)
- **Area**: Sort by annotation area (smallest to largest)
- **Cluster**: Sort by K-Means clustering results from Embedding Viewer (enabled when clusters available)

**Display Controls**
- **Isolate Selection**: Show only selected annotations (double-click empty space to exit)
- **Size Adjustment**: <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd> to resize gallery thumbnails (32-256px range)
- **Sticky Headers**: Automatically highlights the current group header as you scroll

**Selection & Navigation**
- <kbd>Left-Click</kbd>: Select a single annotation
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add/remove from selection
- <kbd>Shift</kbd> + <kbd>Left-Click</kbd>: Select range between two annotations
- <kbd>Ctrl</kbd> + <kbd>Drag</kbd>: Box-select multiple annotations
- <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Navigate to annotation in Annotation Window (zoomed/centered)
- <kbd>Double-Click</kbd>: Unselect all, exit isolation view
- <kbd>Ctrl</kbd> + <kbd>A</kbd>: Select all annotations in current filtered view
- <kbd>Ctrl</kbd> + <kbd>Space</kbd>: Confirm selected annotations with top machine confidence prediction

#### Embedding Viewer Window

The embedding viewer displays annotations as points in 2D or 3D space, where proximity indicates semantic similarity. This enables:

**Feature Extraction & Dimensionality Reduction**
- **Model Category**: Choose Color Features (HSV-based), YOLO, Transformer, TIMM, OpenCLIP, or Live Models (deployed models)
- **Model Selection**: Pick specific model from selected category
- **Embedding Technique**: PCA (fast, linear), LDA (class-aware), TSNE (nonlinear, slow), UMAP (fast nonlinear)
- **Dimensionality**: 2D (faster) or 3D (interactive rotation)
- **Apply Embeddings Button**: Extract features and run dimensionality reduction pipeline (shows progress)

**Advanced Embedding Settings** (expandable "Advanced" panel)
- **PCA before reduction**: Apply PCA dimensionality reduction before TSNE/UMAP/LDA (auto-disabled for PCA)
- **PCA components**: Number of intermediate PCA dimensions (default 50; reduces high-dim features before final reduction)
- **UMAP settings**:
  - **n_neighbors**: Number of neighbors for UMAP (2-150; default 15; higher = more global structure)
  - **min_dist**: Minimum point separation in UMAP output (0.00-0.99; lower = tighter packing)
- **TSNE settings**:
  - **Perplexity**: Effective neighborhood size (5-50; default 20)
  - **Exaggeration**: Cluster separation strength (5.0-60.0; default 5.0; higher = more distinct clusters)

**Clustering (K-Means)**
- **Cluster Button**: Run K-Means on current embedding to partition points into clusters
- **K value**: Number of clusters (2-50; default 3)
- **Cluster Space**: Cluster on 2D projected coordinates or full high-dimensional feature vectors
- **Clear Button**: Remove cluster boundaries and disable Cluster sort option
- Cluster results are colored and can be sorted via the Annotation Viewer's "Cluster" sort option

**Display & Navigation**
- **Display Mode Toggle**: Switch between dots and sprite/thumbnail view
- **Locate Button**: Show crosshair pointer to selected annotation
- **Center Button**: Pan and zoom view to center on selected point(s)
- **Home Button**: Reset view to fit all points

**Interactive Controls**
- <kbd>Left-Click</kbd>: Select an annotation
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add/remove from selection
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd> + <kbd>Drag</kbd>: Box-select multiple points
- <kbd>Right-Click</kbd> + <kbd>Drag</kbd>: Pan the view
- <kbd>Mouse Wheel</kbd>: Zoom in/out
- <kbd>Ctrl</kbd> + <kbd>Wheel</kbd>: Adjust point size
- <kbd>Double-Click</kbd>: Unselect all, exit isolation mode

**Mislabel Detection & Analysis**
- Outlier detection is performed manually by examining points that are visually distant from their label cluster in the embedding space
- Points far from their label cluster can be selected and examined in detail for potential mislabels
- **Isolate Selection**: Show only selected points (double-click empty space to exit)
  - Useful for examining specific clusters or suspicious regions in isolation

#### Feature Caching System (CacheManager)

The Explorer automatically caches extracted features to accelerate re-loading the same models:

**How Caching Works**
- Features are stored in `.cache/embedding/` directory (created automatically)
- Metadata stored in SQLite database (`manager.db`)
- Feature vectors stored in FAISS indexes (one per model, enabling fast similarity search)
- Cache is model-specific: switching models or changing extraction parameters invalidates the cache

**Cache Behavior**
- First time running a model: features extracted and automatically cached
- Second time with same model: features loaded from cache (much faster)
- Deleting annotations: cache entries removed automatically
- Changing model/settings: stale cache detected and rebuilt as needed

**Multi-Model Support**
- Cache supports multiple models simultaneously
- Each model has its own FAISS index file (`features_{model_key}.faiss`)
- Switching between models uses corresponding cached features

**Performance**
- **Color Features**: 70 dims (8×8 Hue-Saturation histogram + H/S/V moments), instant CPU extraction
- **YOLO / Transformer / TIMM / OpenCLIP**: Model-dependent feature dimensions, GPU-accelerated (if CUDA available)
- **Batch Processing**: Default batch size 512 annotations; cache flush every 4 batches to balance memory and I/O
- **Thread-Safety**: Concurrent access via SQLite locks prevents cache corruption during multi-threaded operations

**Tips**
- Use dual monitors: place Annotation Viewer and Annotation Window side-by-side to review selected annotations
- For exploring mislabels: use Embedding Viewer to identify visual outliers (points far from their label cluster), isolate them, and examine in the gallery
- Larger clusters often indicate common features; smaller/scattered points may be edge cases or mislabels
- Apply filter in Annotation Viewer before running embeddings to focus on specific images/labels
- Use Cluster sort in the Annotation Viewer after running K-Means to organize results by cluster membership

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
  - <kbd>Ctrl</kbd> + <kbd>1</kbd>: Enable Live Classification mode (requires a loaded classification model)
    - Hover to see real-time predicted labels and confidence scores

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

- **Feature Select Tool**: After a feature model is deployed
  - <kbd>Space</kbd> (or <kbd>Left-Click</kbd>, <kbd>Left-Click</kbd>): Define a work area first
  - **Binary Mode** (one object):
    - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add a positive prototype
    - <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Add a negative prototype
    - <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd>: Adjust the similarity threshold
    - Colormap dropdown and opacity slider control the heatmap overlay
  - **Multi-class Mode** (one blob per label):
    - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Assign the patch to the selected label (switch labels to add more classes)
    - <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Undo that label's last point
    - <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd>: Adjust the reject threshold
    - Preview is colored per label and tracks the annotation transparency slider
  - **Common Controls**:
    - <kbd>Ctrl</kbd> + <kbd>Alt</kbd>: Toggle between Binary and Multi-class mode
    - <kbd>N</kbd>: Refresh the yellow crosshair point suggestion
    - <kbd>Space</kbd>: Finalize and create a Polygon/Mask annotation
    - <kbd>Backspace</kbd>: Clear all points and start over

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
- **Rotate**: <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd> + <kbd>Drag</kbd>
  - Drag left/right to rotate; drag up/down to adjust rotation speed
  - Useful for examining annotations at different angles

### Baking and Unbaking Annotations

**Baking and Unbaking** allows converting between vector annotations and mask annotations:
- **Bake**: Converts vector annotations (patches, rectangles, polygons) into a mask annotation
  - Converts the geometric outlines into rasterized pixel-based regions
  - Useful for combining multiple overlapping annotations into a single mask
  - Accessed via <kbd>Ctrl</kbd> + <kbd>R</kbd> in Select tool (opens a dialog)
  - Vector annotations are rasterized and added to the mask

- **Unbake**: Converts mask regions back into vector polygon annotations
  - Traces mask pixel regions into polygonal outlines
  - Useful for converting mask-based predictions into editable vector form
  - Accessed via <kbd>Ctrl</kbd> + <kbd>R</kbd> in Select tool (opens a dialog)
  - Each connected mask region becomes a separate polygon annotation

### Video Playback Controls
When a video is loaded, additional playback controls appear:
- **Play/Pause**: Toggle video playback
- **Stop**: Stop playback and return to paused state
- **Step Back/Forward**: Move one frame at a time (disabled during playback)
- **Jump to First/Last Frame**: Jump to the beginning or end of the video
- **Jump to Previous/Next Annotated Frame**: Navigate between frames with annotations
- **Jump to Previous/Next Keyframe**: Navigate between marked keyframe frames
- **Seek Slider**: Scrub through video frames with visual feedback
  - Red tick marks indicate frames with annotations
  - Gold tick marks indicate keyframe frames
  - <kbd>Ctrl</kbd> + <kbd>Hover</kbd>: Preview frame at cursor position
  - <kbd>Ctrl</kbd> + <kbd>Click</kbd>: Jump to frame directly
- **Keyframe Toggle**: Star button to mark/unmark the current frame as a keyframe
  - Keyframes appear as gold tick marks on the slider
  - Useful for marking important frames for review or processing
- **Frame Counter**: Displays current frame number and total frame count

## Label Window
- **Move Label**: <kbd>Right-Click</kbd> and drag to reorder labels in the window
- **Label Visibility**: Each label has a checkbox to show/hide annotations of that label
  - Hidden labels retain their data and can be shown again at any time
  - Transparency slider affects all labels (both visible and hidden)

### Label Management Toolbar
Action buttons for label operations:
- **Add Label** (<kbd>+</kbd> icon): Create a new label with custom name and color
- **Delete Label** (<kbd>-</kbd> icon): Delete the selected label (disabled if no selection)
- **Edit Label** / **Merge Labels**: Modify the selected label's name/color, or merge multiple labels into one (disabled if no selection)
- **Map Labels** (bulk): Perform bulk mapping to merge multiple source labels into a single target label (disabled if no selection)
- **Lock Label** (lock icon): Lock the currently selected label to prevent accidental modifications
  - Locked labels remain locked until manually unlocked
- **Toggle All** (asterisk icon): Quickly toggle visibility of all labels
  - If all labels visible: clicking hides all
  - If any labels hidden: clicking shows all
  - Fast batch operation for many labels

### Filter & Search
- **Filter Labels**: Type in the filter box to search for specific labels by name
- Results update in real-time as you type

### Label & Annotation Counts
- **Label Count**: Displays total number of labels in the project
- **Annotation Count**: Shows annotation statistics
  - When no annotation selected: Total annotation count
  - When one annotation selected: Its index in the sequence
  - When multiple annotations selected: Number of selected annotations
  - **Editable**: Click and edit to navigate to a specific annotation by index (when in Select mode)

## Image Window
- **Select Image**: <kbd>Double-Click</kbd> on a row to load the image in the annotation window
- **Highlight Image**: <kbd>Single-Click</kbd> on a row to highlight it
  - <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Select multiple, non-adjacent rows
  - <kbd>Shift</kbd> + <kbd>Left-Click</kbd>: Select multiple, adjacent rows

### Filtering & Search
Multi-select filters and search bars to control which images are displayed:
- **Filters** (select multiple to combine):
  - **Highlighted**: Show only highlighted images
  - **Image**: Show standard image files
  - **Ortho**: Show orthomosaic GeoTIFF files
  - **Video**: Show video files
  - **Has Z-Channel**: Show images with Z-channel data
  - **Has Predictions**: Show images with model predictions
  - **Has Annotations**: Show images with annotations
  - **No Annotations**: Show images without annotations
- **Search Images**: Filter by filename or file path substring
- **Search Labels**: Filter by annotation label name (shows images containing that label)

### Information & Navigation
- **Current Image**: Displays the currently selected image in the annotation window
- **Highlighted Count**: Shows how many images are highlighted
- **Total Count**: Shows total images in the current filtered view
- **Home Button**: Click to center the table on the currently selected image
- **Toggle Highlighted Button**: Quickly highlight all filtered images or unhighlight if all are highlighted

### Context Menu
<kbd>Right-Click</kbd> on one or more highlighted rows to:
- Check / uncheck highlighted rows
- **Batch Inference**: Run inference using loaded models (opens Batch Inference Dialog)
- **Import Z-channel**: Import Z-channel data for highlighted rows (opens Z-channel import dialog)
- **Remove Z-channel**: Remove Z-channel data from highlighted rows
- **Delete Annotations**: Remove all annotations from highlighted rows
- **Delete Images**: Remove images and associated annotations from the project

### Image Preview
- **Tooltips**: Hover over a row to show image metadata (path, dimensions, annotation count, etc.)
- **Thumbnail**: Hold <kbd>Ctrl</kbd> while hovering over a row to show a preview thumbnail

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
- **Tooltips**: Hover over the window while an annotation is selected to see its metadata

## Hotkeys

### Tips & Quick Reference
- **Hover over any tool button, image row, or window** to see helpful tooltips explaining functionality
- **Many controls use <kbd>Ctrl</kbd>** as a modifier key for enhanced functionality
- **Work Area and SAM/YOLOE tools use <kbd>Space</kbd>** as a quick confirm key
- **Interactive hotkey map available** at [Hotkeys Map](https://jordan-pierce.github.io/CoralNet-Toolbox/hot-keys) with visual keyboard layout

### General
- <kbd>Escape</kbd>: Exit the program
- <kbd>Ctrl</kbd> + <kbd>S</kbd>: Save the project
- <kbd>Ctrl</kbd> + <kbd>Z</kbd>: Undo the last addition or deletion of an annotation
- <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Z</kbd>: Redo the previously undone addition or deletion of an annotation

### Navigation
- <kbd>Alt</kbd> + <kbd>Up</kbd>/<kbd>Down</kbd>: Cycle through images
- <kbd>Ctrl</kbd> + <kbd>Up</kbd>/<kbd>Down</kbd>: Cycle through labels
- <kbd>Ctrl</kbd> + <kbd>Left</kbd>/<kbd>Right</kbd>: Cycle through annotations

### Selection & Annotation Editing
- <kbd>Ctrl</kbd> + <kbd>A</kbd>: Select all annotations, unselect all annotations (press twice to toggle)
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add/remove annotation to current selection
- <kbd>Ctrl</kbd> + <kbd>Drag</kbd>: Create rectangle selection to select multiple annotations
- <kbd>Ctrl</kbd> + <kbd>Delete</kbd> / <kbd>Backspace</kbd>: Remove selected annotation(s)
- <kbd>Ctrl</kbd> + <kbd>Space</kbd>: Confirm prediction for selected annotation with top machine confidence
- <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd>: Resize selected annotation or patch preview
- <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Mouse Wheel</kbd>: Change polygon vertex count
- <kbd>Ctrl</kbd> + <kbd>X</kbd>: Cut a polygon annotation, explode a multi-polygon, or subtract polygons
- <kbd>Ctrl</kbd> + <kbd>C</kbd>: Combine multiple selected annotations (if same type and label)
- <kbd>Ctrl</kbd> + <kbd>Shift</kbd>: Show resize handles for selected annotations
- <kbd>Ctrl</kbd> + <kbd>R</kbd>: Bake or unbake annotations (opens dialog to choose between baking vectors into mask or unbaking mask into vectors; Select tool must be active)
- <kbd>Backspace</kbd> / <kbd>Delete</kbd>: Cancel current drawing (rectangle, polygon, work area, cutting line)

### Tool Control
- <kbd>Ctrl</kbd> + <kbd>Alt</kbd>: Switch between tools within the existing tool group
  - Switch from SelectTool to active annotation tool (e.g., PatchTool when PatchAnnotation selected)
  - Switch from annotation tool back to SelectTool
  - Toggle between Brush and Erase tools
  - Toggle Feature Select Tool binary/multi-class mode
- <kbd>Ctrl</kbd> + <kbd>Alt</kbd>: Create temporary work area from current view (hold to maintain)

### Work Areas
- <kbd>Space</kbd>: Create a work area from the current view (or define via left-click twice)
- <kbd>Ctrl</kbd> + <kbd>Shift</kbd>: Show removal buttons on existing work areas
- <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Backspace</kbd>: Remove all work areas in the current image
- <kbd>Backspace</kbd>: Cancel work area drawing

### Patch Tool
- <kbd>Ctrl</kbd> + <kbd>1</kbd>: Enable Live Classification mode (requires loaded classification model)
  - Hover to see real-time predicted labels and confidence scores

### SAM / YOLOE Tools
- <kbd>Space</kbd>: Set work area, run prediction, finalize, and exit
- <kbd>Backspace</kbd>: Discard unfinalized predictions
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add positive point (SAM Tool)
- <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Add negative point (SAM Tool)

### Feature Select Tool
- <kbd>Space</kbd> (or <kbd>Left-Click</kbd>, <kbd>Left-Click</kbd>): Define work area first
- <kbd>Ctrl</kbd> + <kbd>Alt</kbd>: Toggle between Binary and Multi-class mode
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add positive prototype (binary) / assign patch to selected label (multi-class)
- <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Add negative prototype (binary) / undo last point for label (multi-class)
- <kbd>Ctrl</kbd> + <kbd>Mouse Wheel</kbd>: Adjust similarity/reject threshold
- <kbd>N</kbd>: Refresh the yellow crosshair point suggestion
- <kbd>Space</kbd>: Finalize and create Polygon/Mask annotation
- <kbd>Backspace</kbd>: Clear all points and start over

### Model Predictions
After a model is loaded, use these shortcuts to run inference:
- <kbd>Ctrl</kbd> + <kbd>1</kbd>: Classification - Predict on selected Patch or all Review-labeled patches
- <kbd>Ctrl</kbd> + <kbd>2</kbd>: Object Detection - Make predictions using detection model
- <kbd>Ctrl</kbd> + <kbd>3</kbd>: Instance Segmentation - Make predictions using segmentation model
- <kbd>Ctrl</kbd> + <kbd>4</kbd>: Semantic Segmentation - Make predictions on entire image
- <kbd>Ctrl</kbd> + <kbd>5</kbd>: FastSAM - Generate automatic segmentations
- <kbd>Ctrl</kbd> + <kbd>6</kbd>: YOLOE / See Anything - Make predictions using See Anything model

### Confidence Window
- <kbd>1</kbd>-<kbd>5</kbd>: Quick-select from top 5 predictions
- <kbd>Left-Click</kbd> on confidence bar: Change annotation's label and verify it

### Mouse Controls
- <kbd>Left-Click</kbd>: Select annotation, add point, start/end shape
- <kbd>Left-Click</kbd> + <kbd>Drag</kbd>: Move selected annotation, draw shape
- <kbd>Ctrl</kbd> + <kbd>Left-Click</kbd>: Add/remove from selection, add positive point/prototype
- <kbd>Shift</kbd> + <kbd>Left-Click</kbd>: Select range of items (Image Window)
- <kbd>Ctrl</kbd> + <kbd>Drag</kbd>: Box-select multiple annotations
- <kbd>Right-Click</kbd> + <kbd>Drag</kbd>: Pan the image viewer
- <kbd>Right-Click</kbd>: Open context menu (Image Window)
- <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Center Annotation Window on selected annotation
- <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd>: Add negative point/prototype
- <kbd>Double-Click</kbd>: Load image (Image Window)
- <kbd>Mouse Wheel</kbd>: Zoom in/out, adjust patch/brush size (with Ctrl)

### Annotation Window Navigation
- <kbd>Mouse Wheel</kbd>: Zoom in and out
- <kbd>Right-Click</kbd> + <kbd>Hold</kbd> + <kbd>Drag</kbd>: Pan the image
- <kbd>Ctrl</kbd> + <kbd>Right-Click</kbd> + <kbd>Drag</kbd>: Rotate the image (left/right drag for rotation, up/down for speed)


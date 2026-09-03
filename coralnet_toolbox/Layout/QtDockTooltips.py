"""Tooltips for the dockable main windows.

Each entry is the tooltip shown when the cursor rests on a dock's tab, the
label clicked to raise that window. They describe the window as a whole: what
it shows, how it is driven, and the controls that are not exposed as buttons.
The widgets inside each dock carry their own tooltips, so these stay at the
window level rather than repeating them.

Keyed by the dock's object name, the same string passed to DockWrapper, so the
wrapper can look its own text up without the call sites repeating it.

Plain text, matching the tooltips elsewhere in the application: a title line, a
blank line, then paragraphs hand-wrapped with newlines. Avoid text that Qt
could mistake for HTML, since tooltips are rendered with Qt::AutoText.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


DOCK_TOOLTIPS = {

    "AnnotationDock":
        "Annotation Window\n"
        "\n"
        "Displays the active image, video frame, or orthomosaic\n"
        "with its annotations. Selections made here drive the\n"
        "Confidence, Metadata, Gallery, and Embeddings windows.\n"
        "\n"
        "Double-click an image in the Rasters window to load it,\n"
        "select a label in the Labels window, then choose a tool\n"
        "from the main toolbar to annotate.\n"
        "\n"
        "Mouse wheel zooms, right-click and drag pans, and\n"
        "Ctrl + right-click and drag rotates. The toolbars around\n"
        "the canvas hold display options, thresholds, and the\n"
        "video playback controls.",

    "RastersDock":
        "Rasters Window\n"
        "\n"
        "Lists every image, orthomosaic, and video imported into\n"
        "the project.\n"
        "\n"
        "Double-click a row to load it in the Annotation window.\n"
        "Single-click highlights a row instead; use Ctrl or Shift\n"
        "to highlight several. Batch operations act on the\n"
        "highlighted rows, not on the loaded image.\n"
        "\n"
        "Use the filter and search toolbars to narrow the list,\n"
        "then right-click the highlighted rows for batch\n"
        "inference, Z-channel import, or deletion.\n"
        "\n"
        "Hovering a row shows its path, dimensions, and counts.\n"
        "Hold Ctrl while hovering to preview the image itself.",

    "LabelsDock":
        "Labels Window\n"
        "\n"
        "Lists the labels defined for the project. The selected\n"
        "label is the one applied to new annotations.\n"
        "\n"
        "Click a label to make it active. Each label's checkbox\n"
        "shows or hides its annotations without deleting them,\n"
        "and right-click and drag reorders the list.\n"
        "\n"
        "The toolbars add, edit, merge, delete, and filter labels,\n"
        "and the counts beside each label show how many\n"
        "annotations use it. Hovering a label also shows its full\n"
        "name, color, and ID.",

    "ConfidenceDock":
        "Confidence Window\n"
        "\n"
        "Shows the cropped image and top five predicted labels for\n"
        "the selected annotation. Tabbed with the Metadata window;\n"
        "both follow the current selection.\n"
        "\n"
        "Click a confidence bar, or press 1-5, to assign that\n"
        "label to the annotation and mark it verified. Prev and\n"
        "Next step through annotations in creation order.\n"
        "\n"
        "The toggle beside the dimensions switches between user\n"
        "and machine confidence when both are available.\n"
        "Hovering the cropped image shows the annotation's\n"
        "computed metadata.",

    "MetaDataDock":
        "Metadata Window\n"
        "\n"
        "Shows metadata for the selected annotation(s): the custom\n"
        "fields defined for the project, built-in values computed\n"
        "from the geometry, and raw imported data. Tabbed with the\n"
        "Confidence window.\n"
        "\n"
        "Custom fields are editable; built-in and raw rows are\n"
        "read-only. With several annotations selected, an edit is\n"
        "written to all of them, and fields whose values differ\n"
        "show <multiple values> until one is entered.\n"
        "\n"
        "Fields are defined once for the project from the toolbar,\n"
        "then filled in per annotation. Built-in measurements use\n"
        "the unit set in the status bar. Mask annotations are not\n"
        "included.",

    "GalleryDock":
        "Gallery Window\n"
        "\n"
        "Shows every annotation in the project as a grid of\n"
        "cropped images, independent of the image loaded in the\n"
        "Annotation window. Paired with the Embeddings window;\n"
        "the two share a single selection.\n"
        "\n"
        "Set the image, label, and annotation type filters, then\n"
        "press Apply Filter to populate the grid; Clear resets it.\n"
        "Sorting groups the crops by label, image, confidence, or\n"
        "area, or by the clusters and similarity ranking produced\n"
        "in the Embeddings window.\n"
        "\n"
        "Controls:\n"
        "Left-click selects, Ctrl + click adds or removes,\n"
        "Shift + click selects a range, and Ctrl + drag\n"
        "box-selects. Ctrl + A selects the whole filtered view.\n"
        "Ctrl + wheel resizes the thumbnails. Ctrl + right-click\n"
        "centers the annotation in the Annotation window without\n"
        "changing the selection. Ctrl + space confirms the top\n"
        "prediction for the selection. Double-click clears the\n"
        "selection and leaves isolation.",

    "EmbeddingsDock":
        "Embeddings Window\n"
        "\n"
        "Plots the annotations shown in the Gallery window as\n"
        "points in 2D or 3D, positioned so that visually similar\n"
        "crops fall near each other. Used to find mislabels and to\n"
        "gather annotations that look alike.\n"
        "\n"
        "Filter in the Gallery window first, then choose a feature\n"
        "model and a reduction technique and press Apply\n"
        "Embeddings. Features are cached per model, so re-running\n"
        "the same model is faster. Cluster runs K-Means on the\n"
        "result, which the Gallery window can then sort by.\n"
        "\n"
        "Controls:\n"
        "Left-click selects, Ctrl + click adds or removes, and\n"
        "Ctrl + drag box-selects. Ctrl + A selects every point.\n"
        "Right-click and drag pans, the wheel zooms, and\n"
        "Ctrl + wheel resizes the points.\n"
        "Ctrl + Shift + wheel grows or shrinks the selection by\n"
        "similarity, and Escape restores the selection it started\n"
        "from. Ctrl + right-click centers the annotation in the\n"
        "Annotation window. Double-click clears the selection and\n"
        "leaves isolation.",

    "PerformanceDock":
        "Performance Window\n"
        "\n"
        "Displays real-time CPU, memory, GPU, disk, and network\n"
        "usage as sparkline graphs.\n"
        "\n"
        "Use it during inference, training, or a large import to\n"
        "see whether a run is limited by GPU memory, disk, or CPU,\n"
        "and to size batch and image settings accordingly.\n"
        "\n"
        "Monitoring runs only while the window is visible.",

    "TimerDock":
        "Timer Window\n"
        "\n"
        "Tracks time spent working.\n"
        "\n"
        "Start and Stop control the session timer, and Reset\n"
        "returns it to zero. A second timer runs whenever the\n"
        "application is open and accumulates the total duration\n"
        "across sessions, which Reset does not clear.\n"
        "\n"
        "Start, stop, and reset events are logged with the\n"
        "session.",
}

"""
BaseCanvas - Lightweight, reusable QGraphicsView subclass for image display and navigation.

This class encapsulates pure viewing responsibilities: image display, zoom/pan navigation,
Z-channel visualization, and marker slots. It is designed to be inherited by AnnotationWindow
and reused in Phase 2's context matrix for multi-viewport displays.
"""

import math
import time
import warnings
import traceback
import numpy as np

from PyQt5.QtGui import (QMouseEvent, QPixmap, QImage, QBrush, QColor, QPen,
                         QTransform, QPainter, QPainterPath, QCursor, qRgba)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer, QSize, QObject, QEvent
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsLineItem,
                             QGraphicsItem, QGraphicsPathItem, QLabel, QApplication, QFrame)

from coralnet_toolbox.utilities import get_view_scale, get_colormap

from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


#-------------------------------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------------------------------


class FastImageItem(QGraphicsItem):
    """A high-performance image item that bypasses QPixmap and draws directly via OpenGL."""
    def __init__(self):
        super().__init__()
        self._image = None
        self._scene_rect = QRectF(0, 0, 100, 100)
        self._readonly_paths = []
        # Cached QImage of all readonly paths rendered at base-image resolution.
        # Built lazily on first paint after set_readonly_annotations() so that the
        # per-frame paint cost is one drawImage call instead of 14K drawPaths.
        self._readonly_cache = None
        self._readonly_dirty = False

        # --- CRITICAL: Initialize the mask variables here! ---
        self._mask_image = None
        self._mask_opacity = 1.0

        # Optimize for rapidly changing content
        self.setCacheMode(QGraphicsItem.NoCache)

    def set_image(self, qimage, target_size=None):
        """Set the image to be drawn by this item, keeping a reference to the original QImage.

        The deep copy this used to make was a second full-resolution buffer --
        3.2 GB on a 32k raster -- bought for nothing: the QImage handed in is
        owned by the Raster, outlives this item, and is never mutated in place.
        Holding the reference is enough, and it keeps alive whatever the QImage
        itself references (`rasterio_to_qimage` pins its numpy buffer on the
        QImage as `ndarray_reference`).
        """
        if qimage is not None and not qimage.isNull():
            self._image = qimage
        else:
            self._image = qimage
        if target_size is None and self._image is not None and not self._image.isNull():
            target_width = self._image.width()
            target_height = self._image.height()
        else:
            try:
                target_width = int(target_size.width())
                target_height = int(target_size.height())
            except Exception:
                try:
                    target_width = int(target_size[0])
                    target_height = int(target_size[1])
                except Exception:
                    target_width = self._image.width() if self._image is not None and not self._image.isNull() else 100
                    target_height = self._image.height() if self._image is not None and not self._image.isNull() else 100

        self._scene_rect = QRectF(0, 0, max(1, int(target_width)), max(1, int(target_height)))
        # Image dimensions changed — invalidate the readonly cache so it rebuilds
        # at the correct resolution on next paint.
        self._readonly_cache = None
        self._readonly_dirty = bool(self._readonly_paths)
        try:
            self.update()
        except RuntimeError:
            pass

    def set_mask_image(self, qimage, opacity=1.0):
        """Provide a mask image to be drawn natively on top of the base image."""
        if qimage is not None and not qimage.isNull():
            # Zero-copy pointer to the live numpy array
            self._mask_image = qimage 
        else:
            self._mask_image = None
        self._mask_opacity = opacity
        try:
            self.update()
        except RuntimeError:
            pass

    def set_readonly_annotations(self, paths_data):
        """Pass a list of ready-to-draw paths: (QPainterPath, QColor, opacity).

        The paths are stored verbatim and rendered into a QImage cache lazily on
        the next paint(); subsequent paints just blit the cache, so pan/zoom no
        longer pays the per-path dispatch cost.
        """
        self._readonly_paths = paths_data
        self._readonly_cache = None  # Free old cache immediately
        self._readonly_dirty = bool(paths_data)
        try:
            self.update()
        except RuntimeError:
            pass

    def _build_readonly_cache(self):
        """Render all readonly paths into a single QImage at base-image resolution.

        Pen widths become fixed image-pixel widths (no cosmetic scaling), so
        outlines look slightly thinner at extreme zoom-out, but the fill carries
        the visual signal there anyway.
        """
        self._readonly_dirty = False
        if self._image is None or self._image.isNull() or not self._readonly_paths:
            self._readonly_cache = None
            return

        img = QImage(max(1, int(round(self._scene_rect.width()))), max(1, int(round(self._scene_rect.height()))), QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Shadow pen is constant — build once. Width is in image pixels.
        shadow_pen = QPen(QColor(0, 0, 0, 130), 4.0, Qt.SolidLine)
        shadow_pen.setCapStyle(Qt.RoundCap)
        shadow_pen.setJoinStyle(Qt.RoundJoin)

        for path, color_val, transparency in self._readonly_paths:
            # PASS 1: fill + dark halo
            fill_color = QColor(color_val)
            fill_color.setAlpha(transparency)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(shadow_pen)
            painter.drawPath(path)

            # PASS 2: crisp colored inner border
            painter.setBrush(Qt.NoBrush)
            pen_color = QColor(color_val)
            pen_color.setAlpha(255)
            main_pen = QPen(pen_color, 2.0, Qt.SolidLine)
            main_pen.setCapStyle(Qt.RoundCap)
            main_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(main_pen)
            painter.drawPath(path)

        painter.end()
        self._readonly_cache = img

    def boundingRect(self):
        """Return the bounding rectangle of the image for proper redraw regions."""
        if self._image is None or self._image.isNull():
            return QRectF(0, 0, 100, 100) # Fallback safe rect
        return QRectF(self._scene_rect)

    def paint(self, painter, option, widget):
        """Custom paint method to draw the image, mask, and annotations in a single pass."""
        # 1. Draw the video frame directly from RAM to the OpenGL Viewport
        if self._image is not None and not self._image.isNull():
            painter.drawImage(self._scene_rect, self._image)

        # 2. Draw the mask overlay natively (using getattr as a failsafe)
        mask = getattr(self, '_mask_image', None)
        if mask is not None and not mask.isNull():
            painter.setOpacity(self._mask_opacity)
            painter.drawImage(0, 0, mask)
            painter.setOpacity(1.0) # Reset opacity

        # 3. Draw all readonly annotations as a single pre-rendered QImage blit.
        if self._readonly_paths:
            if self._readonly_dirty or self._readonly_cache is None:
                self._build_readonly_cache()
            if self._readonly_cache is not None:
                painter.drawImage(0, 0, self._readonly_cache)


def phantom_group_key(annotation, is_selected=None):
    """The phantom-layer group key for one annotation: colour, alpha, selected.

    Single source of truth: BaseCanvas builds the whole layer from it and
    AnnotationWindow.refresh_phantom_annotations rebuilds one group from it.
    Those two used to construct the tuple independently, which is precisely how
    the full and incremental paths drifted apart before.
    """
    c = annotation.label.color
    if is_selected is None:
        is_selected = bool(annotation.is_selected)
    return (c.red(), c.green(), c.blue(), annotation.transparency, bool(is_selected))


class PhantomHitIndex:
    """Uniform grid over annotation bounding boxes, for click resolution.

    Selecting an annotation means finding which one is under the cursor, and
    the phantom layer draws as merged paths that Qt cannot hit-test back to an
    annotation. So the scan is done in Python over every annotation in the
    image -- 4 ms per click at 15k annotations, on every click including the
    ones that land on empty space to deselect.

    Built as a side effect of rendering the phantom layer, over exactly the
    annotations that layer draws. That is deliberate: it inherits the phantom
    layer's invalidation for free. Any mutation that would leave this index
    stale -- a move, a delete, a label change -- already has to rebuild the
    phantom layer or the canvas would be drawing the annotation in the wrong
    place, which is a louder bug than a missed click.
    """

    CELL_PX = 512.0

    def __init__(self, annotations):
        self._cells = {}
        cell = self.CELL_PX
        for annotation in annotations:
            bbox = getattr(annotation, 'cropped_bbox', None)
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            # Register in every cell the bbox touches, so a shape larger than
            # one cell is still found from anywhere inside it.
            for cx in range(int(x0 // cell), int(x1 // cell) + 1):
                for cy in range(int(y0 // cell), int(y1 // cell) + 1):
                    self._cells.setdefault((cx, cy), []).append(annotation)

    def candidates(self, x, y):
        """Annotations whose bbox cell contains the point, in insertion order.

        Callers wanting topmost-first should reverse the result, matching the
        ``reversed(all_annotations)`` the linear scan uses for Z-order.
        """
        cell = self.CELL_PX
        return self._cells.get((int(x // cell), int(y // cell)), ())


class PhantomGroupItem(QGraphicsPathItem):
    """One colour group of the phantom layer, bucketed on a coarse grid.

    Still one scene item per colour group, but the group's sub-paths are split
    across a spatial grid and paint() draws only the buckets the view actually
    exposes. That collapses two independent costs:

    Culling. Qt hands an item its whole bounding rect as ``exposedRect`` unless
    ItemUsesExtendedStyleOption is set, so this layer used to re-walk every
    sub-path in the image every frame regardless of zoom -- 13k 50-vertex
    polygons cost 60 ms at 32x with five of them on screen.

    Rasterizer locality. A single QPainterPath spanning a 16k image with 663k
    elements costs far more to rasterize than the same geometry drawn as a few
    hundred local paths, because the scanline structure is built across the
    whole extent. That is why bucketing also wins ~10x at fit zoom, where
    nothing is culled at all: 246 ms -> 24 ms.

    Below LOD_PEN_CUTOFF (scene px -> screen px scale) the cosmetic outline is
    invisible anyway, so the pen is skipped and only the fill is drawn.

    One deliberate rendering change comes with this. A single merged path let
    WindingFill union overlapping same-label shapes so their fill was blended
    once; drawing per-bucket means two same-label annotations that overlap
    across a bucket seam blend twice, and their overlap reads slightly more
    opaque. Nothing is ever lost or added -- only the alpha in that overlap
    changes -- and the effect scales with how much a label's annotations cover
    each other: measured at 0.05% of pixels when a label's annotations cover 5%
    of the image, ~2% at 25% coverage, and 23-38% once a single label's
    annotations blanket the image. Different labels were always separate items
    and so always blended twice; this only ever affects a label against itself.
    """
    LOD_PEN_CUTOFF = 0.25

    def __init__(self, buckets, bucket_px, bounds, overhang):
        """
        Args:
            buckets (dict): (cell_x, cell_y) -> QPainterPath of that cell's
                annotations. A cell is keyed by each annotation's bbox centre,
                so no annotation is ever split across two paths.
            bucket_px (float): grid pitch in scene units.
            bounds (QRectF): union of every bucket's content.
            overhang (float): the furthest any annotation reaches beyond the
                cell it was filed under. Queries are inflated by it, because an
                annotation wider than the grid pitch is still drawn whole from
                its centre cell and would otherwise vanish when the view showed
                only its edge.
        """
        super().__init__()
        self._buckets = buckets
        self._bucket_px = bucket_px
        self._bounds = bounds
        self._overhang = overhang
        # Without this Qt does not bother narrowing exposedRect, and the whole
        # point of the bucketing is lost.
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def boundingRect(self):
        # Computed from the buckets rather than from a merged path: keeping a
        # second copy of every element purely to let QGraphicsPathItem derive
        # this rect doubled both layer-build time and the layer's memory.
        return self._bounds

    def shape(self):
        """Empty -- this layer is decorative and must never be hit-tested.

        QGraphicsPathItem.shape() strokes the item's path, and QGraphicsScene
        .items() calls it on every mouse press: on 13k 50-vertex polygons that
        measured 291 ms per click. Nothing wants these items back from a hit
        test. SelectTool maps Qt items to annotations only for fully
        materialised ones, which a merged group is not, and resolves phantoms
        through the canvas's own spatial index instead -- see
        get_phantom_hit_index.
        """
        return QPainterPath()

    def paint(self, painter, option, widget=None):
        rect = option.exposedRect
        pitch = self._bucket_px
        pad = self._overhang
        get = self._buckets.get

        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        painter.setBrush(self.brush())
        painter.setPen(Qt.NoPen if lod < self.LOD_PEN_CUTOFF else self.pen())

        for cx in range(int((rect.left() - pad) // pitch),
                        int((rect.right() + pad) // pitch) + 1):
            for cy in range(int((rect.top() - pad) // pitch),
                            int((rect.bottom() + pad) // pitch) + 1):
                path = get((cx, cy))
                if path is not None:
                    painter.drawPath(path)


def phantom_bucket_px(scene_rect, n_annotations):
    """Grid pitch for the phantom layer, in scene units.

    Sized so a bucket holds a handful of annotations. Both ends of that matter
    and they pull against each other: buckets far larger than the annotation
    spacing stop culling anything (at 4096 px on a 16k image, paint at 16x was
    11 ms against 1.4 ms at 512), while buckets much smaller degenerate to one
    drawPath per annotation and give the rasterizer nothing to batch (128 px
    cost 36 ms at fit zoom against 24 ms at 512, and tripled layer-build time).

    The measured optimum tracked annotations-per-bucket rather than image size
    or annotation count alone -- 256 px was best on a dense 4k image, 512 px on
    a sparse 16k one, and both fall out of a target of ~16 per bucket.

    Snapped to a power of two so the answer is stable. ``n_annotations`` counts
    only the annotations being drawn phantom, which drops every time the user
    selects something, and a pitch that drifts with it moves the bucket seams --
    changing the alpha where same-label annotations overlap across one. A raw
    formula gave 366 px for 2000 annotations and 397 px once 300 of them were
    selected, so selecting and deselecting left 3.5k pixels a different shade
    than they started. Quantising absorbs that: the count has to roughly double
    before the pitch moves at all, and the two measured optima are powers of two
    already, so nothing is given up for it.
    """
    area = max(1.0, scene_rect.width() * scene_rect.height())
    pitch = math.sqrt(area * 16.0 / max(1, n_annotations))
    pitch = 2.0 ** round(math.log2(max(1.0, pitch)))
    return min(4096.0, max(256.0, pitch))


def bucket_annotation_paths(annotations, bucket_px):
    """Sort one group's cached painter paths into grid cells.

    Args:
        annotations: the group's annotations.
        bucket_px (float): grid pitch in scene units.

    Returns:
        (buckets, bounds, overhang) ready for PhantomGroupItem, or
        (None, None, None) if nothing in the group had a usable path.
    """
    buckets = {}
    get = buckets.get
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    widest = 0.0

    for annotation in annotations:
        try:
            path = annotation.get_cached_painter_path()
        except (NotImplementedError, AttributeError):
            continue
        if path is None or path.isEmpty():
            continue

        rect = path.boundingRect()
        x0 = rect.x()
        y0 = rect.y()
        w = rect.width()
        h = rect.height()

        key = (int((x0 + w * 0.5) // bucket_px),
               int((y0 + h * 0.5) // bucket_px))
        cell = get(key)
        if cell is None:
            # WindingFill on every bucket, matching the single merged path this
            # replaced: overlapping same-label shapes must union, not punch
            # holes in each other. Correctness still depends on rings arriving
            # with normalized winding -- see Annotation._add_ring.
            cell = QPainterPath()
            cell.setFillRule(Qt.WindingFill)
            buckets[key] = cell
        cell.addPath(path)

        # Tracked as floats rather than QRectF.united() per annotation: the
        # rect algebra was the single largest line in a build profile.
        if x0 < min_x:
            min_x = x0
        if y0 < min_y:
            min_y = y0
        if x0 + w > max_x:
            max_x = x0 + w
        if y0 + h > max_y:
            max_y = y0 + h
        if w > widest:
            widest = w
        if h > widest:
            widest = h

    if not buckets:
        return None, None, None

    # An annotation is filed by its centre, so it reaches at most half its own
    # width past the cell it landed in. Half the widest annotation is therefore
    # a safe bound on the overhang, and costs two comparisons instead of four
    # subtractions per annotation.
    #
    # A cosmetic pen also strokes half a device pixel outside the fill, which
    # exceeds half a scene unit only when zoomed out; a small constant margin
    # covers it, and an over-large bounding rect costs nothing but a marginally
    # wider exposedRect.
    bounds = QRectF(min_x - 2.0, min_y - 2.0,
                    (max_x - min_x) + 4.0, (max_y - min_y) + 4.0)
    return buckets, bounds, widest * 0.5


class ColorMapOverlay:
    """A reusable indexed-8 colormap overlay layer for a QGraphicsScene.

    Renders a ``uint8`` index field through a 256-entry color table as a single
    QGraphicsPixmapItem. Recoloring is a cheap color-table swap (no per-pixel
    numpy LUT); opacity is a single ``item.setOpacity()``. The index grid is
    scaled with smooth interpolation to fill a target scene rect, so the SAME
    overlay serves both a full-resolution field (Z-channel depth) and a small
    upsampled grid (feature similarity).

    Palette convention:
        index 0     -> fully transparent (nodata / invalid / off)
        index 1..N  -> colormap ramp (N = 255, or 254 when a scrim is reserved)
        index 255   -> optional scrim color (e.g. below-threshold dimming)
    """

    def __init__(self, z_value=-5, smooth=True):
        self._z_value = z_value
        self._smooth = smooth
        self.item = None
        self._index = None          # uint8 [h, w] palette indices
        self._color_table = None    # list[QRgb], 256 entries
        self.colormap_name = None
        self._opacity = 0.5
        self._target_rect = None
        self._scrim_rgba = None

    def _ensure_item(self, scene):
        """Create (or re-create on a new scene) the backing pixmap item."""
        if self.item is None or self.item.scene() is not scene:
            self.item = QGraphicsPixmapItem()
            if self._smooth:
                self.item.setTransformationMode(Qt.SmoothTransformation)
            self.item.setZValue(self._z_value)
            self.item.setOpacity(self._opacity)
            self.item.setAcceptHoverEvents(False)
            scene.addItem(self.item)
        return self.item

    @staticmethod
    def _build_color_table(colormap_name, scrim_rgba=None):
        """Return a 256-entry ARGB table; index 0 transparent, 255 optional scrim."""
        n_levels = 254 if scrim_rgba is not None else 255
        cmap = get_colormap(colormap_name)
        if cmap is None:
            raise ValueError(f"Unknown colormap: {colormap_name!r}")
        lut = cmap.getLookupTable(nPts=n_levels, alpha=True)
        table = [0]  # index 0 -> 0x00000000 (fully transparent)
        for i in range(n_levels):
            r, g, b, a = (int(v) for v in lut[i])
            table.append(qRgba(r, g, b, a))
        # Pad to 256 (when a scrim is reserved we stopped at 1..254).
        while len(table) < 256:
            table.append(table[-1])
        if scrim_rgba is not None:
            sr, sg, sb, sa = scrim_rgba
            table[255] = qRgba(int(sr), int(sg), int(sb), int(sa))
        return table

    def set_colormap(self, colormap_name, scrim_rgba=None):
        """Swap the colormap (and optional scrim) and re-blit the existing field."""
        self.colormap_name = colormap_name
        self._scrim_rgba = scrim_rgba
        self._color_table = self._build_color_table(colormap_name, scrim_rgba)
        self._reblit()

    def set_indices(self, scene, index_array, target_rect=None):
        """Store a new uint8 index field and (re)build the pixmap on ``scene``."""
        self._index = np.ascontiguousarray(index_array.astype(np.uint8, copy=False))
        if target_rect is not None:
            self._target_rect = target_rect
        if self._color_table is None:
            self._color_table = self._build_color_table('Plasma', self._scrim_rgba)
        self._ensure_item(scene)
        self._reblit()
        self.item.show()

    def _reblit(self):
        """Rebuild the pixmap from the cached index field + color table."""
        if self.item is None or self._index is None or self._color_table is None:
            return
        h, w = self._index.shape
        q_img = QImage(self._index.data, w, h, w, QImage.Format_Indexed8)
        q_img.setColorTable(self._color_table)
        # .copy() detaches the pixmap from the live numpy buffer.
        self.item.setPixmap(QPixmap.fromImage(q_img.copy()))

        rect = self._target_rect
        if rect is not None and rect.width() > 0 and rect.height() > 0:
            self.item.setTransform(
                QTransform().scale(rect.width() / float(w), rect.height() / float(h))
            )
            self.item.setPos(rect.left(), rect.top())
        else:
            self.item.setTransform(QTransform())
            self.item.setPos(0, 0)

    def set_opacity(self, opacity):
        """Set overlay opacity in [0, 1]."""
        self._opacity = max(0.0, min(1.0, opacity))
        if self.item is not None:
            self.item.setOpacity(self._opacity)

    def show(self):
        if self.item is not None:
            self.item.show()

    def hide(self):
        if self.item is not None:
            self.item.hide()

    def is_visible(self):
        return self.item is not None and self.item.isVisible()

    def clear(self):
        """Remove the overlay item from its scene and drop the cached field."""
        if self.item is not None:
            try:
                if self.item.scene() is not None:
                    self.item.scene().removeItem(self.item)
            except Exception:
                pass
        self.item = None
        self._index = None


class BaseCanvas(QGraphicsView):
    """
    Lightweight viewport for image display with native zoom/pan navigation.
    
    Signals:
        viewNavigated: Emitted after zoom/pan, carrying (center_x, center_y, zoom_factor)
        mouseHovered: Emitted on mouse move, carrying scene coordinates (x, y)
    
    Attributes:
        scene: QGraphicsScene for rendering
        active_image: Whether an image is currently loaded
        pixmap_image: Always None. The canvas draws from a QImage; use
            get_image_dimensions()/get_image_rect() for geometry
        current_image_path: String identifier for the current image
        zoom_factor: Current zoom level
        z_item: QGraphicsPixmapItem for Z-channel visualization layer
    """
    
    viewNavigated = pyqtSignal(float, float, float)  # center_x, center_y, zoom_factor
    mouseHovered = pyqtSignal(float, float)  # scene_x, scene_y
    
    def __init__(self, parent=None):
        """Initialize the base canvas."""
        super().__init__(parent)
        
        # Create and set the scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set dark background for the annotation workspace
        try:
            self.scene.setBackgroundBrush(QBrush(app_theme.BACKGROUND_COLOR))
            self.setBackgroundBrush(QBrush(app_theme.BACKGROUND_COLOR))
            self.viewport().setStyleSheet(f"background-color: {app_theme.BACKGROUND_COLOR.name()};")
        except Exception:
            pass
        
        # Image state
        self.pixmap_image = None
        self.active_image = False
        self.current_image_path = None
        self._base_image_item = None  # Reference to the base image QGraphicsPixmapItem
        self._image_dimensions = None
        
        # Navigation state
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0  # Tracks absolute rotation in degrees
        self._rotate_active = False
        self._rotate_start_angle = 0.0
        self._rotate_start_mouse_angle = 0.0
        self._rotate_start_canvas_angle = 0.0
        self._min_zoom = 1.0
        self._pan_active = False
        self._pan_start = None
        
        # Z-channel (depth) data state. The depth *data* keeps its z_ names; the
        # *rendering* is delegated to a generic ColorMapOverlay so the same
        # indexed-8 path also serves the feature-similarity overlay.
        self._z_overlay = ColorMapOverlay(z_value=-5, smooth=True)
        # The colormap dropdown / opacity slider drive whichever overlay is the
        # active target; the Z (depth) overlay is the default. The FeatureSelect
        # tool repoints this at the feature overlay while it is active.
        self._active_colormap_overlay = self._z_overlay
        self.z_data_raw = None  # Raw Z-channel data
        # Indexed visualization: a single-byte palette index per pixel (0 = nodata,
        # transparent; 1..255 = normalized value). Colormap changes only swap the
        # color table — no per-pixel numpy LUT expansion, ¼ the pixmap upload.
        self.z_index = None        # uint8 [h, w] palette indices
        self.z_colormap_name = None  # currently applied colormap ('None' = hidden)
        self.z_data_min = None  # Min value in valid data
        self.z_data_max = None  # Max value in valid data
        self.z_data_shape = None  # Shape of Z-channel array
        self.z_nodata_mask = None  # Boolean mask of invalid pixels
        self.dynamic_z_scaling = False  # Whether to rescale based on visible area
        self._dynamic_range_timer = QTimer()  # Debounce timer for dynamic range updates
        self._dynamic_range_timer.setSingleShot(True)
        self._dynamic_range_timer.timeout.connect(self.update_dynamic_range)
        # default debounce delay (ms) — AnnotationWindow can override
        self.dynamic_range_update_delay = 500
        # Marker slots (containers; Phase 4 will populate these)
        self._static_marker = None
        self._dynamic_marker = None
        self._cursor_preview_item = None  # Preview rect for tool cursor propagation
        self._mask_overlay_item = None    # Read-only MaskAnnotation overlay for brush propagation
        # Feature-similarity heatmap overlay (FeatureSelectTool). Same indexed-8
        # ColorMapOverlay machinery as the Z layer; sits just above it.
        self._feature_overlay = ColorMapOverlay(z_value=-4, smooth=True)
        # Multi-class label overlay (FeatureSelectTool multi-class mode). Reuses
        # the indexed-8 ColorMapOverlay, but its table maps index k+1 -> a label
        # color (0 = unlabeled, transparent). Nearest scaling keeps class regions
        # crisp; sits just above the feature overlay, still below annotations.
        self._label_overlay = ColorMapOverlay(z_value=-3, smooth=False)
        self._perimeter_overlay = None    # Viewport border overlay

        # Read-only annotation overlays (Phase 6).
        # Dict keyed by (r, g, b, transparency, is_selected) so incremental
        # updates can patch just the affected path item instead of rebuilding
        # every group.
        self._readonly_annotation_items = {}
        # Grid pitch the phantom layer was last built at, so an incremental
        # single-group rebuild files its annotations into the same cells the
        # rest of the layer already uses. Recomputed on every full rebuild.
        self._phantom_bucket_px = 512.0
        # Layer stacking order, by group key. Every phantom group shares one Z
        # value, so Qt stacks them by insertion order — which an incremental
        # rebuild would otherwise disturb. See _restack_phantom_group.
        self._phantom_group_order = []
        # Spatial index over the same annotations, for click resolution. Built
        # lazily on first click after each layer rebuild; None means fall back
        # to a linear scan. See get_phantom_hit_index.
        self._phantom_hit_index = None
        self._phantom_hit_source = None
        self._phantom_hit_epoch = None

        # Placeholder label for empty canvas
        self._placeholder_label = QLabel(
            "No image loaded\nImport or drag and drop an image.",
            self.viewport()
        )
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setStyleSheet(
            f"color: {app_theme.TEXT_MUTED_COLOR.name()}; font-size: 14px; background-color: transparent;"
        )
        
        # View transformation settings
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
        
        # --- NEW OPTIMIZATION FLAGS ---
        # 1. SmartViewportUpdate analyzes the bounding rects of changes to decide 
        # whether to redraw a specific region or the whole viewport.
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        
        # 2. Prevent Qt from saving/restoring the painter state for every single item.
        # This saves massive CPU overhead when rendering thousands of items.
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState)

        # Favor visual quality over raw interaction speed.
        self.setRenderHints(
            QPainter.Antialiasing |
            QPainter.TextAntialiasing |
            QPainter.SmoothPixmapTransform
        )

        # During pan/zoom/rotate, drop expensive render hints; restore shortly
        # after the interaction goes idle.
        self._quality_hints = (QPainter.Antialiasing |
                               QPainter.TextAntialiasing |
                               QPainter.SmoothPixmapTransform)
        self._fast_hints_active = False
        self._interaction_idle_timer = QTimer(self)
        self._interaction_idle_timer.setSingleShot(True)
        self._interaction_idle_timer.setInterval(150)
        self._interaction_idle_timer.timeout.connect(self._restore_quality_hints)

        # Wheel-zoom coalescing. 16 ms is one frame at 60 Hz: long enough to
        # absorb a burst, short enough that a single deliberate click of the
        # wheel still feels immediate.
        self._pending_wheel_factor = 1.0
        self._pending_wheel_pos = None
        self._wheel_coalesce_timer = QTimer(self)
        self._wheel_coalesce_timer.setSingleShot(True)
        self._wheel_coalesce_timer.setInterval(16)
        self._wheel_coalesce_timer.timeout.connect(self._flush_pending_wheel_zoom)

    # ==================== Navigation Events ====================
    
    def wheelEvent(self, event: QMouseEvent):
        """Handle mouse wheel events for zooming."""
        if not self.active_image:
            return

        self._enter_fast_paint_mode()

        # One flick of a wheel delivers a dozen events. Applying each one
        # immediately changes the view transform a dozen times, and every
        # change repaints the whole viewport to show an intermediate zoom
        # nobody perceives. Accumulate instead and apply once a frame: twelve
        # repaints become one.
        #
        # The anchor is the newest event's position. A burst comes from a
        # stationary cursor, so the last position is the one the user means to
        # zoom around, and it is what applying the events one by one would have
        # converged to anyway.
        self._pending_wheel_factor *= 1.1 if event.angleDelta().y() > 0 else 0.9
        self._pending_wheel_pos = event.pos()
        if not self._wheel_coalesce_timer.isActive():
            self._wheel_coalesce_timer.start()

    def _flush_pending_wheel_zoom(self):
        """Apply one burst's worth of accumulated wheel zoom."""
        factor = self._pending_wheel_factor
        pos = self._pending_wheel_pos
        self._pending_wheel_factor = 1.0
        self._pending_wheel_pos = None
        if pos is None or factor == 1.0:
            return
        self._apply_wheel_zoom(factor, pos)

    def _apply_wheel_zoom(self, factor, anchor_pos):
        """Scale the view by ``factor``, keeping the scene point under
        ``anchor_pos`` fixed.

        Args:
            factor (float): multiplicative zoom, accumulated across a burst.
            anchor_pos (QPoint): viewport position to hold steady.
        """
        # Calculate new zoom level
        new_zoom = self.zoom_factor * factor
        
        # Prevent zooming out beyond minimum
        if new_zoom < self._min_zoom and factor < 1:
            new_zoom = self._min_zoom
            factor = new_zoom / self.zoom_factor
            
            # Apply zoom
            self.scale(factor, factor)
            self.zoom_factor = new_zoom
            
            # Center image when at minimum zoom
            self.centerOn(self.scene.sceneRect().center())
            self._emit_view_navigated()
            return
        
        # Store position before zoom for anchor-under-mouse
        old_pos = self.mapToScene(anchor_pos)

        # Apply zoom
        self.scale(factor, factor)
        self.zoom_factor = new_zoom

        # Correct position for natural zoom effect
        new_pos = self.mapToScene(anchor_pos)
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        
        # When zoomed to minimum, ensure perfect centering
        if abs(new_zoom - self._min_zoom) < 0.01:
            self.centerOn(self.scene.sceneRect().center())
        
        self._emit_view_navigated()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for rotation or panning."""
        if event.button() == Qt.RightButton and self.active_image:
            # Check for Ctrl + RightButton = rotation interaction
            if event.modifiers() == Qt.ControlModifier:
                # Initiate Rotation
                self._rotate_active = True
                self.setCursor(Qt.ClosedHandCursor)
                
                # Calculate the starting angle relative to the viewport center
                center = self.viewport().rect().center()
                dx = event.pos().x() - center.x()
                dy = event.pos().y() - center.y()
                # Store the baseline angle of the mouse and the current canvas rotation
                self._rotate_start_mouse_angle = np.degrees(np.arctan2(dy, dx))
                self._rotate_start_canvas_angle = self.rotation_angle
            else:
                # Initiate Native Pan
                self._pan_active = True
                self._pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for rotation, panning, and hover tracking."""
        self._mouse_move_event_impl(event)

    def _mouse_move_event_impl(self, event: QMouseEvent):
        # Handle rotation
        if self._rotate_active and self.active_image:
            self._enter_fast_paint_mode()
            center = self.viewport().rect().center()
            dx = event.pos().x() - center.x()
            dy = event.pos().y() - center.y()
            current_mouse_angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate how much the mouse has rotated since the click
            angle_delta = current_mouse_angle - self._rotate_start_mouse_angle
            new_canvas_angle = self._rotate_start_canvas_angle + angle_delta
            
            # Apply absolute rotation
            self._set_absolute_rotation(new_canvas_angle)
            
            # Emit navigation signal so context matrix syncs live
            self._emit_view_navigated()
        
        # Handle panning
        elif self._pan_active:
            if not self.active_image:
                self._pan_active = False
                return

            self._enter_fast_paint_mode()

            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            
            # Adjust scrollbars
            h_scroll = self.horizontalScrollBar()
            v_scroll = self.verticalScrollBar()
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
        
        # Emit hover signal with scene coordinates
        scene_pos = self.mapToScene(event.pos())
        # emit floats for higher precision (consumers may cast if needed)
        self.mouseHovered.emit(scene_pos.x(), scene_pos.y())
        
        super().mouseMoveEvent(event)

    def _pointer_over_self(self) -> bool:
        """True while the pointer is physically inside this view's frame."""
        try:
            return self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        except Exception:
            return False

    def on_pointer_left(self):
        """Drop hover-only graphics because the pointer left this canvas.

        Both the cursor preview and the dynamic marker are written by *other*
        widgets so nothing on the canvas itself used to take them down — 
        they stranded on every context tile once the pointer left the source widget.

        Subclasses extend this rather than leaveEvent so both entry points below
        stay covered. Must stay idempotent: it can run twice for one exit.
        """
        self.clear_cursor_preview()
        self.clear_dynamic_marker()

    def leaveEvent(self, event):
        """Primary exit path: Qt sends Leave to every widget the pointer left,
        ancestors of the viewport included."""
        self.on_pointer_left()
        super().leaveEvent(event)

    def viewportEvent(self, event):
        """Backstop for the viewport's own Leave.

        A Leave delivered only to the viewport child does NOT reach
        leaveEvent() — QAbstractScrollArea does not forward it to the view — so
        this catches any exit the ancestor-chain dispatch misses. Gated on the
        pointer genuinely being outside the frame, because moving onto a
        scrollbar also leaves the viewport while the view still holds the cursor.
        """
        if event.type() == QEvent.Leave and not self._pointer_over_self():
            self.on_pointer_left()
        return super().viewportEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events for rotation or panning."""
        if event.button() == Qt.RightButton:
            if self._rotate_active:
                self._rotate_active = False
            if self._pan_active:
                self._pan_active = False
            
            self.setCursor(Qt.ArrowCursor)
            self._emit_view_navigated()
        else:
            super().mouseReleaseEvent(event)
    
    def resizeEvent(self, event):
        """Handle resize events to maintain proper view fitting."""
        super().resizeEvent(event)
        
        # Fit view to image after resize
        if self.active_image and self.scene:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            # Sync zoom state after fitInView
            self._calculate_min_zoom()
            self.zoom_factor = get_view_scale(self.transform())
        
        # Keep placeholder geometry in sync
        try:
            if self._placeholder_label and self._placeholder_label.isVisible():
                self._placeholder_label.setGeometry(self.viewport().rect())
        except Exception:
            pass

        try:
            self._sync_perimeter_overlay_geometry()
        except Exception:
            pass

    # ==================== Placeholder Management ====================
    
    def _show_placeholder(self, text: str = None):
        """Show the centered placeholder label with optional custom text."""
        try:
            if text:
                self._placeholder_label.setText(text)
            self._placeholder_label.setGeometry(self.viewport().rect())
            self._placeholder_label.show()
        except Exception:
            pass
    
    def _hide_placeholder(self):
        """Hide the placeholder label."""
        try:
            self._placeholder_label.hide()
        except Exception:
            pass

    # ==================== Canvas Perimeter Overlay ====================

    def _sync_perimeter_overlay_geometry(self):
        """Keep the perimeter overlay aligned with the canvas widget."""
        if self._perimeter_overlay is None:
            return

        self._perimeter_overlay.setGeometry(self.rect())
        self._perimeter_overlay.raise_()

    def clear_perimeter_overlay(self):
        """Clear any perimeter border from the canvas."""
        if self._perimeter_overlay is None:
            return

        try:
            self._perimeter_overlay.hide()
            self._perimeter_overlay.setStyleSheet("background: transparent; border: none;")
        except Exception:
            pass

    def set_perimeter_overlay(self, color, width):
        """Draw a perimeter around the canvas using the given color and width."""
        self.clear_perimeter_overlay()

        border_color = QColor(color)
        border_width = max(0, int(round(width)))
        if not border_color.isValid() or border_width <= 0:
            return

        if self._perimeter_overlay is None:
            self._perimeter_overlay = QFrame(self)
            self._perimeter_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self._perimeter_overlay.setFrameShape(QFrame.NoFrame)
            self._perimeter_overlay.setAutoFillBackground(False)

        self._sync_perimeter_overlay_geometry()
        self._perimeter_overlay.setStyleSheet(
            f"background: transparent; border: {border_width}px solid {border_color.name()};"
        )
        self._perimeter_overlay.show()
        self._perimeter_overlay.raise_()
    
    # ==================== Scene Management ====================
    
    def clear_scene(self):
        """Clear the graphics scene and reset related variables."""
        # Stop any pending dynamic range update
        self._dynamic_range_timer.stop()
        self.clear_perimeter_overlay()
        
        # Clean up scene items
        if self.scene:
            for item in list(self.scene.items()):
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
                    if hasattr(item, 'deleteLater'):
                        item.deleteLater()
        
        # Clear and recreate scene
        if self.scene:
            self.scene.deleteLater()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Reset image references
        self._base_image_item = None
        self._image_dimensions = None

        # The scene-clearing loop above already removed the overlay items; reset
        # the overlay wrappers (keeps colormap/opacity so they persist across
        # image loads).
        self._z_overlay.clear()
        self._feature_overlay.clear()
        self._label_overlay.clear()

        # Clear read-only annotation overlay references
        self._readonly_annotation_items = {}

        # Clear Z-channel data
        self.z_data_raw = None
        self.z_index = None
        self.z_colormap_name = None
        self.z_data_min = None
        self.z_data_max = None
        self.z_data_shape = None
        self.z_nodata_mask = None

        # Allow subclasses to clean up their scene-dependent items
        self._on_scene_cleared()
        
        # Show placeholder
        self._show_placeholder("No image loaded")
    
    def _on_scene_cleared(self):
        """Hook for subclasses to perform cleanup after scene is cleared."""
        # Recreate marker slots on the new scene
        try:
            self._init_markers()
        except Exception:
            traceback.print_exc()
    
    # ==================== Image Loading ====================
    
    def load_visuals(self, q_image, image_path, raster=None, image_dimensions=None):
        """
        Load and display a QImage with optional Z-channel visualization.
        
        This is the canonical entry point for loading images into the canvas.
        
        Args:
            q_image (QImage): The full-resolution image to display
            image_path (str): Path identifier for the image
            raster (Raster, optional): Raster object with Z-channel data
            image_dimensions (tuple[int, int], optional): Logical image dimensions
                to preserve scene coordinates when q_image is downscaled.
        """
        # Clear previous state
        self.clear_scene()
        
        # Hide placeholder
        self._hide_placeholder()
        
        # The displayed image, as a QImage. Everything on the paint path wants a
        # QImage; the QPixmap below exists only for callers that still ask this
        # object for its dimensions, and costs a third full-resolution buffer
        # (4.3 GB on a 32k raster) to answer a question `_image_dimensions`
        # already answers.
        source_image = q_image if isinstance(q_image, QImage) else QPixmap(q_image).toImage()

        self.pixmap_image = None

        if image_dimensions is None:
            image_dimensions = (source_image.width(), source_image.height())

        try:
            image_width = max(1, int(image_dimensions[0]))
            image_height = max(1, int(image_dimensions[1]))
        except Exception:
            image_width = source_image.width()
            image_height = source_image.height()

        self._image_dimensions = (image_width, image_height)

        self._base_image_item = FastImageItem()
        img_to_pass = source_image
        self._base_image_item.set_image(img_to_pass, target_size=self._image_dimensions)
        self._base_image_item.setZValue(-10)
        self.scene.addItem(self._base_image_item)
        self.scene.setSceneRect(QRectF(0, 0, image_width, image_height))
        
        # Update state
        self.current_image_path = image_path
        self.active_image = True
        
        # Load Z-channel visualization if available
        if raster is not None:
            self._load_z_channel_visualization(raster)
        
        # Fit to view and calculate min zoom
        self.fit_to_image()
    
    def fit_to_image(self):
        """Fit the entire image in the view and recalculate min zoom."""
        if not self.active_image:
            return
        
        image_rect = self.get_image_rect()
        self.fitInView(image_rect, Qt.KeepAspectRatio)
        
        # Recalculate minimum zoom
        self._calculate_min_zoom()
        
        # Sync zoom_factor to the actual transform scale after fitInView
        self.zoom_factor = get_view_scale(self.transform())
        
        self._emit_view_navigated()
    
    def _calculate_min_zoom(self):
        """Calculate the minimum zoom factor needed to fit image in viewport."""
        if not self.scene or not self.active_image:
            self._min_zoom = 1.0
            return
        
        view_rect = self.viewport().rect()
        scene_rect = self.scene.sceneRect()
        
        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            self._min_zoom = 1.0
            return
        
        width_ratio = view_rect.width() / scene_rect.width()
        height_ratio = view_rect.height() / scene_rect.height()
        
        # ---> Allow the view to scale down as far as needed to fit huge images! <---
        self._min_zoom = max(min(width_ratio, height_ratio), 0.0001)
    
    # ==================== Viewport Control API ====================
    
    def center_on_pixel(self, x, y):
        """Center the view on the given image pixel coordinate."""
        # ---> Prevent anchor fighting during panning <---
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        
        self.centerOn(QPointF(x, y))
        
        self.setTransformationAnchor(old_anchor)
    
    def set_zoom_level(self, factor):
        """Set the absolute view transform scale, preserving current rotation."""
        if factor <= 0:
            return
        
        self._apply_transform(factor, self.rotation_angle)
    
    def snap_to_target(self, target_x, target_y, relative_zoom, angle_degrees=0.0):
        """
        Snap to a specific pixel location with a proportional zoom level and synchronized rotation.
        
        Args:
            target_x (float): Target pixel X coordinate
            target_y (float): Target pixel Y coordinate
            relative_zoom (float): Zoom ratio relative to fit-to-view
                                   (1.0 = fit whole image, 2.0 = 2x beyond fit, etc.)
            angle_degrees (float): Target rotation angle in degrees (default 0.0)
        """
        if not self.active_image:
            return
        
        absolute_zoom = self._min_zoom * relative_zoom
        
        # 1. Get the pivot point (image center in scene coordinates)
        image_rect = self.get_image_rect()
        pivot_x = image_rect.width() / 2.0
        pivot_y = image_rect.height() / 2.0
        
        # 2. Build the explicit transform: translate -> rotate -> scale -> translate
        transform = QTransform()
        transform.translate(pivot_x, pivot_y)
        transform.rotate(angle_degrees)
        transform.scale(absolute_zoom, absolute_zoom)
        transform.translate(-pivot_x, -pivot_y)
        
        # 3. Apply the transform directly
        self.setTransform(transform)
        self.zoom_factor = absolute_zoom
        self.rotation_angle = angle_degrees
        
        # 4. Snap directly to the target pixel (no pan-restoration needed here)
        self.center_on_pixel(target_x, target_y)
        self._emit_view_navigated()
    
    def _apply_transform(self, zoom, angle):
        """
        Builds and applies the transform matrix from scratch, pivoting on the image center.
        This mirrors the QtiViewWidget approach to prevent pendulum-swinging during rotation.
        
        The transform sequence is:
        1. Translate origin to image center
        2. Apply rotation around that center
        3. Apply zoom scaling
        4. Translate origin back
        
        Then restore the view so the pan position is preserved.
        """
        if not self.active_image:
            return

        # 1. Save where we are currently looking so we don't lose our pan position
        current_view_center = self.mapToScene(self.viewport().rect().center())
        
        # 2. Get the pivot point (the center of the image in scene coordinates)
        image_rect = self.get_image_rect()
        pivot_x = image_rect.width() / 2.0
        pivot_y = image_rect.height() / 2.0
        
        # 3. Build the explicit transform: translate -> rotate -> scale -> translate
        transform = QTransform()
        transform.translate(pivot_x, pivot_y)
        transform.rotate(angle)
        transform.scale(zoom, zoom)
        transform.translate(-pivot_x, -pivot_y)
        
        # 4. Apply the matrix
        self.setTransform(transform)
        
        # 5. Restore the pan position directly
        self.centerOn(current_view_center)
        
        # 6. Update state trackers
        self.zoom_factor = zoom
        self.rotation_angle = angle

    def _set_absolute_rotation(self, angle_degrees):
        """Apply absolute rotation, triggered by mouse movement."""
        self._apply_transform(self.zoom_factor, angle_degrees)
    
    # ==================== Helper Methods ====================
    
    def viewportToScene(self):
        """Convert viewport rectangle to scene coordinates."""
        # Use the QRect overload (returns a QPolygonF of all 4 mapped corners)
        # instead of mapping just topLeft/bottomRight: under a rotated view
        # transform those two screen corners no longer correspond to the
        # scene's min/max x/y, which can yield a QRectF with negative
        # width/height. boundingRect() gives the correct axis-aligned bounding
        # box of the visible region for any rotation angle.
        return self.mapToScene(self.viewport().rect()).boundingRect()
    
    def get_image_dimensions(self):
        """Get the dimensions of the currently loaded image."""
        if self._image_dimensions is not None:
            return self._image_dimensions
        return 0, 0
    
    def get_image_rect(self):
        """Get the bounding rectangle of the currently loaded image in scene coordinates."""
        if self._image_dimensions is not None:
            return QRectF(0, 0, self._image_dimensions[0], self._image_dimensions[1])
        return QRectF()
    
    def _emit_view_navigated(self):
        """Emit viewNavigated signal with current center and zoom."""
        if self.active_image:
            center = self.mapToScene(self.viewport().rect().center())
            self.viewNavigated.emit(center.x(), center.y(), self.zoom_factor)

    def _enter_fast_paint_mode(self):
        """Temporarily disable AA / smooth sampling for fluid interaction."""
        if not self._fast_hints_active:
            self._fast_hints_active = True
            self.setRenderHints(QPainter.TextAntialiasing)
        self._interaction_idle_timer.start()

    def _restore_quality_hints(self):
        self._fast_hints_active = False
        self.setRenderHints(self._quality_hints)
        self.viewport().update()

    # ==================== Read-Only Annotation Overlays (Phase 6) ====================
    
    def _render_annotations_readonly(self, annotations):
        """Render annotations as non-interactive overlays on this canvas.

        Paths are merged per unique (label_color, transparency, is_selected)
        group so that N annotations with L distinct labels produce only L scene
        items instead of N — dramatically faster for large annotation sets.

        Items are stored in ``_readonly_annotation_items`` keyed by that same
        tuple so incremental callers can update a single group without
        rebuilding the entire layer.

        Args:
            annotations (list): List of Annotation objects to display.
        """
        from coralnet_toolbox.Annotations import MaskAnnotation
        from collections import defaultdict

        self._clear_readonly_annotations()

        if not annotations:
            return

        # Group by (r, g, b, transparency, is_selected). is_selected separates
        # phantom-selected annotations so they are drawn with the selected-state
        # pen (delegates to the same create_pen() used by Annotation._create_pen,
        # eliminating the previous copy-paste).
        #
        # Annotation._add_ring normalizes ring winding at the source, so
        # opposite-wound subpaths do not cancel under WindingFill and the
        # "cutout" artifacts that toFillPolygon() was introduced to fix cannot
        # arise. Not calling toFillPolygon also drops the connector segments its
        # rewinding inserts — Qt documents that rewinding "inserts addition
        # lines in the polygon" — which this layer was stroking as a chord
        # across every polygon that had a hole.
        groups = defaultdict(list)
        group_styles = {}  # key -> (QColor, transparency, is_selected)

        for annotation in annotations:
            if isinstance(annotation, MaskAnnotation):
                continue
            key = phantom_group_key(annotation)
            groups[key].append(annotation)
            if key not in group_styles:
                group_styles[key] = (QColor(annotation.label.color),
                                     annotation.transparency,
                                     bool(annotation.is_selected))

        # One pitch for the whole layer, remembered so that an incremental
        # single-group rebuild files its annotations into the same cells.
        self._phantom_bucket_px = phantom_bucket_px(self.scene.sceneRect(),
                                                    len(annotations))

        # NoIndex while bulk-adding: one index rebuild at the end beats N
        # insertions.  update_readonly_group deliberately does NOT do this — for
        # a single addItem the toggle would force a full rebuild for nothing.
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        # Sorted, not in encounter order. All phantom groups share one Z value,
        # so this order *is* the paint order, and building it from whichever
        # annotation happened to be seen first made it depend on the selection:
        # selecting the leading run of one label moved that label's group to the
        # back of the stack, and deselecting left it there, permanently changing
        # the blend wherever two labels overlap. Sorting by the colour key is
        # just as arbitrary but does not move. It also draws a colour's
        # selected-state group after its unselected one, is_selected being the
        # key's last element, which is the order that layer wants anyway.
        for key in sorted(groups):
            group = groups[key]
            color, transparency, is_selected = group_styles[key]
            item = self._make_phantom_item(group, color, transparency, is_selected)
            if item is None:
                continue
            self.scene.addItem(item)
            # Store by key so individual groups can be patched incrementally.
            self._readonly_annotation_items[key] = item
        self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self._phantom_group_order = list(self._readonly_annotation_items)

        # Remember what this layer was built from, but do not index it yet:
        # most rebuilds (inference results, a transparency change, toggling a
        # label's visibility) are never followed by a click, and indexing 15k
        # annotations costs 14 ms. The first hit-test pays for it instead.
        self._phantom_hit_source = annotations
        self._phantom_hit_index = None

        self.viewport().update()

    def get_phantom_hit_index(self):
        """The click-resolution index for the current phantom layer.

        Built on first use after a rebuild and reused until some annotation's
        geometry changes. Both halves of that matter: building costs 14 ms on a
        15k-annotation image while a click saves 4 ms, so an index thrown away
        on every selection change would cost more than it earns. Selection does
        not move anything, so it does not invalidate; the geometry epoch does.

        Returns None when the layer has not been built, in which case callers
        fall back to scanning every annotation.
        """
        if self._phantom_hit_source is None:
            return None
        # Imported here, not at module scope: this module is imported by the
        # annotation classes, so a top-level import would close the cycle.
        from coralnet_toolbox.Annotations.QtAnnotation import geometry_epoch

        epoch = geometry_epoch()
        if self._phantom_hit_index is None or self._phantom_hit_epoch != epoch:
            self._phantom_hit_index = PhantomHitIndex(self._phantom_hit_source)
            self._phantom_hit_epoch = epoch
        return self._phantom_hit_index

    def _clear_readonly_annotations(self):
        """Remove all read-only annotation items from the scene."""
        for item in self._readonly_annotation_items.values():
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass
        self._readonly_annotation_items = {}
        self._phantom_group_order = []
        # Stale index is worse than none: it would silently misresolve clicks.
        self._phantom_hit_index = None
        self._phantom_hit_source = None

    def _make_phantom_item(self, annotations, color, transparency, is_selected):
        """Build the single scene item that draws one phantom colour group.

        Both the full rebuild and the incremental single-group rebuild go
        through here. They previously each built their own item and drifted
        apart — different item class, different fill-merge strategy — so that
        selecting an annotation silently swapped one colour group onto a
        differently-shaped rendering path that deselecting never undid.

        Args:
            annotations (list): the group's annotations.
            color (QColor): the label colour for the group.
            transparency (int): alpha applied to the fill.
            is_selected (bool): whether this group is the selected-state group.

        Returns:
            A configured PhantomGroupItem not yet added to the scene, or None
            when no annotation in the group had a drawable path.
        """
        from coralnet_toolbox.Annotations.QtAnnotation import create_pen

        fill_color = QColor(color)
        fill_color.setAlpha(transparency)

        if is_selected:
            # Delegate to the shared create_pen() — no more copy-paste constants.
            pen = create_pen(color, is_selected=True)
        else:
            # Phantom unselected: thinner 1 px pen is intentionally lighter
            # than the 2 px pen used by a fully-owned annotation item.
            pen = QPen(color, 1)
            pen.setCosmetic(True)

        buckets, bounds, overhang = bucket_annotation_paths(
            annotations, self._phantom_bucket_px)
        if buckets is None:
            return None

        item = PhantomGroupItem(buckets, self._phantom_bucket_px, bounds, overhang)
        item.setBrush(QBrush(fill_color))
        item.setPen(pen)
        # Explicitly uncached, and it must stay that way.
        #
        # This used to set DeviceCoordinateCache so that panning would blit a
        # pixmap instead of re-stroking every subpath. That is sound for an
        # item of bounded size and ruinous for this one, which spans the whole
        # image: the cache is sized by the item's device-space rect, so it
        # grows with the square of the zoom factor and Qt's cache bookkeeping
        # comes to dwarf the drawing it was meant to avoid. On a 1 MP image
        # with 100 polygons at 16x zoom that measured 1237 ms per frame with
        # one annotation on screen, against 0.3 ms uncached.
        #
        # Uncached is only cheap because paint() culls. Qt does not reject
        # off-screen sub-paths cheaply enough to rely on at this scale: before
        # the bucketing above, an uncached layer of 13k 50-vertex polygons
        # still cost 60 ms a frame at 32x zoom with five of them visible,
        # because every element was walked regardless. Culling is what makes
        # the cache unnecessary, not the rasterizer.
        item.setCacheMode(QGraphicsItem.NoCache)
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setFlag(QGraphicsItem.ItemIsMovable, False)
        item.setAcceptHoverEvents(False)
        item.setZValue(10)
        return item

    def update_readonly_group(self, key, annotations):
        """Rebuild ONE phantom group item in place.

        Args:
            key: group key from ``phantom_group_key``.
            annotations: the full current set of annotations belonging to that
                group (may be empty, which removes the item).

        Only valid once the layer has been built by _render_annotations_readonly;
        callers must fall back to a full refresh otherwise.
        """
        # Remove the existing item for this key
        old_item = self._readonly_annotation_items.pop(key, None)
        if old_item is not None:
            try:
                if old_item.scene() is not None:
                    old_item.scene().removeItem(old_item)
            except Exception:
                pass

        if not annotations:
            self.viewport().update()
            return

        r, g, b, transparency, is_selected = key
        item = self._make_phantom_item(annotations, QColor(r, g, b), transparency, is_selected)
        if item is None:
            self.viewport().update()
            return

        self.scene.addItem(item)
        self._readonly_annotation_items[key] = item
        self._restack_phantom_group(key, item)
        self.viewport().update()

    def _restack_phantom_group(self, key, item):
        """Put a rebuilt group back where it was in the layer's stacking order.

        All phantom groups share one Z value, so Qt orders them by insertion
        and scene.addItem() appends. Rebuilding one group on every selection
        change would therefore float that colour above the rest, visibly
        shifting the blend anywhere two labels overlap — 791 pixels on a 13k
        annotation image, changing nothing about what is drawn, only the order
        it is drawn in.

        Args:
            key: group key from ``phantom_group_key``.
            item: the freshly built item for that group.
        """
        order = self._phantom_group_order
        if key not in order:
            # A group that did not exist at the last full rebuild (a label
            # recoloured, say). The end of the stack is where it belongs.
            order.append(key)
            return

        for later_key in order[order.index(key) + 1:]:
            following = self._readonly_annotation_items.get(later_key)
            if following is not None:
                item.stackBefore(following)
                return

    def _highlight_readonly_annotation(self, annotation_id, highlighted):
        """Highlight or un-highlight a read-only annotation overlay.

        Args:
            annotation_id (str): The annotation UUID to highlight.
            highlighted (bool): Whether to highlight (True) or revert (False).
        """
        for item in self._readonly_annotation_items.values():
            if getattr(item, '_source_annotation_id', None) == annotation_id:
                # Restore original style from annotation data
                original_color = item.brush().color()
                original_color.setAlpha(255)
                pen = QPen(original_color, 1)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.setZValue(5)
                break
    
    # ==================== Z-Channel Visualization ====================
    
    @staticmethod
    def _normalize_to_index(data, vmin, vmax, nodata_mask):
        """Map ``data`` to a uint8 palette-index array.

        Index 0 is reserved for nodata (rendered transparent via the color
        table); valid values are scaled into 1..255. Building this one-byte
        array (instead of an expanded RGBA image) is what lets a colormap change
        be a cheap color-table swap.
        """
        index = np.ones(data.shape, dtype=np.uint8)
        if vmin is not None and vmax is not None and vmax > vmin:
            scaled = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
            index = (1 + scaled * 254).astype(np.uint8)
        index[nodata_mask] = 0
        return np.ascontiguousarray(index)

    def _load_z_channel_visualization(self, raster):
        """
        Load and initialize the Z-channel (depth) visualization.

        Args:
            raster: The Raster object containing Z-channel data
        """
        # Clean up any previous depth overlay.
        self._z_overlay.clear()

        # Check if Z-channel data is available
        if raster.z_channel_lazy is None:
            return

        try:
            z_data = raster.z_channel_lazy

            # Store raw Z-channel data
            self.z_data_raw = z_data.copy()
            self.z_data_shape = z_data.shape

            # Create mask for NaN and nodata values
            nodata_mask = np.isnan(z_data)
            if raster.z_nodata is not None:
                nodata_mask |= (z_data == raster.z_nodata)

            # Full-range min/max over valid data
            valid_data = z_data[~nodata_mask]
            if len(valid_data) > 0:
                self.z_data_min = np.min(valid_data)
                self.z_data_max = np.max(valid_data)
            else:
                self.z_data_min = 0.0
                self.z_data_max = 1.0

            # Build the palette-index array (full range) once.
            self.z_nodata_mask = nodata_mask
            self.z_index = self._normalize_to_index(
                z_data, self.z_data_min, self.z_data_max, nodata_mask
            )
            self.z_colormap_name = None

            # Hand the index field to the overlay (full-image rect), default
            # opacity, then hide until a colormap is selected via
            # update_overlay_colormap()/_update_z_colormap().
            self._z_overlay.set_indices(self.scene, self.z_index,
                                        target_rect=self.get_image_rect())
            self._z_overlay.set_opacity(0.5)
            self._z_overlay.hide()

        except Exception:
            traceback.print_exc()
            self._z_overlay.clear()

    # ---- Generic colormap-overlay dispatch (drives the active overlay) ----

    def set_active_colormap_overlay(self, which):
        """Point the colormap dropdown / opacity slider at an overlay.

        Args:
            which (str): 'z' for the depth overlay (default) or 'feature' for the
                feature-similarity overlay.
        """
        self._active_colormap_overlay = (
            self._feature_overlay if which == 'feature' else self._z_overlay
        )

    def update_overlay_colormap(self, colormap_name):
        """Apply a colormap to the active overlay ('None' hides it)."""
        if self._active_colormap_overlay is self._z_overlay:
            self._update_z_colormap(colormap_name)
            return
        # Feature overlay: a colormap change is a pure table swap. Reserve the
        # top index as a below-threshold scrim so the thresholded preview reads
        # as dimmed rather than fully exposed.
        try:
            if colormap_name == 'None':
                self._feature_overlay.hide()
                return
            self._feature_overlay.set_colormap(colormap_name, scrim_rgba=(0, 0, 0, 200))
            self._feature_overlay.show()
        except Exception:
            traceback.print_exc()

    def set_overlay_opacity(self, opacity):
        """Set the opacity of the active colormap overlay."""
        self._active_colormap_overlay.set_opacity(opacity)

    # ---- Z-channel (depth) specifics ----

    def _update_z_colormap(self, colormap_name):
        """Apply a colormap to the depth overlay ('None' hides it)."""
        if self.z_index is None or self._z_overlay.item is None:
            return

        try:
            self.z_colormap_name = colormap_name
            if colormap_name == 'None':
                self._z_overlay.hide()
                return

            # A colormap change is just a palette swap on the existing indices.
            self._z_overlay.set_colormap(colormap_name)

            # Dynamic scaling re-derives the indices for the visible range; for the
            # full range the cached indices blit straight through the new table.
            if self.dynamic_z_scaling:
                self.update_dynamic_range()
            self._z_overlay.show()
        except Exception:
            traceback.print_exc()

    def toggle_dynamic_z_scaling(self, enabled):
        """
        Toggle dynamic Z-range scaling based on visible area.

        Args:
            enabled (bool): Whether to enable dynamic scaling
        """
        self.dynamic_z_scaling = enabled

        if self._z_overlay.item is None:
            return
        if enabled:
            self.update_dynamic_range()
        else:
            self._reset_z_channel_to_full_range(None)

    def schedule_dynamic_range_update(self):
        """Schedule a debounced dynamic range update."""
        if not self.dynamic_z_scaling:
            return

        self._dynamic_range_timer.stop()
        self._dynamic_range_timer.start(self.dynamic_range_update_delay)

    def update_dynamic_range(self):
        """Update Z-channel visualization for visible viewport range."""
        if not self.dynamic_z_scaling or self._z_overlay.item is None or self.z_data_raw is None:
            return

        try:
            # Get visible area in scene coordinates
            visible_rect = self.viewportToScene()

            # Clamp to image bounds
            image_rect = self.get_image_rect()
            visible_rect = visible_rect.intersected(image_rect)

            if visible_rect.isEmpty():
                return

            # Get pixel coordinates
            x1, y1 = int(visible_rect.left()), int(visible_rect.top())
            x2, y2 = int(visible_rect.right()), int(visible_rect.bottom())

            # Clamp to array bounds
            h, w = self.z_data_shape
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))

            # Calculate dynamic range from visible data
            visible_data = self.z_data_raw[y1:y2, x1:x2]
            valid_mask = ~np.isnan(visible_data)

            if np.any(valid_mask):
                dynamic_min = np.min(visible_data[valid_mask])
                dynamic_max = np.max(visible_data[valid_mask])
            else:
                dynamic_min = self.z_data_min
                dynamic_max = self.z_data_max

            # Re-derive palette indices for the visible (dynamic) range. The
            # overlay's color table is left untouched, so the user's selected
            # colormap is preserved.
            nodata_mask = self.z_nodata_mask
            if nodata_mask is None:
                nodata_mask = np.isnan(self.z_data_raw)
            self.z_index = self._normalize_to_index(
                self.z_data_raw, dynamic_min, dynamic_max, nodata_mask
            )
            self._z_overlay.set_indices(self.scene, self.z_index,
                                        target_rect=self.get_image_rect())

        except Exception:
            traceback.print_exc()

    def _reset_z_channel_to_full_range(self, colormap_name):
        """
        Reset Z-channel visualization to full data range.

        Args:
            colormap_name (str, optional): Colormap to apply. If None, uses 'Plasma'.
        """
        if colormap_name is None:
            colormap_name = 'Plasma'

        if self._z_overlay.item is None or self.z_data_raw is None:
            return

        try:
            if colormap_name != 'None':
                # Rebuild full-range indices and swap to the requested palette.
                nodata_mask = self.z_nodata_mask
                if nodata_mask is None:
                    nodata_mask = np.isnan(self.z_data_raw)
                self.z_index = self._normalize_to_index(
                    self.z_data_raw, self.z_data_min, self.z_data_max, nodata_mask
                )
                self.z_colormap_name = colormap_name
                self._z_overlay.set_colormap(colormap_name)
                self._z_overlay.set_indices(self.scene, self.z_index,
                                            target_rect=self.get_image_rect())
        except Exception:
            traceback.print_exc()

    def clear_z_channel_visualization(self, image_path):
        """
        Clear Z-channel visualization for a specific image.

        Args:
            image_path (str): Path of the image with removed Z-channel
        """
        if image_path != self.current_image_path:
            return

        self._z_overlay.clear()

        # Clear cached data
        self.z_data_raw = None
        self.z_index = None
        self.z_colormap_name = None
        self.z_data_min = None
        self.z_data_max = None
        self.z_data_shape = None
        self.z_nodata_mask = None

    # ==================== Marker Slots (Phase 4) ====================

    def update_static_marker(self, x, y, color=None):
        """Update the static focal point marker at image pixel (x, y).
        
        Args:
            x, y: Pixel coordinates in image space.
            color: QColor for the marker. Default: QColor(0, 255, 0) (green).
        """
        if self._static_marker is None:
            return
        try:
            # Bounds check
            image_w, image_h = self.get_image_dimensions()
            if image_w and not (0 <= x < image_w and 0 <= y < image_h):
                self._static_marker.hide()
                return

            self._static_marker.setPos(x, y)
            color = color or QColor(0, 255, 0)
            pen = QPen(color, 2)
            for child in self._static_marker.childItems():
                try:
                    child.setPen(pen)
                except Exception:
                    pass
            self._static_marker.show()
        except Exception:
            traceback.print_exc()

    def clear_static_marker(self):
        """Clear the static focal point marker."""
        if self._static_marker is not None:
            try:
                self._static_marker.hide()
            except Exception:
                pass

    def update_dynamic_marker(self, x, y, color=None, is_valid=True):
        """Update the dynamic hover marker at image pixel (x, y).
        
        Args:
            x, y: Pixel coordinates in image space.
            color: QColor for the marker. Default: keeps current color.
            is_valid: If False, use dashed pen (occluded/estimated).
        """
        if self._dynamic_marker is None:
            return
        try:
            # Bounds check
            image_w, image_h = self.get_image_dimensions()
            if image_w and not (0 <= x < image_w and 0 <= y < image_h):
                self._dynamic_marker.hide()
                return

            self._dynamic_marker.setPos(x, y)
            pen = QPen(color or QColor(0, 255, 0), 2)
            if not is_valid:
                pen.setStyle(Qt.DashLine)
            else:
                pen.setStyle(Qt.SolidLine)
            self._dynamic_marker.setPen(pen)
            self._dynamic_marker.show()
        except Exception:
            traceback.print_exc()

    def clear_dynamic_marker(self):
        """Clear the dynamic hover marker."""
        if self._dynamic_marker is not None:
            try:
                self._dynamic_marker.hide()
            except Exception:
                pass

    # ==================== Marker Initialization ====================
    def _init_markers(self):
        """Create marker graphics items (hidden by default) and add to the scene."""
        try:
            # remove existing if leftover
            if self._static_marker is not None:
                try:
                    if self._static_marker.scene() == self.scene:
                        try:
                            self.scene.removeItem(self._static_marker)
                        except Exception:
                            pass
                except RuntimeError:
                    pass
                self._static_marker = None
                
            if self._dynamic_marker is not None:
                try:
                    if self._dynamic_marker.scene() == self.scene:
                        try:
                            self.scene.removeItem(self._dynamic_marker)
                        except Exception:
                            pass
                except RuntimeError:
                    pass
                self._dynamic_marker = None

            # Static crosshair group
            self._static_marker = QGraphicsItemGroup()
            pen = QPen(QColor(255, 64, 64))
            pen.setWidth(2)
            lh = QGraphicsLineItem(-12, 0, 12, 0)
            lv = QGraphicsLineItem(0, -12, 0, 12)
            el = QGraphicsEllipseItem(-6, -6, 12, 12)
            for it in (lh, lv, el):
                it.setPen(pen)
                it.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                self._static_marker.addToGroup(it)
            self._static_marker.setZValue(100)
            self.scene.addItem(self._static_marker)
            self._static_marker.hide()

            # Dynamic hover circle
            self._dynamic_marker = QGraphicsEllipseItem(-5, -5, 10, 10)
            self._dynamic_marker.setPen(QPen(QColor(255, 200, 0)))
            self._dynamic_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self._dynamic_marker.setZValue(100)
            self.scene.addItem(self._dynamic_marker)
            self._dynamic_marker.hide()

            # Tool cursor preview rect (created lazily; just reset the reference here)
            self._cursor_preview_item = None
            # Mask overlay item (created lazily; reset reference on scene rebuild)
            self._mask_overlay_item = None
            # The Z and feature ColorMapOverlays persist across scene rebuilds;
            # clear_scene() resets their item references, so nothing to do here.
        except Exception:
            traceback.print_exc()

    # ==================== Read-Only Annotations (Phase 6) ====================

    def render_readonly_annotations(self, annotation_data_list):
        """Render read-only annotations as live vector overlays."""
        if not self.active_image:
            if isinstance(self._base_image_item, FastImageItem):
                self._base_image_item.set_readonly_annotations([])
            self._clear_readonly_annotations()
            return

        annotations = annotation_data_list or []
        if isinstance(self._base_image_item, FastImageItem):
            self._base_image_item.set_readonly_annotations([])
        self._clear_readonly_annotations()
        self._render_annotations_readonly(annotations)

    # ==================== Cursor Preview (Tool Propagation) ====================

    def update_cursor_preview(self, u: float, v: float, item_factory):
        """Show a tool cursor preview at image pixel (u, v).

        The item is produced by item_factory(u, v) so each tool can supply its
        own style (patch square, brush circle, etc.).

        Args:
            u, v: Centre pixel coordinates in image space.
            item_factory: callable(u, v) -> QGraphicsItem
        """
        # Remove the previous preview item before creating a new one
        self.clear_cursor_preview()

        try:
            item = item_factory(u, v)
            item.setZValue(101)  # Above markers
            item.setFlag(QGraphicsItem.ItemIsSelectable, False)
            item.setFlag(QGraphicsItem.ItemIsMovable, False)
            item.setAcceptHoverEvents(False)
            self.scene.addItem(item)
            self._cursor_preview_item = item
        except Exception:
            pass

    def clear_cursor_preview(self):
        """Remove the tool cursor preview item from the scene."""
        if self._cursor_preview_item is not None:
            try:
                if self._cursor_preview_item.scene() is not None:
                    self._cursor_preview_item.scene().removeItem(self._cursor_preview_item)
            except Exception:
                pass
            self._cursor_preview_item = None

    # ==================== Mask Overlay (Brush Propagation) ====================

    def set_mask_overlay(self, mask_annotation):
        """Display or refresh a MaskAnnotation as a read-only overlay on this canvas.

        Creates a lightweight MaskGraphicsItem that paints directly from
        mask_annotation.qimage (kept up-to-date by update_mask) so the view stays
        in sync with every brush stroke without rebuilding the full pixmap.
        Safe to call repeatedly — reuses the existing item unless the annotation changes.
        """
        from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskGraphicsItem

        item = self._mask_overlay_item
        needs_new = (
            item is None
            or item.scene() is None
            or item.mask_annotation is not mask_annotation
        )
        if needs_new:
            self.clear_mask_overlay()
            item = MaskGraphicsItem(mask_annotation)
            item.setZValue(2)  # Above base image, below markers
            item.setAcceptHoverEvents(False)
            self.scene.addItem(item)
            self._mask_overlay_item = item

        item.update()

    def clear_mask_overlay(self):
        """Remove the mask overlay item from the scene."""
        if self._mask_overlay_item is not None:
            try:
                if self._mask_overlay_item.scene() is not None:
                    self._mask_overlay_item.scene().removeItem(self._mask_overlay_item)
            except Exception:
                pass
            self._mask_overlay_item = None

    # ==================== Feature Similarity Overlay (FeatureSelectTool) ====================

    def set_feature_overlay(self, index_array, rect=None):
        """Display a feature-similarity field as an indexed colormap overlay.

        ``index_array`` is a ``uint8 [gh, gw]`` palette-index field at
        feature-grid / preview resolution:
            0       -> transparent (invalid / off)
            1..254  -> colormap ramp (normalized similarity)
            255     -> below-threshold scrim
        It is scaled with smooth interpolation to fill ``rect`` (scene coords;
        None = whole image). The overlay's colormap and opacity are owned by the
        active colormap controls (see ``set_active_colormap_overlay``), so a
        colormap or opacity change is a cheap table swap / ``setOpacity`` with no
        recompute here. The item sits above the base image (-10) and Z-channel
        (-5) but below annotations (>= 0).

        Scaling/positioning the grid to ``rect`` uses the SAME proportional
        mapping a caller should use to map a pixel back to a grid cell, so the
        cursor and the lit region stay aligned regardless of aspect ratio.
        """
        if index_array is None:
            self.clear_feature_overlay()
            return

        try:
            arr = np.ascontiguousarray(index_array.astype(np.uint8, copy=False))
            gh, gw = arr.shape[:2]
            if gh <= 0 or gw <= 0:
                self.clear_feature_overlay()
                return

            if rect is None:
                image_w, image_h = self.get_image_dimensions()
                rect = QRectF(0, 0, float(image_w or gw), float(image_h or gh))

            # Ensure a colormap table exists (Plasma + scrim) before the first
            # blit, in case the tool hasn't pushed one through the dropdown yet.
            if self._feature_overlay._color_table is None:
                self._feature_overlay.set_colormap('Plasma', scrim_rgba=(0, 0, 0, 200))

            self._feature_overlay.set_indices(self.scene, arr, target_rect=rect)
        except Exception:
            traceback.print_exc()

    def clear_feature_overlay(self):
        """Remove the feature similarity overlay from the scene."""
        self._feature_overlay.clear()

    def set_label_overlay(self, index_array, colors, rect=None, alpha=160):
        """Display a multi-class label field as an indexed color overlay.

        ``index_array`` is ``uint8 [gh, gw]``: 0 = unlabeled (transparent),
        ``k + 1`` = class k. ``colors[k]`` is the ``(r, g, b)`` for class k. The
        grid is scaled to fill ``rect`` (scene coords; None = whole image) using
        the SAME proportional mapping as the feature overlay, so the cursor and
        the painted regions stay aligned. Reuses the indexed-8 ColorMapOverlay;
        only the color table differs (label colors instead of a colormap ramp).
        """
        if index_array is None:
            self.clear_label_overlay()
            return
        try:
            arr = np.ascontiguousarray(index_array.astype(np.uint8, copy=False))
            gh, gw = arr.shape[:2]
            if gh <= 0 or gw <= 0:
                self.clear_label_overlay()
                return

            if rect is None:
                image_w, image_h = self.get_image_dimensions()
                rect = QRectF(0, 0, float(image_w or gw), float(image_h or gh))

            table = [0] * 256  # index 0 -> fully transparent (unlabeled)
            for k, rgb in enumerate(colors):
                if k + 1 > 255:
                    break
                r, g, b = (int(v) for v in rgb)
                table[k + 1] = qRgba(r, g, b, int(alpha))

            self._label_overlay._color_table = table
            self._label_overlay.set_opacity(1.0)
            self._label_overlay.set_indices(self.scene, arr, target_rect=rect)
        except Exception:
            traceback.print_exc()

    def clear_label_overlay(self):
        """Remove the multi-class label overlay from the scene."""
        self._label_overlay.clear()
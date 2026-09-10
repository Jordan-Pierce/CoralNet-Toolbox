from collections import OrderedDict

from PyQt5 import QtCore, QtGui, QtWidgets

from coralnet_toolbox import theme as app_theme

ATTENTION_COLOR = app_theme.ATTENTION_COLOR


class AnnotationListModel(QtCore.QAbstractListModel):
    DataItemRole = QtCore.Qt.UserRole + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flat_items = []
        self._id_to_row = {}
        # group expanded state map
        self._group_expanded = {}
        # keep the last grouped input so we can rebuild on toggle
        self._grouped_items = []

    def set_grouped_items(self, grouped_items):
        """
        grouped_items: list of tuples (group_key, group_color, [AnnotationDataItem,...])
        """
        self.beginResetModel()
        self._grouped_items = list(grouped_items)
        flat = []
        id_to_row = {}
        row = 0
        for group_key, group_color, items in grouped_items:
            if group_key:
                expanded = self._group_expanded.get(group_key, True)
                flat.append({
                    "type": "header",
                    "key": group_key,
                    # The count matters most when a group is collapsed and
                    # its members are out of sight
                    "text": f"{group_key}  ·  {len(items):,}",
                    "color": group_color,
                    "expanded": expanded,
                })
                row += 1
            if (not group_key) or self._group_expanded.get(group_key, True):
                for it in items:
                    flat.append({"type": "annotation", "item": it})
                    id_to_row[it.annotation.id] = row
                    row += 1

        self._flat_items = flat
        self._id_to_row = id_to_row
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._flat_items)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == self.DataItemRole:
            return self._flat_items[index.row()]
        if role == QtCore.Qt.ToolTipRole:
            entry = self._flat_items[index.row()]
            if entry.get("type") == "annotation":
                return entry["item"].get_tooltip_text()
            return "Click to collapse or expand, Ctrl+click to select the whole group"
        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        item = self._flat_items[index.row()]
        if item.get("type") == "header":
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def toggle_group(self, group_key):
        # flip expanded state
        self._group_expanded[group_key] = not self._group_expanded.get(group_key, True)
        # Rebuild flat items from stored grouped input
        grouped_items = getattr(self, '_grouped_items', [])
        self.set_grouped_items(grouped_items)
        # notify view
        self.layoutChanged.emit()


class AnnotationItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, item_size=96, header_height=32, parent=None):
        super().__init__(parent)
        self.item_size = item_size
        self.header_height = header_height
        self._pixmap_cache = OrderedDict()
        self._cache_max = 256

    def sizeHint(self, option, index):
        data = index.data(AnnotationListModel.DataItemRole)
        if not data:
            return QtCore.QSize(self.item_size, self.item_size)
        if data.get("type") == "header":
            # Try to span the full view width for header rows so they appear
            try:
                if option and getattr(option, 'widget', None):
                    w = option.widget
                    try:
                        width = w.viewport().width()
                    except Exception:
                        width = w.width()
                else:
                    width = max(400, self.item_size * 6)
            except Exception:
                width = max(400, self.item_size * 6)
            return QtCore.QSize(width, self.header_height)
        item = data.get("item")
        aspect = getattr(item, 'aspect_ratio', 1.0)
        width = max(10, int(self.item_size * aspect))
        return QtCore.QSize(width, self.item_size)

    def paint(self, painter, option, index):
        data = index.data(AnnotationListModel.DataItemRole)
        if not data:
            return

        rect = option.rect
        if data.get("type") == "header":
            color = data.get("color")
            bg = (
                QtGui.QColor('#333333')
                if color is None
                else (QtGui.QColor(color) if not isinstance(color, QtGui.QColor) else color)
            )
            painter.fillRect(rect, bg)

            # Determine readable text color based on luminance
            try:
                r, g, b = bg.red(), bg.green(), bg.blue()
                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                text_color = QtGui.QColor('#000000') if luminance > 0.5 else QtGui.QColor('#ffffff')
            except Exception:
                text_color = QtGui.QColor('#ffffff')

            # Draw header text and chevron
            painter.setPen(QtGui.QPen(text_color))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            text = data.get('text', '')
            painter.drawText(rect.adjusted(12, 0, 0, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)

            # chevron on right
            expanded = data.get('expanded', True)
            chev = '▾' if expanded else '▸'
            chev_rect = rect.adjusted(-28, 0, -8, 0)
            painter.drawText(chev_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, chev)
            return

        # annotation
        item = data.get('item')
        ann = item.annotation
        try:
            color = QtGui.QColor(item.effective_color)
            label = item.effective_label.short_label_code
        except Exception:
            color, label = QtGui.QColor(app_theme.TEXT_SECONDARY_COLOR), None

        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.State_MouseOver)
        size = self.item_size
        tile = QtCore.QRectF(rect).adjusted(1, 1, -1, -1)
        radius = max(3.0, min(8.0, size * 0.06))

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # --- The crop, filling the tile and clipped to its rounded corners ---
        outline = QtGui.QPainterPath()
        outline.addRoundedRect(tile, radius, radius)
        painter.setClipPath(outline)
        painter.fillRect(tile, app_theme.SURFACE_COLOR)
        pix = self._tile_pixmap(ann, rect.size(), painter.device().devicePixelRatioF())
        if pix is not None:
            painter.drawPixmap(rect.topLeft(), pix)
        if hovered and not selected:
            painter.fillRect(tile, QtGui.QColor(255, 255, 255, 28))
        painter.setClipping(False)

        # --- Frame: a ring in the label colour when selected, a hairline otherwise ---
        if selected:
            ring = QtGui.QPen(color)
            ring.setWidthF(max(2.0, min(3.5, size * 0.03)))
        else:
            ring = QtGui.QPen(QtGui.QColor(app_theme.ACCENT_HOVER_COLOR if hovered else app_theme.SURFACE_BORDER_COLOR))
            ring.setWidthF(1.0)
        inset = ring.widthF() / 2
        painter.setPen(ring)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(tile.adjusted(inset, inset, -inset, -inset), radius, radius)

        # --- Badges, dropped as the tile shrinks so they never bury the crop ---
        roomy = size >= 56
        if label:
            if roomy:
                self._paint_label_pill(painter, tile, label, color, item.has_preview_changes())
            else:
                # Too small for text: the label colour along the bottom edge
                painter.save()
                painter.setClipPath(outline)
                painter.fillRect(QtCore.QRectF(tile.left(), tile.bottom() - 3, tile.width(), 3), color)
                painter.restore()
        if not ann.verified:
            self._paint_unverified_badge(painter, tile, item, roomy)
        if selected and size >= 48:
            self._paint_check_badge(painter, tile, color)

        painter.restore()

    # --- Tile helpers ---

    @staticmethod
    def _text_color_on(color):
        """Black or white, whichever reads better on the given colour."""
        luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
        return QtGui.QColor('#000000') if luminance > 0.5 else QtGui.QColor('#ffffff')

    def _badge_font(self, painter):
        """A small bold font that scales with the tile, within readable limits."""
        font = QtGui.QFont(painter.font())
        font.setBold(True)
        font.setPointSizeF(max(7.0, min(10.0, self.item_size * 0.085)))
        return font

    def _tile_pixmap(self, ann, size, device_ratio):
        """The annotation's crop, scaled to cover the tile and centre-cropped to it.

        Scaled once per size and screen density and cached. Drawing the
        covering pixmap straight into the tile rect instead would stretch it,
        and scaling at logical size blurs it on a high-DPI screen.
        """
        key = (ann.id, size.width(), size.height(), device_ratio)
        pix = self._pixmap_cache.get(key)
        if pix is not None:
            self._pixmap_cache.move_to_end(key)
            return pix

        try:
            source = None
            if hasattr(ann, 'get_cropped_image_graphic'):
                source = ann.get_cropped_image_graphic()
            elif hasattr(ann, 'get_cropped_image'):
                source = ann.get_cropped_image()
            if source is None or source.isNull():
                return None
        except Exception:
            return None

        target = QtCore.QSize(max(1, round(size.width() * device_ratio)), max(1, round(size.height() * device_ratio)))
        scaled = source.scaled(target, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        pix = scaled.copy(
            (scaled.width() - target.width()) // 2,
            (scaled.height() - target.height()) // 2,
            target.width(),
            target.height(),
        )
        pix.setDevicePixelRatio(device_ratio)

        self._pixmap_cache[key] = pix
        while len(self._pixmap_cache) > self._cache_max:
            self._pixmap_cache.popitem(last=False)
        return pix

    def _paint_label_pill(self, painter, tile, label, color, pending):
        """The label as a compact pill in the bottom-left corner.

        A pending relabel -- previewed in the Explorer but not yet applied --
        gets a dashed outline, so what has and has not been committed is
        visible across a whole page of tiles at once.
        """
        font = self._badge_font(painter)
        font.setItalic(pending)
        metrics = QtGui.QFontMetrics(font)
        margin = max(3.0, self.item_size * 0.05)
        pad = max(4.0, self.item_size * 0.05)
        height = metrics.height() + 2
        max_width = tile.width() - 2 * margin
        text = metrics.elidedText(str(label), QtCore.Qt.ElideRight, int(max_width - 2 * pad))
        width = min(max_width, metrics.horizontalAdvance(text) + 2 * pad)
        pill = QtCore.QRectF(tile.left() + margin, tile.bottom() - margin - height, width, height)

        fill = QtGui.QColor(color)
        fill.setAlpha(235)
        if pending:
            pen = QtGui.QPen(self._text_color_on(color))
            pen.setStyle(QtCore.Qt.DashLine)
            pen.setWidthF(1.2)
            painter.setPen(pen)
        else:
            painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(pill, height / 2, height / 2)

        painter.setFont(font)
        painter.setPen(self._text_color_on(color))
        painter.drawText(pill, QtCore.Qt.AlignCenter, text)

    def _paint_unverified_badge(self, painter, tile, item, roomy):
        """Flag an unverified annotation in the top-left corner, with its confidence when there is room.

        The same amber the Confidence window uses for "unverified", so a
        prediction reads as one in both places.
        """
        margin = max(3.0, self.item_size * 0.05)
        confidence = item.get_confidence_value() if item.annotation.machine_confidence else None

        if roomy and confidence is not None:
            font = self._badge_font(painter)
            font.setBold(False)
            metrics = QtGui.QFontMetrics(font)
            text = f"{confidence * 100:.0f}%"
            height = metrics.height() + 2
            width = metrics.horizontalAdvance(text) + 2 * max(4.0, self.item_size * 0.05)
            badge = QtCore.QRectF(tile.left() + margin, tile.top() + margin, width, height)
            pen = QtGui.QPen(ATTENTION_COLOR)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 170))
            painter.drawRoundedRect(badge, height / 2, height / 2)
            painter.setFont(font)
            painter.setPen(ATTENTION_COLOR)
            painter.drawText(badge, QtCore.Qt.AlignCenter, text)
        else:
            dot = max(5.0, min(9.0, self.item_size * 0.08))
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 200))
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.setBrush(ATTENTION_COLOR)
            painter.drawEllipse(QtCore.QRectF(tile.left() + margin, tile.top() + margin, dot, dot))

    def _paint_check_badge(self, painter, tile, color):
        """A tick in the top-right corner, so a selection reads even where the ring is thin."""
        diameter = max(12.0, min(20.0, self.item_size * 0.17))
        margin = max(3.0, self.item_size * 0.05)
        circle = QtCore.QRectF(tile.right() - margin - diameter, tile.top() + margin, diameter, diameter)

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 160))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(color)
        painter.drawEllipse(circle)

        tick = QtGui.QPainterPath()
        tick.moveTo(circle.left() + diameter * 0.27, circle.top() + diameter * 0.52)
        tick.lineTo(circle.left() + diameter * 0.44, circle.top() + diameter * 0.68)
        tick.lineTo(circle.left() + diameter * 0.74, circle.top() + diameter * 0.34)
        pen = QtGui.QPen(self._text_color_on(color))
        pen.setWidthF(max(1.5, diameter * 0.12))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(tick)

    def editorEvent(self, event, model, option, index):
        # Toggle group on header click
        # Left-click on header toggles group; Ctrl+left-click selects all in group
        try:
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                data = index.data(AnnotationListModel.DataItemRole)
                if data and data.get('type') == 'header':
                    group_key = data.get('key')
                    modifiers = QtWidgets.QApplication.keyboardModifiers()
                    if modifiers & QtCore.Qt.ControlModifier:
                        self._select_group_annotations(model, group_key)
                        return True
                    model._group_expanded[group_key] = not model._group_expanded.get(group_key, True)
                    # rebuild using stored grouped items
                    grouped_items = getattr(model, '_grouped_items', [])
                    model.set_grouped_items(grouped_items)
                    model.layoutChanged.emit()
                    return True

            # Ctrl + Right-click on an annotation: navigate to AnnotationWindow
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.RightButton:
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers & QtCore.Qt.ControlModifier:
                    data = index.data(AnnotationListModel.DataItemRole)
                    if data and data.get('type') == 'annotation':
                        ann = data['item'].annotation
                        # model.parent() should be the AnnotationViewerWindow instance
                        viewer = getattr(model, 'parent', lambda: None)()
                        if viewer is None:
                            try:
                                viewer = model.parent()
                            except Exception:
                                viewer = None
                        try:
                            if viewer and hasattr(viewer, 'main_window') and hasattr(viewer, 'annotation_window'):
                                # Navigate only. This gesture answers "where is this
                                # one?", so the gallery / embedding selection the user
                                # has built up is deliberately left untouched.
                                manager = getattr(viewer.main_window, 'selection_manager', None)
                                if manager is not None and hasattr(manager, 'navigate_to_annotation'):
                                    manager.navigate_to_annotation(ann.id)
                                    return True

                                # Fallback for when no SelectionManager is wired up.
                                try:
                                    if viewer.annotation_window.current_image_path != ann.image_path:
                                        viewer.annotation_window.set_image(ann.image_path)
                                except Exception:
                                    pass
                                try:
                                    viewer.annotation_window.select_annotation(ann, quiet_mode=True)
                                except Exception:
                                    pass
                                try:
                                    zoom = getattr(viewer.annotation_window, 'center_and_zoom_on_annotation', None)
                                    if zoom is None:
                                        zoom = getattr(viewer.annotation_window, 'center_on_annotation', None)
                                    if zoom is not None:
                                        zoom(ann)
                                except Exception:
                                    pass
                                try:
                                    if hasattr(viewer.main_window, 'confidence_window'):
                                        viewer.main_window.confidence_window.display_cropped_image(ann)
                                except Exception:
                                    pass
                                return True
                        except Exception:
                            pass
        except Exception:
            pass
        return super().editorEvent(event, model, option, index)

    def _select_group_annotations(self, model, group_key):
        """Select all annotations belonging to the given group."""
        viewer = getattr(model, 'parent', lambda: None)()
        if viewer is None:
            return
        ids = []
        for gk, _gc, items in getattr(model, '_grouped_items', []):
            if gk == group_key:
                ids.extend(it.annotation.id for it in items)
                break
        if ids and hasattr(viewer, 'render_selection_from_ids'):
            viewer.render_selection_from_ids(set(ids))
            if hasattr(viewer, 'selection_changed'):
                viewer.selection_changed.emit(ids)

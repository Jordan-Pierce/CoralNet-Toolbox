from collections import OrderedDict

from PyQt5 import QtCore, QtGui, QtWidgets

from coralnet_toolbox.Explorer.core.QtDataItem import AnnotationDataItem


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
                    "text": group_key,
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
            bg = QtGui.QColor('#333333') if color is None else (QtGui.QColor(color) if not isinstance(color, QtGui.QColor) else color)
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

        key = (ann.id, self.item_size)
        pix = self._pixmap_cache.get(key)
        if pix is None:
            try:
                # Prefer the annotation API to get a QPixmap suitable for GUI drawing
                source_pixmap = None
                if hasattr(ann, 'get_cropped_image_graphic'):
                    source_pixmap = ann.get_cropped_image_graphic()
                elif hasattr(ann, 'get_cropped_image'):
                    source_pixmap = ann.get_cropped_image()

                if source_pixmap and not source_pixmap.isNull():
                    pix = source_pixmap.scaled(rect.size(), QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
                else:
                    pix = QtGui.QPixmap(rect.size())
                    pix.fill(QtGui.QColor('#222'))
            except Exception:
                pix = QtGui.QPixmap(rect.size())
                pix.fill(QtGui.QColor('#444'))
            self._pixmap_cache[key] = pix
            # trim cache
            while len(self._pixmap_cache) > self._cache_max:
                self._pixmap_cache.popitem(last=False)

        # draw image
        if pix:
            painter.drawPixmap(rect, pix)

        # selection border: use the item's effective color when available
        if option.state & QtWidgets.QStyle.State_Selected:
            try:
                color = item.effective_color if hasattr(item, 'effective_color') else QtGui.QColor('#ffd700')
                pen = QtGui.QPen(QtGui.QColor(color))
                pen.setWidth(4)
                pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(rect.adjusted(2, 2, -2, -2))
            except Exception:
                pen = QtGui.QPen(QtGui.QColor('#ffd700'))
                pen.setWidth(4)
                pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(rect.adjusted(2, 2, -2, -2))

        # nametag with readable text color
        label = None
        try:
            label = item.effective_label.short_label_code if hasattr(item, 'effective_label') else (item.annotation.label.short_label_code if item.annotation and item.annotation.label else None)
        except Exception:
            label = None

        if label:
            tag_rect = QtCore.QRect(rect.right() - 60, rect.bottom() - 20, 58, 18)
            bg = item.effective_color if hasattr(item, 'effective_color') else QtGui.QColor('#000000')
            painter.fillRect(tag_rect, bg)
            # choose text color with sufficient contrast
            try:
                r, g, b = bg.red(), bg.green(), bg.blue()
                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                text_color = QtGui.QColor('#000000') if luminance > 0.5 else QtGui.QColor('#ffffff')
            except Exception:
                text_color = QtGui.QColor('#ffffff')
            painter.setPen(QtGui.QPen(text_color))
            painter.drawText(tag_rect, QtCore.Qt.AlignCenter, str(label))

    def editorEvent(self, event, model, option, index):
        # Toggle group on header click
        # Left-click on header toggles group
        try:
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                data = index.data(AnnotationListModel.DataItemRole)
                if data and data.get('type') == 'header':
                    group_key = data.get('key')
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
                                # select in gallery
                                try:
                                    viewer.clear_selection()
                                except Exception:
                                    pass
                                try:
                                    viewer.render_selection_from_ids([ann.id])
                                except Exception:
                                    pass

                                # Change image if needed and select annotation in AnnotationWindow
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
                                    if hasattr(viewer.annotation_window, 'center_on_annotation'):
                                        viewer.annotation_window.center_on_annotation(ann)
                                except Exception:
                                    pass
                                return True
                        except Exception:
                            pass
        except Exception:
            pass
        return super().editorEvent(event, model, option, index)

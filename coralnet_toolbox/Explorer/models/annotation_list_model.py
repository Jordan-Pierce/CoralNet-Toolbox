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
            return QtCore.QSize(option.rect.width(), self.header_height)
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
            bg = QtGui.QColor('#333333') if color is None else color
            painter.fillRect(rect, bg)
            # draw chevron
            expanded = data.get('expanded', True)
            pen = QtGui.QPen(QtGui.QColor('#ffffff'))
            painter.setPen(pen)
            text = data.get('text', '')
            painter.drawText(rect.adjusted(8, 0, 0, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)
            # chevron on right
            chev = '▾' if expanded else '▸'
            painter.drawText(rect.adjusted(-24, 0, -8, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, chev)
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

        # selection border
        if option.state & QtWidgets.QStyle.State_Selected:
            pen = QtGui.QPen(QtGui.QColor('#ffd700'))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(2, 2, -2, -2))

        # nametag
        label = getattr(item, 'short_label_code', None) or getattr(item, 'annotation', None) and getattr(item.annotation, 'label', None) and getattr(item.annotation.label, 'short_label_code', '')
        if label:
            tag_rect = QtCore.QRect(rect.right() - 60, rect.bottom() - 20, 58, 18)
            bg = getattr(item, 'effective_color', QtGui.QColor('#000000'))
            painter.fillRect(tag_rect, bg)
            painter.setPen(QtGui.QPen(QtGui.QColor('#fff')))
            painter.drawText(tag_rect, QtCore.Qt.AlignCenter, str(label))

    def editorEvent(self, event, model, option, index):
        # Toggle group on header click
        if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            data = index.data(AnnotationListModel.DataItemRole)
            if data and data.get('type') == 'header':
                group_key = data.get('key')
                # flip expanded state stored in model
                model._group_expanded[group_key] = not model._group_expanded.get(group_key, True)
                # ask view to rebuild by emitting layoutChanged; caller may re-set grouped items
                model.layoutChanged.emit()
                return True
        return super().editorEvent(event, model, option, index)

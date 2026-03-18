"""
Lightweight selection performance micro-benchmark.
Run with: python -m pytest tests/benchmark_selection.py -q
(or) python tests/benchmark_selection.py

This script constructs minimal GUI objects and measures:
- AnnotationListModel.set_grouped_items
- AnnotationViewerWindow.render_selection_from_ids

It uses a QApplication; keep in mind running it will open no visible windows.
"""

import sys
import time

from PyQt5.QtWidgets import QApplication

from coralnet_toolbox.Explorer.models.annotation_list_model import AnnotationListModel
from coralnet_toolbox.Explorer.ui.QtAnnotationViewerWindow import AnnotationViewerWindow
from coralnet_toolbox.QtAnnotationWindow import AnnotationWindow
from coralnet_toolbox.Explorer.core.QtDataItem import AnnotationDataItem


class MockLabel:
    def __init__(self):
        self.id = 'lbl'
        self.color = '#ffffff'
        self.short_label_code = 'L'


class MockAnnotation:
    def __init__(self, aid):
        self.id = str(aid)
        self.image_path = f"/tmp/img_{aid}.png"
        self.label = MockLabel()
        self.verified = False

    # Minimal stubs used by AnnotationDataItem
    def get_cropped_image(self):
        return None

    def get_cropped_image_graphic(self):
        return None


def make_grouped_items(n):
    items = []
    for i in range(n):
        ann = MockAnnotation(i)
        data_item = AnnotationDataItem(ann)
        items.append(data_item)
    # single group
    return [("All", None, items)]


def benchmark_set_grouped_items(n):
    model = AnnotationListModel()
    groups = make_grouped_items(n)
    start = time.perf_counter()
    model.set_grouped_items(groups)
    dur = time.perf_counter() - start
    print(f"set_grouped_items: n={n} time={dur:.4f}s")
    return dur


def benchmark_render_selection(n):
    # Create minimal app and windows
    app = QApplication.instance() or QApplication(sys.argv)
    class MW:
        pass
    mw = MW()
    mw.animation_manager = None
    # Create annotation window first (AnnotationViewerWindow expects it on main_window)
    ann_win = AnnotationWindow(mw)
    mw.annotation_window = ann_win
    mw.label_window = None

    viewer = AnnotationViewerWindow(mw)

    groups = make_grouped_items(n)
    # Synchronously set groups to build model mapping
    viewer.list_model.set_grouped_items(groups)

    ids = [str(i) for i in range(n)]
    start = time.perf_counter()
    viewer.render_selection_from_ids(ids)
    dur = time.perf_counter() - start
    print(f"render_selection_from_ids: n={n} time={dur:.4f}s")
    return dur


if __name__ == '__main__':
    sizes = [10, 100, 1000, 5000]
    for s in sizes:
        benchmark_set_grouped_items(s)
        benchmark_render_selection(s)

    print('Done')

import sys
import cv2
import numpy as np
import argparse
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QCheckBox, QComboBox, QGroupBox, QFormLayout, 
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

class InteractiveView(QGraphicsView):
    roi_selected = pyqtSignal(QRectF, bool) # bool: True for Positive, False for Negative

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.drawing = False
        self.is_positive = True
        self.start_point = QPointF()
        self.current_rect_item = None
        
        # Panning state
        self.panning = False
        self.pan_start = QPointF()

    def wheelEvent(self, event):
        # Zoom in/out
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        # Middle click for panning
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Left/Right click for drawing ROIs
        scene_pos = self.mapToScene(event.pos())
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self.drawing = True
            self.is_positive = (event.button() == Qt.LeftButton)
            self.start_point = scene_pos
            
            color = Qt.green if self.is_positive else Qt.red
            self.current_rect_item = self.scene().addRect(QRectF(self.start_point, self.start_point), QPen(color, 2, Qt.DashLine))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_start = event.pos()
            return

        if self.drawing and self.current_rect_item:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, scene_pos).normalized()
            self.current_rect_item.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if event.button() in (Qt.LeftButton, Qt.RightButton) and self.drawing:
            self.drawing = False
            if self.current_rect_item:
                rect = self.current_rect_item.rect()
                self.scene().removeItem(self.current_rect_item)
                self.current_rect_item = None
                self.roi_selected.emit(rect, self.is_positive)

        super().mouseReleaseEvent(event)

class CVAnnotatorV2(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Classic CV Annotation - V2 (Pan/Zoom/Negative Reinforcement)")
        
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)
            
        self.hsv_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)
        
        # Data banks for positive and negative color signatures
        self.pos_colors = []
        self.neg_colors = []
        self.target_area_baseline = 0

        self.initUI()
        self.update_view(self.original_img)

    def initUI(self):
        main_layout = QHBoxLayout()

        # Setup Scene and View
        self.scene = QGraphicsScene()
        self.view = InteractiveView(self.scene)
        self.view.roi_selected.connect(self.handle_roi)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        main_layout.addWidget(self.view)

        # Control Panel
        control_panel = QGroupBox("Refinement Engine")
        control_panel.setFixedWidth(320)
        form_layout = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Polygon", "Bounding Box", "Semantic Mask"])
        self.mode_combo.currentIndexChanged.connect(self.process_pipeline)
        form_layout.addRow("Output Mode:", self.mode_combo)

        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(5, 100)
        self.tol_slider.setValue(25)
        self.tol_slider.valueChanged.connect(self.process_pipeline)
        form_layout.addRow("Color Tolerance:", self.tol_slider)

        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(0, 10000)
        self.min_area_slider.setValue(10)
        self.min_area_slider.valueChanged.connect(self.process_pipeline)
        form_layout.addRow("Min Area (px):", self.min_area_slider)
        
        self.max_area_slider = QSlider(Qt.Horizontal)
        self.max_area_slider.setRange(100, 500000)
        self.max_area_slider.setValue(500000)
        self.max_area_slider.valueChanged.connect(self.process_pipeline)
        form_layout.addRow("Max Area (px):", self.max_area_slider)

        self.clear_btn = QPushButton("Clear Memory Banks")
        self.clear_btn.clicked.connect(self.clear_memory)
        form_layout.addRow("", self.clear_btn)

        control_panel.setLayout(form_layout)
        main_layout.addWidget(control_panel)
        self.setLayout(main_layout)
        self.resize(1200, 800)

    def clear_memory(self):
        self.pos_colors.clear()
        self.neg_colors.clear()
        self.target_area_baseline = 0
        self.update_view(self.original_img)

    def handle_roi(self, rect, is_positive):
        x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        # Constrain to image bounds
        img_h, img_w = self.hsv_img.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_w - x), min(h, img_h - y)
        
        if w < 2 or h < 2: return 
        
        roi = self.hsv_img[y:y+h, x:x+w]
        mean_color = cv2.mean(roi)[:3]
        
        if is_positive:
            self.pos_colors.append(np.array(mean_color))
            # Set baseline area and auto-adjust sliders
            self.target_area_baseline = w * h
            
            # Auto-set min/max area to 20% and 500% of target
            min_a = int(self.target_area_baseline * 0.2)
            max_a = int(self.target_area_baseline * 5.0)
            self.min_area_slider.setValue(min_a)
            self.max_area_slider.setValue(max_a)
        else:
            self.neg_colors.append(np.array(mean_color))
            
        self.process_pipeline()

    def build_color_mask(self, color_list, tol):
        mask = np.zeros(self.hsv_img.shape[:2], dtype=np.uint8)
        for color in color_list:
            lower = np.array([max(0, color[0] - tol), max(0, color[1] - tol*2), max(0, color[2] - tol*2)])
            upper = np.array([min(179, color[0] + tol), min(255, color[1] + tol*2), min(255, color[2] + tol*2)])
            current_mask = cv2.inRange(self.hsv_img, lower, upper)
            mask = cv2.bitwise_or(mask, current_mask)
        return mask

    def process_pipeline(self):
        if not self.pos_colors:
            self.update_view(self.original_img)
            return

        tol = self.tol_slider.value()
        
        # 1. Build Positive Mask
        pos_mask = self.build_color_mask(self.pos_colors, tol)
        
        # 2. Apply Negative Reinforcement (Subtractive)
        if self.neg_colors:
            neg_mask = self.build_color_mask(self.neg_colors, tol)
            pos_mask = cv2.bitwise_and(pos_mask, cv2.bitwise_not(neg_mask))

        # Cleanup
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(pos_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        output_img = self.original_img.copy()
        mode = self.mode_combo.currentText()
        min_area = self.min_area_slider.value()
        max_area = self.max_area_slider.value()
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if mode == "Semantic Mask":
            green_overlay = np.zeros_like(output_img)
            green_overlay[mask == 255] = [0, 255, 0]
            output_img = cv2.addWeighted(output_img, 0.7, green_overlay, 0.3, 0)
        else:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter by dynamic size heuristics
                if area < min_area or area > max_area:
                    continue
                
                if mode == "Bounding Box":
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                elif mode == "Polygon":
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    cv2.drawContours(output_img, [approx], 0, (0, 255, 0), 2)

        self.update_view(output_img)

    def update_view(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = CVAnnotatorV2(args.image)
    ex.show()
    sys.exit(app.exec_())
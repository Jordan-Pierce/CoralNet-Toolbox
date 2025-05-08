import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os 

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QComboBox, QSpinBox, QHBoxLayout,
                             QWidget, QStackedWidget, QGridLayout, QMessageBox,
                             QDialog, QListWidget, QPushButton, QFileDialog,
                             QGraphicsView)
from PyQt5.QtCore import Qt                           

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class GraphicsUtility:
    """
    Utility class for standardizing graphics appearance across different tools.
    Provides methods to calculate appropriate thickness and sizes for various graphics elements
    based on the current view scale.
    """
    
    @staticmethod
    def get_workarea_thickness(view):
        """
        Calculate line thickness for work areas so it appears visually consistent 
        regardless of zoom or image size.
        
        Args:
            view (QGraphicsView): The view containing the graphics
            
        Returns:
            int: Appropriate line thickness in pixels
        """
        if not hasattr(view, 'pixmap_image') or not view.pixmap_image:
            return 5  # fallback thickness
            
        # Get the current zoom scale from the view's transformation matrix
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero
            
        desired_px = 5  # Desired thickness in screen pixels
        thickness = max(1, int(round(desired_px / scale)))
        return thickness
    
    @staticmethod
    def get_rectangle_graphic_thickness(view):
        """
        Calculate line thickness for rectangles so it appears visually consistent 
        regardless of zoom or image size.
        
        Args:
            view (QGraphicsView): The view containing the graphics
            
        Returns:
            int: Appropriate line thickness in pixels
        """
        if not hasattr(view, 'pixmap_image') or not view.pixmap_image:
            return 2  # fallback thickness
            
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero
            
        desired_px = 2  # Desired thickness in screen pixels
        thickness = max(1, int(round(desired_px / scale)))
        return thickness
    
    @staticmethod
    def get_handle_size(view):
        """
        Calculate the size for handles (resize handles, control points) so they 
        appear visually consistent regardless of zoom or image size.
        
        Args:
            view (QGraphicsView): The view containing the graphics
            
        Returns:
            int: Appropriate handle size in pixels
        """
        if not hasattr(view, 'pixmap_image') or not view.pixmap_image:
            return 10  # fallback size
            
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero
            
        desired_px = 10  # Desired handle size in screen pixels
        size = max(6, int(round(desired_px / scale)))
        return size
    
    @staticmethod
    def get_path_thickness(view):
        """
        Calculate line thickness for paths (cutting lines, polylines) so they 
        appear visually consistent regardless of zoom or image size.
        
        Args:
            view (QGraphicsView): The view containing the graphics
            
        Returns:
            int: Appropriate line thickness in pixels
        """
        if not hasattr(view, 'pixmap_image') or not view.pixmap_image:
            return 3  # fallback thickness
            
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero
            
        desired_px = 3  # Desired thickness in screen pixels
        thickness = max(1, int(round(desired_px / scale)))
        return thickness
    
    @staticmethod
    def get_dot_size(view):
        """
        Calculate the size for dots (points, markers) so they appear visually 
        consistent regardless of zoom or image size.
        
        Args:
            view (QGraphicsView): The view containing the graphics
            
        Returns:
            int: Appropriate dot size in pixels
        """
        if not hasattr(view, 'pixmap_image') or not view.pixmap_image:
            return 8  # fallback size
            
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero
            
        desired_px = 8  # Desired dot size in screen pixels
        size = max(4, int(round(desired_px / scale)))
        return size
    
    @staticmethod
    def get_selection_thickness(view):
        """
        Calculate line thickness for selection rectangle so it appears visually 
        consistent regardless of zoom or image size.
        
        Args:
            view (QGraphicsView): The view containing the graphics
            
        Returns:
            int: Appropriate line thickness in pixels
        """
        extent = view.viewportToScene() if hasattr(view, 'viewportToScene') else None
        if not extent:
            return 2  # fallback thickness
            
        view_width = round(extent.width())
        view_height = round(extent.height())
        return max(2, min(5, max(view_width, view_height) // 1000))
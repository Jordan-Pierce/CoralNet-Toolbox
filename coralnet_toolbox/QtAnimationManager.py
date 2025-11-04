import warnings

from PyQt5.QtCore import QTimer, QObject

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnimationManager(QObject):
    """
    Manages a single, global timer to handle animations for all
    registered objects (like Annotations and WorkAreas).
    
    This avoids creating thousands of QTimer objects, which can
    exhaust system resources and crash the application.
    """
    
    def __init__(self, parent=None):
        """Initialize the manager."""
        super().__init__(parent)
        
        # The set of all objects that are currently animating
        self.animating_objects = set()
        
        # The single, global timer that drives all animations
        self.global_animation_timer = QTimer(self)
        self.global_animation_timer.timeout.connect(self._update_all_animations)

    def start_timer(self, interval=50):
        """
        Starts the global animation timer.
        
        Args:
            interval (int): The interval in milliseconds (e.g., 50ms = 20 FPS).
        """
        if not self.global_animation_timer.isActive():
            self.global_animation_timer.start(interval)
            
    def stop_timer(self):
        """Stops the global animation timer."""
        self.global_animation_timer.stop()

    def register_animating_object(self, obj):
        """
        Add an object to the set of items to be animated.
        
        Args:
            obj: The object to animate. Must have:
                 - is_graphics_item_valid() -> bool
                 - tick_animation()
        """
        self.animating_objects.add(obj)
        
    def unregister_animating_object(self, obj):
        """
        Remove an object from the set of items to be animated.
        
        Args:
            obj: The object to stop animating.
        """
        self.animating_objects.discard(obj)

    def _update_all_animations(self):
        """
        This is the engine, called by the timer on every tick.
        It loops through all registered objects and calls their
        tick_animation() method.
        """
        if not self.animating_objects:
            return
            
        # Loop over a static copy (list) of the set.
        # This allows objects to be safely unregistered during iteration
        # (e.g., if they are deleted).
        for obj in list(self.animating_objects):
            try:
                # 1. Safely check if the object is still "alive"
                #    by checking its graphics item.
                if obj and obj.is_graphics_item_valid():
                    # 2. If alive, call its public animation method
                    obj.tick_animation()
                else:
                    # 3. If it's not valid (e.g., deleted), unregister it
                    self.unregister_animating_object(obj)
            
            except (RuntimeError, AttributeError):
                # 4. Catch any other errors (e.g., C++ object deleted)
                #    and unregister the object to prevent further errors.
                self.unregister_animating_object(obj)
from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.Annotations import PolygonAnnotation


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------    

class SubtractSubTool(SubTool):
    """
    A SubTool to perform a "cookie cutter" subtraction operation on selected annotations.
    This tool activates, performs its action, and immediately deactivates.
    """

    def __init__(self, parent_tool):
        super().__init__(parent_tool)

    def activate(self, event, **kwargs):
        """
        Activates the subtraction operation.
        Expects 'selected_annotations' in kwargs.
        """
        super().activate(event)
        
        selected_annotations = kwargs.get('selected_annotations', [])

        # --- 1. Perform Pre-activation Checks ---
        if len(selected_annotations) < 2:
            # Not enough annotations to perform a subtraction.
            self.parent_tool.deactivate_subtool()
            return

        # Check if all annotations are verified PolygonAnnotations
        if not all(isinstance(anno, PolygonAnnotation) and anno.verified for anno in selected_annotations):
            self.parent_tool.deactivate_subtool()
            return

        # --- 2. Identify Base and Cutters ---
        # The last selected annotation is the base, the rest are cutters.
        base_annotation = selected_annotations[-1]
        cutter_annotations = selected_annotations[:-1]

        # --- 3. Perform the Subtraction ---
        result_annotation = PolygonAnnotation.subtract(base_annotation, cutter_annotations)

        if not result_annotation:
            self.parent_tool.deactivate_subtool()
            return

        # --- 4. Update the Annotation Window ---
        # Add the new combined annotation to the scene
        self.annotation_window.add_annotation_from_tool(result_annotation)
        
        # Delete the original annotations that were used in the operation
        for anno in selected_annotations:
            self.annotation_window.delete_annotation(anno.id)
        
        # Select the new combined annotation
        self.annotation_window.select_annotation(result_annotation)

        # --- 5. Deactivate Immediately ---
        self.parent_tool.deactivate_subtool()


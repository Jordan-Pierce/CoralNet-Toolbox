from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.Annotations import PolygonAnnotation
from coralnet_toolbox.Annotations import MultiPolygonAnnotation


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
            self.parent_tool.deactivate_subtool()
            return

        # Check if all annotations are verified Polygon or MultiPolygon Annotations
        allowed_types = (PolygonAnnotation, MultiPolygonAnnotation)
        if not all(isinstance(anno, allowed_types) and anno.verified for anno in selected_annotations):
            self.parent_tool.deactivate_subtool()
            return

        # --- 2. Identify Base and Cutters ---
        # The last selected annotation is the base, the rest are cutters.
        base_annotation = selected_annotations[-1]
        cutter_annotations = selected_annotations[:-1]

        # --- 3. Perform the Subtraction ---
        result_annotations = PolygonAnnotation.subtract(base_annotation, cutter_annotations)

        if not result_annotations:
            self.parent_tool.deactivate_subtool()
            return

        # --- 4. Update the Annotation Window ---
        # Delete the original annotations that were used in the operation
        for anno in selected_annotations:
            self.annotation_window.delete_annotation(anno.id)
        
        # Add the new resulting annotations to the scene
        for new_anno in result_annotations:
            self.annotation_window.add_annotation_from_tool(new_anno)
        
        # Select the new annotations
        self.annotation_window.unselect_annotations()
        for new_anno in result_annotations:
            self.annotation_window.select_annotation(new_anno, multi_select=True)

        # --- 5. Deactivate Immediately ---
        self.parent_tool.deactivate_subtool()
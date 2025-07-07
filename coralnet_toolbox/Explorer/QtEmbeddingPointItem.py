import warnings

from PyQt5.QtWidgets import QGraphicsEllipseItem, QStyle

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingPointItem(QGraphicsEllipseItem):
    """
    A custom QGraphicsEllipseItem that prevents the default selection
    rectangle from being drawn, and dynamically gets its color from the
    shared AnnotationDataItem.
    """

    def paint(self, painter, option, widget):
        # Get the shared data item, which holds the current state
        data_item = self.data(0)
        if data_item:
            # Set the brush color based on the item's effective color
            # This ensures preview colors are reflected instantly.
            self.setBrush(data_item.effective_color)

        # Remove the 'State_Selected' flag to prevent the default box
        option.state &= ~QStyle.State_Selected
        super(EmbeddingPointItem, self).paint(painter, option, widget)

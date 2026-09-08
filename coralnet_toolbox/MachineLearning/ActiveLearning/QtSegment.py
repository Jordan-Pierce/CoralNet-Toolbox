import warnings

from coralnet_toolbox.MachineLearning.ActiveLearning.QtBase import Base

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    """Active Learning rounds that train an instance segmentor.

    Learns outlines rather than boxes, so it reads Polygons only.
    """

    # Read while Base.__init__ builds the layout, so it has to be class-level.
    task = 'segment'

import warnings

from coralnet_toolbox.MachineLearning.ActiveLearning.QtBase import Base

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    """Active Learning rounds that train a detector.

    Learns boxes, so it reads Rectangles, Polygons and Patches -- anything with
    a bounding box.
    """

    # Read while Base.__init__ builds the layout, so it has to be class-level.
    task = 'detect'

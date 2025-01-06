import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from coralnet_toolbox.MachineLearning.TileDataset.QtBase import Base

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Tile Detection Dataset")
        
        self.deploy_model_dialog = main_window.detect_deploy_model_dialog
        
    def setup_task_specific_layout(self):
        """
        Set up the layout with both generic and classification-specific options.
        """
        pass
        
    def batch_inference(self):
        """
        Perform batch inference on the selected images.
        """
        src = self.src_edit.text()
        dst = self.dst_edit.text()

        config = TileConfig(
            slice_wh=eval(self.slice_wh_edit.text()),
            overlap_wh=eval(self.overlap_wh_edit.text()),
            ext=self.ext_edit.text(),
            annotation_type=self.annotation_type_combo.currentText(),
            densify_factor=self.densify_factor_spinbox.value(),
            smoothing_tolerance=self.smoothing_tolerance_spinbox.value(),
            train_ratio=self.train_ratio_spinbox.value(),
            valid_ratio=self.valid_ratio_spinbox.value(),
            test_ratio=self.test_ratio_spinbox.value(),
            margins=eval(self.margins_edit.text())
        )

        tiler = YoloTiler(
            source=src,
            target=dst,
            config=config,
            num_viz_samples=15,
        )

        tiler.run()

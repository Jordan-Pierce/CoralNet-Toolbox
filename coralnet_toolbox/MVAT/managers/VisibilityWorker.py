import traceback
from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager


class WorkerSignals(QObject):
    """
    Signals emitted by the worker.
    finished: dict mapping image_path -> result dict
    error: str traceback
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


class VisibilityWorker(QObject):
    """
    Background worker for computing camera visibility maps.

    Accepts only raw data (numpy arrays, lists, dicts). Must be moved to a QThread
    before starting: worker.moveToThread(thread); thread.started.connect(worker.run)
    """

    def __init__(self, points_world, camera_params_dict, compute_depth_maps=True):
        super().__init__()
        self.points_world = points_world
        # camera_params_dict: {image_path: (K, R, t, width, height)}
        self.camera_params_dict = camera_params_dict
        self.compute_depth_maps = compute_depth_maps
        self.signals = WorkerSignals()

    def run(self):
        try:
            paths = list(self.camera_params_dict.keys())
            params_list = list(self.camera_params_dict.values())

            # Call the batch visibility API
            batch_results = VisibilityManager.compute_batch_visibility(
                points_world=self.points_world,
                camera_params_list=params_list,
                compute_depth_map=self.compute_depth_maps
            )

            # Map results back to camera paths
            mapped = {p: r for p, r in zip(paths, batch_results)}

            self.signals.finished.emit(mapped)

        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{e}\n{tb}")

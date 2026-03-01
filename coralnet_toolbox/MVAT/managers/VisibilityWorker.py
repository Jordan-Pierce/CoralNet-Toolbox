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
        # camera_params_dict: {
        #   image_path: (K, R, t, width, height)  # for perspective
        #   OR
        #   image_path: ('ortho', transform_matrix_inv, width, height)  # for orthographic
        # }
        self.camera_params_dict = camera_params_dict
        self.compute_depth_maps = compute_depth_maps
        self.signals = WorkerSignals()

    def run(self):
        try:
            # Separate orthographic and perspective cameras
            ortho_params = {}
            perspective_params = {}
            
            for path, params in self.camera_params_dict.items():
                # Be defensive: params can contain numpy arrays (e.g. K) and
                # comparing an array to a string raises a ValueError. Detect
                # orthographic entries by ensuring the first element is a
                # str and equals 'ortho'. Otherwise treat as perspective.
                try:
                    first = params[0]
                except Exception:
                    # Malformed/empty params -> treat as perspective entry
                    perspective_params[path] = params
                    continue

                if isinstance(first, str) and first == 'ortho':
                    # Extract: ('ortho', transform_matrix_inv, width, height)
                    _, transform_inv, width, height = params
                    ortho_params[path] = (transform_inv, width, height)
                else:
                    # Perspective camera: (K, R, t, width, height)
                    perspective_params[path] = params
            
            results = {}
            
            # PERSPECTIVE: Use existing GPU batch processing
            if perspective_params:
                paths = list(perspective_params.keys())
                params_list = list(perspective_params.values())
                
                batch_results = VisibilityManager.compute_batch_visibility(
                    points_world=self.points_world,
                    camera_params_list=params_list,
                    compute_depth_map=self.compute_depth_maps
                )
                
                for p, r in zip(paths, batch_results):
                    results[p] = r
            
            # ORTHOGRAPHIC: Use affine transform processing
            if ortho_params:
                for path, (transform_inv, width, height) in ortho_params.items():
                    result = VisibilityManager.compute_orthographic_visibility(
                        points_world=self.points_world,
                        transform_matrix_inv=transform_inv,
                        width=width,
                        height=height
                    )
                    results[path] = result
            
            self.signals.finished.emit(results)

        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{e}\n{tb}")

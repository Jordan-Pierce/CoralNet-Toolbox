from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TileProcessor:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.tile_params = {}
        self.tile_inference_params = {}
    
    def params_set(self):
        return self.tile_params and self.tile_inference_params
        
    def set_tile_params(self, tile_params):
        self.tile_params = tile_params
    
    def set_tile_inference_params(self, tile_inference_params):
        self.tile_inference_params = tile_inference_params
        
    def set_params(self, tile_params, tile_inference_params):
        self.set_tile_params(tile_params)
        self.set_tile_inference_params(tile_inference_params)
        
    
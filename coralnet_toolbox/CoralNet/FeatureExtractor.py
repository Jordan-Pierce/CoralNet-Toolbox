from io import BytesIO
from collections import OrderedDict

import timm
import torch

from coralnet_toolbox.CoralNet import EfficientNet


# ----------------------------------------------------------------------------------------------------------------------
# Classes 
# ----------------------------------------------------------------------------------------------------------------------


class FeatureExtractor:
    def __init__(self, model_name='efficientnet-b0', weights_file=None):

        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.weights_file = weights_file

        self.model = self.load_model()
        self.file_name = None

    @staticmethod
    def _load_weights(model, weights_datastream: BytesIO):
        """
        Load model weights, original weights were saved with DataParallel
        Create new OrderedDict that does not contain `module`.
        :param model: Currently support EfficientNet
        :param weights_datastream: model weights, already loaded from storage
        :return: well trained model
        """
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load weights
        state_dicts = torch.load(weights_datastream, map_location=device)

        new_state_dicts = OrderedDict()
        for k, v in state_dicts['net'].items():
            name = k[7:]
            new_state_dicts[name] = v
        model.load_state_dict(new_state_dicts)

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_model(self):
        """Load model from timm library."""
        if self.weights_file is not None and self.weights_file.endswith('.pt'):
            print("Loading pretrained model ",
                  self.model_name, ' from ', self.weights_file)

            # Load efficient net
            efficientnet = EfficientNet.get_model('efficientnet', self.model_name, 1275)

            with open(self.weights_file, 'rb') as f:
                weights_datastream = BytesIO(f.read())
                self.model = self._load_weights(
                    efficientnet, weights_datastream)
        else:
            print("Loading empty model ", self.model_name)
            self.model = timm.create_model(
                self.model_name, pretrained=True, num_classes=0)

        self.model.to(self.device)
        self.model.eval()
        return self.model

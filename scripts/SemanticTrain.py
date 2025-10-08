import os
import sys
import json
import yaml
import glob
import shutil
import inspect
import warnings
import argparse
import traceback
from tqdm import tqdm
from datetime import datetime


import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.meter import AverageValueMeter

import albumentations as albu

torch.cuda.empty_cache()

warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_now():
    """Get current date and time as a formatted string."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def get_segmentation_encoders():
    """Get list of available segmentation model encoders."""
    encoder_options = []

    try:
        options = smp.encoders.get_encoder_names()
        encoder_options = options

    except Exception as e:
        # Fail silently
        pass

    return encoder_options


def get_segmentation_decoders():
    """Get list of available segmentation model decoders."""
    decoder_options = []
    try:

        options = [_ for _, obj in inspect.getmembers(smp) if inspect.isclass(obj)]
        decoder_options = options

    except Exception as e:
        # Fail silently
        pass

    return decoder_options


def get_segmentation_losses():
    """Get list of available segmentation loss functions."""
    loss_options = []

    try:
        options = [attr for attr in dir(smp.losses) if callable(getattr(smp.losses, attr))]
        options = [o for o in options if o != 'Activation']
        loss_options = options

    except Exception as e:
        # Fail silently
        pass

    return loss_options


def get_segmentation_metrics():
    """Get list of available segmentation metrics."""
    metric_options = ['accuracy',
                      'balanced_accuracy',
                      'f1_score',
                      'fbeta_score',
                      'iou_score',
                      'precision',
                      'recall',
                      'sensitivity',
                      'specificity']

    return metric_options


def get_segmentation_optimizers():
    """Get list of available PyTorch optimizers."""
    optimizer_options = []

    try:
        options = [attr for attr in dir(torch.optim) if callable(getattr(torch.optim, attr))]
        optimizer_options = options

    except Exception as e:
        # Fail silently
        pass

    return optimizer_options


def format_logs_pretty(logs, title="Results"):
    """Format logs into a pretty, readable string."""
    if not logs:
        return ""
    
    # Separate loss and metrics
    loss_items = []
    metric_items = []
    
    for k, v in logs.items():
        if 'loss' in k.lower():
            loss_items.append(f"{k}: {v:.4f}")
        else:
            metric_items.append(f"{k}: {v:.4f}")
    
    # Combine all items
    all_items = loss_items + metric_items
    return " | ".join(all_items)


def visualize(save_path=None, save_figure=False, image=None, **masks):
    """Visualize segmentation masks overlaid on the original image."""
    # Get the number of images
    n = len(masks)
    # Plot figure
    plt.figure(figsize=(16, 5))
    # Loop through images, masks, and plot
    for i, (name, mask) in enumerate(masks.items()):
        plt.subplot(1, n, i + 1)
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)

    # Save the figure if save_figure is True and save_path is provided
    if save_figure and save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"🖼️ Figure saved to {save_path}")

    # Show the figure
    plt.close()


def colorize_mask(mask, class_ids, class_colors):
    """Convert a grayscale mask to RGB using class-specific colors."""
    # Initialize the RGB mask with zeros
    height, width = mask.shape[0:2]

    rgb_mask = np.full((height, width, 3), fill_value=0, dtype=np.uint8)

    # dict with index as key, rgb as value
    cmap = {i: class_colors[i_idx] for i_idx, i in enumerate(class_ids)}

    # Loop through all index values
    # Set rgb color in colored mask
    for val in np.unique(mask):
        if val in class_ids and val != 0:  # Skip background class
            color = np.array(cmap[val])
            rgb_mask[mask == val, :] = color.astype(np.uint8)

    return rgb_mask.astype(np.uint8)


def get_training_augmentation(imgsz):
    """Get data augmentation pipeline for training semantic segmentation."""
    train_transform = [
        # Basic spatial transformations
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.2),
        
        # Geometric transformations
        albu.OneOf([
            albu.Rotate(limit=45, p=1.0, border_mode=0, value=0),
            albu.Affine(scale=(0.8, 1.2), rotate=(-45, 45), shear=(-10, 10), p=1.0, mode=0, cval=0),
        ], p=0.5),
        
        # Elastic deformation for robustness
        albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        
        # Grid distortion
        albu.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        
        # Resize and padding
        albu.LongestMaxSize(max_size=imgsz),
        albu.PadIfNeeded(min_height=imgsz, min_width=imgsz, always_apply=True, border_mode=0, value=0),
        
        # Color and intensity augmentations
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            albu.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.8),
        
        # Hue/Saturation/Value adjustments
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        
        # Noise and artifacts
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 50), p=1.0),
            albu.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Blur and sharpening
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1.0),
            albu.MotionBlur(blur_limit=3, p=1.0),
            albu.GaussianBlur(blur_limit=3, p=1.0),
            albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.4),
        
        # Dropout and cutout
        albu.OneOf([
            albu.PixelDropout(dropout_prob=0.1, p=1.0),
            albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
        ], p=0.3),
        
        # Weather effects (subtle)
        albu.OneOf([
            albu.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
            albu.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1.0),
            albu.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, p=1.0),
        ], p=0.1),
    ]

    return albu.Compose(train_transform)


def get_validation_augmentation(imgsz):
    """Get data augmentation pipeline for validation."""
    test_transform = [
        albu.LongestMaxSize(max_size=imgsz),
        albu.PadIfNeeded(min_height=imgsz, min_width=imgsz, always_apply=True, border_mode=0, value=0),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """Convert numpy array to tensor format."""
    if len(x.shape) == 2:
        return x
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Get preprocessing pipeline for model input."""

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def build_dataframe_from_yolo(base_path, train_dir, val_dir, test_dir):
    """
    Build a dataframe from YOLO directory structure.
    
    Args:
        base_path (str): Base path from data.yaml
        train_dir (str): Train directory path
        val_dir (str): Validation directory path  
        test_dir (str): Test directory path (optional)
        
    Returns:
        pd.DataFrame: Dataframe with columns 'Name', 'Image', 'Mask', 'Split'
    """
    data_rows = []
    
    # Helper function to process a single split
    def process_split(split_dir, split_name):
        print(f"Processing {split_name} split at {split_dir}")
        if not split_dir:
            return
            
        images_dir = split_dir
        labels_dir = split_dir.replace('images', 'labels')
        
        if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
            return
            
        image_files = []
        image_files.extend(glob.glob(os.path.join(images_dir, '*.jpg')))
        image_files.extend(glob.glob(os.path.join(images_dir, '*.png')))
        image_files.extend(glob.glob(os.path.join(images_dir, '*.jpeg')))
                
        for img_path in image_files:
            name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(labels_dir, name + '.png')
            
            if os.path.exists(mask_path):
                data_rows.append({
                    'Name': name,
                    'Image': img_path,
                    'Mask': mask_path,
                    'Split': split_name
                })
    
    # Process each split
    process_split(train_dir, 'train')
    process_split(val_dir, 'val')
    process_split(test_dir, 'test')
    
    if not data_rows:
        raise Exception("ERROR: No valid image-mask pairs found in YOLO directories")
    
    return pd.DataFrame(data_rows)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        """Move model components to the specified device."""
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        """Format training logs into a readable string."""
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        """Perform a single batch update (to be implemented by subclasses)."""
        raise NotImplementedError

    def on_epoch_start(self):
        """Called at the start of each epoch."""
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not self.verbose,
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # Convert y_pred
                num_classes = y_pred.shape[1]
                y_pred = torch.argmax(y_pred, axis=1)

                # Calculate the stats
                tp, fp, fn, tn = smp.metrics.functional._get_stats_multiclass(output=y_pred,
                                                                              target=y,
                                                                              num_classes=num_classes,
                                                                              ignore_index=0)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_values = smp.metrics.functional._compute_metric(metric_fn, tp, fp, fn, tn)
                    metrics_meters[metric_fn.__name__].add(torch.mean(metric_values))
                metrics_logs = {k: v.mean.item() for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        """Set model to training mode."""
        self.model.train()

    def batch_update(self, x, y):
        """Perform training batch update with loss computation and backpropagation."""
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def batch_update(self, x, y):
        """Perform validation batch update with loss computation."""
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


class TorchMetic(torch.nn.Module):
    """Wrapper class to make custom metric functions compatible with PyTorch modules."""

    def __init__(self, func):
        """Initialize TorchMetic with a custom metric function."""
        super(TorchMetic, self).__init__()
        self.func = func  # The custom function to be wrapped

    def forward(self, *args, **kwargs):
        """Execute the wrapped metric function on the specified device."""
        # Check if a device is specified in the keyword arguments
        device = kwargs.get('device', 'cpu')

        # Move any input tensors to the specified device
        args = [arg.to(device) if torch.is_tensor(arg) else arg for arg in args]

        # Execute the custom function on the specified device
        result = self.func(*args)

        return result

    @property
    def __name__(self):
        """Return the name of the wrapped function."""
        return self.func.__name__


class Dataset(BaseDataset):
    """Custom dataset class for semantic segmentation with image and mask loading."""

    def __init__(
            self,
            dataframe,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        """Initialize dataset with dataframe containing image and mask paths."""
        assert 'Name' in dataframe.columns, print(f"❌ 'Name' column not found in mask file")
        assert 'Image' in dataframe.columns, print(f"❌ 'Image' column not found in mask file")
        assert 'Mask' in dataframe.columns, print(f"❌ 'Mask' column not found in mask file")

        self.ids = dataframe['Name'].to_list()
        self.masks_fps = dataframe['Mask'].to_list()
        self.images_fps = dataframe['Image'].to_list()

        # convert str names to class values on masks
        self.class_ids = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """Load and preprocess a single image-mask pair."""
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # Validate mask values are within expected range
        unique_values = np.unique(mask)
        max_class_id = len(self.class_ids) - 1
        
        # Check for invalid values
        invalid_values = unique_values[unique_values > max_class_id]
        if len(invalid_values) > 0:
            print(f"⚠️ Found invalid mask values {invalid_values} in {self.masks_fps[i]}")
            print(f"🔧 Expected values 0-{max_class_id}, clamping to valid range")
            # Clamp invalid values to the maximum valid class ID
            mask = np.clip(mask, 0, max_class_id)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return torch.from_numpy(image), torch.from_numpy(mask).long()

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.ids)


class DataConfig:
    """Handles loading and parsing of YOLO data.yaml configuration files."""

    def __init__(self, data_yaml_path):
        """Initialize DataConfig with path to data.yaml file."""
        self.data_yaml_path = data_yaml_path
        self._load_config()
        self._build_class_mappings()
        self._build_dataframe()

    def _load_config(self):
        """Load and parse the data.yaml file."""
        if not os.path.exists(self.data_yaml_path):
            raise Exception(f"ERROR: data.yaml file does not exist: {self.data_yaml_path}")

        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)

        # Extract paths and class info
        self.base_path = self.data_config.get('path', os.path.dirname(self.data_yaml_path))
        self.train_dir = self.data_config['train']
        self.val_dir = self.data_config['val']
        self.test_dir = self.data_config.get('test')
        self.nc = self.data_config['nc']
        self.names = self.data_config['names']

    def _build_class_mappings(self):
        """Build class name and ID mappings with colors."""
        # Build class mappings - Include background class (0)
        self.class_names = [self.names[i] for i in range(self.nc)]  # Include all classes including background
        self.class_ids = list(range(self.nc))  # Include background (0)

        # Generate default colors for classes
        np.random.seed(42)  # For reproducible colors
        self.class_colors = []
        for i in range(len(self.class_names)):
            color = np.random.randint(0, 256, 3).tolist()
            self.class_colors.append(color)

        # Create color_map dict for compatibility
        self.color_map = {}
        for i, name in enumerate(self.class_names):
            self.color_map[name] = {
                'id': self.class_ids[i],
                'color': self.class_colors[i]
            }

    def _build_dataframe(self):
        """Build dataframe from YOLO directory structure."""
        self.dataframe = build_dataframe_from_yolo(
            self.base_path, self.train_dir, self.val_dir, self.test_dir
        )

    def get_split_dataframes(self):
        """Return train, validation, and test dataframes."""
        train_df = self.dataframe[self.dataframe['Split'] == 'train'].copy()
        valid_df = self.dataframe[self.dataframe['Split'] == 'val'].copy()
        test_df = self.dataframe[self.dataframe['Split'] == 'test'].copy()

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, valid_df, test_df


class ModelBuilder:
    """Handles construction and configuration of segmentation models."""

    def __init__(self, encoder_name, decoder_name, num_classes, freeze_encoder=0.0, pre_trained_path=None):
        """Initialize ModelBuilder with model configuration."""
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.pre_trained_path = pre_trained_path

        self._validate_options()
        self._build_model()
        self._load_pretrained_weights()
        self._freeze_encoder()

    def _validate_options(self):
        """Validate that encoder and decoder options are available."""
        if self.encoder_name not in get_segmentation_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_segmentation_encoders()}")

        if self.decoder_name not in get_segmentation_decoders():
            raise Exception(f"ERROR: Decoder must be one of {get_segmentation_decoders()}")

    def _build_model(self):
        """Build the segmentation model."""
        encoder_weights = 'imagenet'

        self.model = getattr(smp, self.decoder_name)(
            encoder_name=self.encoder_name,
            encoder_weights=encoder_weights,
            classes=self.num_classes,
            activation='softmax2d',
        )

        print(f"   • Architecture: {self.encoder_name} → {self.decoder_name}")
        print(f"   • Output classes: {self.num_classes}")

        # Get preprocessing function
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder_name, encoder_weights)

    def _load_pretrained_weights(self):
        """Load pretrained weights if provided."""
        if self.pre_trained_path:
            if not os.path.exists(self.pre_trained_path):
                print("⚠️ Pre-trained encoder path not found, using random initialization")
                return

            pre_trained_model = torch.load(self.pre_trained_path, map_location='cpu')

            try:
                # Getting the encoder name (preprocessing), and encoder state
                encoder_name = pre_trained_model.name
                state_dict = pre_trained_model.encoder.state_dict()
            except:
                encoder_name = pre_trained_model.encoder.name
                state_dict = pre_trained_model.encoder.encoder.state_dict()

            self.model.encoder.load_state_dict(state_dict, strict=True)
            print(f"📥 Loaded pre-trained weights from {encoder_name}")
        else:
            print("📥 Using ImageNet pretrained weights")

    def _freeze_encoder(self):
        """Freeze a percentage of the encoder weights."""
        num_params = len(list(self.model.encoder.parameters()))
        freeze_params = int(num_params * self.freeze_encoder)

        # Give users the ability to freeze N percent of the encoder
        print(f"🧊 Freezing {self.freeze_encoder}% of encoder weights")
        for idx, param in enumerate(self.model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False


class TrainingConfig:
    """Handles configuration of loss functions, optimizers, and metrics for training."""

    def __init__(self, model, loss_function_name, optimizer_name, learning_rate, metrics_list, class_ids, device):
        """Initialize TrainingConfig with training components."""
        self.model = model
        self.loss_function_name = loss_function_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.metrics_list = metrics_list
        self.class_ids = class_ids
        self.device = device

        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_metrics()

    def _setup_loss_function(self):
        """Setup the loss function."""
        assert self.loss_function_name in get_segmentation_losses()

        # For semantic segmentation, use multiclass mode
        mode = 'multiclass'
        self.loss_function = getattr(smp.losses, self.loss_function_name)(mode=mode).to(self.device)
        self.loss_function.__name__ = self.loss_function._get_name()

        # Get the parameters of the loss function using inspect.signature
        params = inspect.signature(self.loss_function.__init__).parameters

        # Set ignore_index to 0 (background) if supported
        if 'ignore_index' in params:
            self.loss_function.ignore_index = 0  # Ignore background class

        print(f"   • Loss: {self.loss_function_name} (background ignored)")

    def _setup_optimizer(self):
        """Setup the optimizer."""
        assert self.optimizer_name in get_segmentation_optimizers()
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), self.learning_rate)
        print(f"   • Optimizer: {self.optimizer_name} (lr={self.learning_rate})")

    def _setup_metrics(self):
        """Setup the evaluation metrics."""
        assert any(m in get_segmentation_metrics() for m in self.metrics_list)
        self.metrics = [getattr(smp.metrics, m) for m in self.metrics_list]

        # Include at least one metric regardless
        if not self.metrics:
            self.metrics.append(smp.metrics.iou_score)

        # Convert to torch metric so can be used on CUDA
        self.metrics = [TorchMetic(m) for m in self.metrics]
        print(f"   • Metrics: {self.metrics_list}")


class ExperimentManager:
    """Manages experiment directories, logging, and result tracking."""

    def __init__(self, output_dir, decoder_name, encoder_name, color_map, metrics):
        """Initialize ExperimentManager with experiment configuration."""
        self.output_dir = output_dir
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.color_map = color_map
        self.metrics = metrics

        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self):
        """Create experiment directories."""
        # Run Name
        self.run = f"{self.encoder_name}_{self.decoder_name}_{get_now()}"

        # Set run directory directly under output_dir
        self.run_dir = os.path.join(self.output_dir, self.run)
        self.weights_dir = os.path.join(self.run_dir, "weights")
        self.logs_dir = os.path.join(self.run_dir, "logs")

        # Make the directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        print(f"📁 Experiment: {self.run}")
        print(f"📁 Run Directory: {self.run_dir}")
        print(f"📁 Weights Directory: {self.weights_dir}")
        print(f"📁 Logs Directory: {self.logs_dir}")

    def _setup_logging(self):
        """Setup logging files and CSV results tracking."""
        # Save the generated color_map to JSON
        color_map_path = os.path.join(self.run_dir, "color_map.json")
        with open(color_map_path, 'w') as f:
            json.dump(self.color_map, f, indent=4)

        # Create results CSV file
        self.results_csv_path = os.path.join(self.logs_dir, "results.csv")
        with open(self.results_csv_path, 'w') as f:
            f.write("epoch,phase,loss")
            for metric in self.metrics:
                f.write(f",{metric.__name__}")
            f.write("\n")

    def save_dataframes(self, train_df, valid_df, test_df):
        """Save dataframes to CSV files in logs directory."""
        train_df.to_csv(os.path.join(self.logs_dir, "Training_Set.csv"), index=False)
        valid_df.to_csv(os.path.join(self.logs_dir, "Validation_Set.csv"), index=False)
        test_df.to_csv(os.path.join(self.logs_dir, "Testing_Set.csv"), index=False)

    def log_metrics(self, epoch, phase, logs):
        """Log metrics to CSV file."""
        row = f"{epoch},{phase},{logs.get('loss', 0)}"
        for metric in self.metrics:
            row += f",{logs.get(metric.__name__, 0)}"
        with open(self.results_csv_path, 'a') as f:
            f.write(row + "\n")

    def save_best_model(self, model, epoch, train_loss, valid_loss):
        """Save the best model weights as best.pt."""
        path = os.path.join(self.weights_dir, 'best.pt')
        torch.save(model, path)
        print(f'💾 Best model saved to {path}')
        return path

    def save_last_model(self, model, epoch):
        """Save the model weights for the current (last) epoch."""
        path = os.path.join(self.weights_dir, 'last.pt')
        torch.save(model, path)
        print(f'💾 Last model saved to {path}')
        return path


class DatasetManager:
    """Manages creation and configuration of training datasets."""

    def __init__(self, train_df, valid_df, test_df, class_ids, preprocessing_fn,
                 augment_data=False, batch_size=8, imgsz=640):
        """Initialize DatasetManager with dataframes and configuration."""
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.class_ids = class_ids
        self.preprocessing_fn = preprocessing_fn
        self.augment_data = augment_data
        self.batch_size = batch_size
        self.imgsz = imgsz

        self._create_datasets()
        self._create_dataloaders()

    def _create_datasets(self):
        """Create train, validation, and test datasets."""
        # Whether to include training augmentation
        if self.augment_data:
            training_augmentation = get_training_augmentation(self.imgsz)
        else:
            training_augmentation = get_validation_augmentation(self.imgsz)

        self.train_dataset = Dataset(
            self.train_df,
            augmentation=training_augmentation,
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.class_ids,
        )

        self.valid_dataset = Dataset(
            self.valid_df,
            augmentation=get_validation_augmentation(self.imgsz),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.class_ids,
        )

        # For visualizing progress
        self.valid_dataset_vis = Dataset(
            self.valid_df,
            augmentation=get_validation_augmentation(self.imgsz),
            classes=self.class_ids,
        )

        # Test dataset for evaluation
        self.test_dataset = Dataset(
            self.test_df,
            augmentation=get_validation_augmentation(self.imgsz),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.class_ids,
        )

        # Test dataset without preprocessing for visualization
        self.test_dataset_vis = Dataset(
            self.test_df,
            classes=self.class_ids,
        )

    def _create_dataloaders(self):
        """Create data loaders for training and validation."""
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

    def visualize_training_samples(self, logs_dir, class_colors):
        """Visualize training samples and save to logs directory."""
        print("\n" + "-" * 50)
        print("👀 GENERATING TRAINING SAMPLE VISUALIZATIONS")
        print("-" * 50)

        # Create a sample version dataset
        sample_dataset = Dataset(self.train_df,
                                 augmentation=get_training_augmentation(self.imgsz),
                                 classes=self.class_ids)

        # Loop through a few samples
        for i in range(5):
            try:
                # Get a random sample from dataset
                image, mask = sample_dataset[np.random.randint(0, len(self.train_df))]
                # Visualize and save to logs dir
                save_path = os.path.join(logs_dir, f'TrainingSample_{i}.png')
                visualize(save_path=save_path,
                          save_figure=True,
                          image=image,
                          mask=colorize_mask(mask, self.class_ids, class_colors))
            except:
                pass
class Trainer:
    """Handles the training loop with early stopping and learning rate scheduling."""

    def __init__(self, model, loss_function, metrics, optimizer, device, num_epochs,
                 train_loader, valid_loader, valid_dataset, valid_dataset_vis,
                 experiment_manager, class_ids, class_colors):
        """Initialize Trainer with training components."""
        self.model = model
        self.loss_function = loss_function
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_dataset = valid_dataset
        self.valid_dataset_vis = valid_dataset_vis
        self.experiment_manager = experiment_manager
        self.class_ids = class_ids
        self.class_colors = class_colors

        # Training state
        self.best_score = float('inf')
        self.best_epoch = 0
        self.since_best = 0
        self.since_drop = 0

        # Create training epochs
        self.train_epoch = TrainEpoch(
            model, loss_function, metrics, optimizer, device, verbose=True
        )
        self.valid_epoch = ValidEpoch(
            model, loss_function, metrics, device, verbose=True
        )

    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 60)
        print("🚀 STARTING SEMANTIC SEGMENTATION TRAINING")
        print("=" * 60)

        print("📋 Training Configuration:")
        print(f"   • Epochs: {self.num_epochs}")
        print(f"   • Device: {self.device}")
        print(f"   • Model: {type(self.model).__name__}")
        print(f"   • Loss: {self.loss_function.__name__}")
        print(f"   • Metrics: {[m.__name__ for m in self.metrics]}")
        print()

        try:
            # Training loop
            for e_idx in range(1, self.num_epochs + 1):
                print(f"\n📊 Epoch {e_idx}/{self.num_epochs}")
                print("-" * 40)

                # Go through an epoch for train, valid
                train_logs = self.train_epoch.run(self.train_loader)
                valid_logs = self.valid_epoch.run(self.valid_loader)

                # Print training metrics
                print(f"  📈 Train: {format_logs_pretty(train_logs)}")
                print(f"  ✅ Valid: {format_logs_pretty(valid_logs)}")

                # Log training metrics to CSV
                self.experiment_manager.log_metrics(e_idx, "train", train_logs)
                self.experiment_manager.log_metrics(e_idx, "valid", valid_logs)

                # Visualize a validation sample
                self._visualize_validation_sample(e_idx)

                # Save the model for the current epoch
                self.experiment_manager.save_last_model(self.model, e_idx)

                # Check for best model and handle early stopping
                should_continue = self._update_training_state(e_idx, train_logs, valid_logs)
                if not should_continue:
                    break

        except KeyboardInterrupt:
            print("⏹️ Training interrupted by user")
        except Exception as e:
            if 'CUDA out of memory' in str(e):
                print(f"⚠️ Not enough GPU memory for the provided parameters")
            self._log_error(e)
            raise Exception(f"❌ There was an issue with training!\n{e}")

        return self.best_epoch

    def _visualize_validation_sample(self, epoch):
        """Visualize a validation sample for the current epoch."""
        try:
            n = np.random.choice(len(self.valid_dataset_vis))
            # Get the image original image without preprocessing
            image_vis = self.valid_dataset_vis[n][0].numpy()
            # Get the preprocessed input for model prediction
            image, gt_mask = self.valid_dataset[n]
            gt_mask = gt_mask.squeeze().numpy()
            x_tensor = image.to(self.device).unsqueeze(0)
            # Make prediction
            pr_mask = self.model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            pr_mask = np.argmax(pr_mask, axis=0)

            # Visualize the colorized results locally
            save_path = os.path.join(self.experiment_manager.logs_dir, f'ValidResult_{epoch}.png')
            visualize(save_path=save_path,
                      save_figure=True,
                      image=image_vis,
                      ground_truth_mask=colorize_mask(gt_mask, self.class_ids, self.class_colors),
                      predicted_mask=colorize_mask(pr_mask, self.class_ids, self.class_colors))
        except Exception as e:
            print(f"⚠️ Could not visualize validation sample for epoch {epoch}: {e}")

    def _update_training_state(self, epoch, train_logs, valid_logs):
        """Update training state and handle early stopping."""
        # Get the loss values
        train_loss = [v for k, v in train_logs.items() if 'loss' in k.lower()][0]
        valid_loss = [v for k, v in valid_logs.items() if 'loss' in k.lower()][0]

        if valid_loss < self.best_score:
            # Update best
            self.best_score = valid_loss
            self.best_epoch = epoch
            self.since_best = 0
            print(f"🏆 New best epoch {epoch}")

            # Save the model
            self.experiment_manager.save_best_model(self.model, epoch, train_loss, valid_loss)
        else:
            # Increment the counters
            self.since_best += 1
            self.since_drop += 1
            print(f"📉 Model did not improve after epoch {epoch}")

        # Overfitting indication
        if train_loss < valid_loss:
            print(f"⚠️ Overfitting detected in epoch {epoch}")

        # Check if it's time to decrease the learning rate
        if (self.since_best >= 5 or train_loss <= valid_loss) and self.since_drop >= 5:
            self.since_drop = 0
            new_lr = self.optimizer.param_groups[0]['lr'] * 0.75
            self.optimizer.param_groups[0]['lr'] = new_lr
            print(f"🔄 Decreased learning rate to {new_lr:.6f} after epoch {epoch}")

        # Exit early if progress stops
        if self.since_best >= 10 and train_loss < valid_loss and self.since_drop >= 5:
            print("🛑 Training plateaued; stopping early")
            return False

        return True

    def _log_error(self, error):
        """Log training error to file."""
        print(f"📄 Error details saved to {self.experiment_manager.logs_dir}Error.txt")
        with open(os.path.join(self.experiment_manager.logs_dir, "Error.txt"), 'a') as file:
            file.write(f"Caught exception: {str(error)}\n")


class Evaluator:
    """Handles model evaluation and result visualization."""

    def __init__(self, model, loss_function, metrics, device, test_dataset, test_dataset_vis,
                 experiment_manager, class_ids, class_colors):
        """Initialize Evaluator with evaluation components."""
        self.model = model
        self.loss_function = loss_function
        self.metrics = metrics
        self.device = device
        self.test_dataset = test_dataset
        self.test_dataset_vis = test_dataset_vis
        self.experiment_manager = experiment_manager
        self.class_ids = class_ids
        self.class_colors = class_colors

        # Get original image dimensions
        self.original_width, self.original_height = Image.open(
            self.test_dataset_vis.dataframe.iloc[0]['Image']
        ).size

    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "-" * 50)
        print("🧪 EVALUATING MODEL ON TEST SET")
        print("-" * 50)

        # Create test dataloader
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Evaluate on the test set
        test_epoch = ValidEpoch(
            model=self.model,
            loss=self.loss_function,
            metrics=self.metrics,
            device=self.device,
        )

        try:
            # Empty cache from training
            torch.cuda.empty_cache()

            # Score on test set
            test_logs = test_epoch.run(test_loader)

            # Print test metrics
            print(f"  🧪 Test: {format_logs_pretty(test_logs)}")

            # Log test metrics to CSV
            self.experiment_manager.log_metrics("best", "test", test_logs)

        except Exception as e:
            # Catch the error
            print(f"ERROR: Could not calculate metrics")
            # Likely Memory
            if 'CUDA out of memory' in str(e):
                print(f"WARNING: Not enough GPU memory for the provided parameters")
            self._log_error(e)

    def visualize_results(self, num_samples=25):
        """Visualize test results."""
        print("\n" + "-" * 50)
        print(f"🎨 VISUALIZING {num_samples} TEST RESULTS")
        print("-" * 50)

        try:
            # Empty cache from testing
            torch.cuda.empty_cache()

            # Loop through samples
            for i in range(num_samples):
                # Get a random sample
                n = np.random.choice(len(self.test_dataset))
                # Get the image original image without preprocessing
                image_vis = self.test_dataset_vis[n][0].numpy()
                # Get the expected input for model
                image, gt_mask = self.test_dataset[n]
                gt_mask = gt_mask.squeeze().numpy()
                gt_mask = cv2.resize(gt_mask, 
                                     (self.original_width, self.original_height), 
                                     interpolation=cv2.INTER_NEAREST)
                # Colorize the ground truth mask
                gt_mask = colorize_mask(gt_mask, self.class_ids, self.class_colors)
                
                # Prepare sample
                x_tensor = image.to(self.device).unsqueeze(0)
                # Make prediction
                pr_mask = self.model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                pr_mask = np.argmax(pr_mask, axis=0)
                pr_mask = cv2.resize(pr_mask, 
                                     (self.original_width, self.original_height), 
                                     interpolation=cv2.INTER_NEAREST)
                # Colorize the predicted mask
                pr_mask = colorize_mask(pr_mask, self.class_ids, self.class_colors)

                try:
                    # Visualize the colorized results locally
                    save_path = os.path.join(self.experiment_manager.logs_dir, f'TestResult_{i}.png')
                    visualize(save_path=save_path,
                              save_figure=True,
                              image=image_vis,
                              ground_truth_mask=gt_mask,
                              predicted_mask=pr_mask)
                except:
                    pass
        except Exception as e:
            # Catch the error
            print(f"ERROR: Could not make predictions")
            # Likely Memory
            if 'CUDA out of memory' in str(e):
                print(f"WARNING: Not enough GPU memory for the provided parameters")
            self._log_error(e)

    def _log_error(self, error):
        """Log evaluation error to file."""
        print(f"📄 Error details saved to {self.experiment_manager.logs_dir}Error.txt")
        with open(os.path.join(self.experiment_manager.logs_dir, "Error.txt"), 'a') as file:
            file.write(f"Caught exception: {str(error)}\n")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """Parse command line arguments and run semantic segmentation training."""
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to YOLO data.yaml file')

    parser.add_argument('--pre_trained_path', type=str, default=None,
                        help='Path to pre-trained model of the same architecture')

    parser.add_argument('--encoder_name', type=str, default='resnet34',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--decoder_name', type=str, default='Unet',
                        help='The convolutional decoder')

    parser.add_argument('--metrics', type=str, nargs='+', default=get_segmentation_metrics(),
                        help='The metrics to evaluate the model')

    parser.add_argument('--loss_function', type=str, default='JaccardLoss',
                        help='The loss function to use to train the model')

    parser.add_argument('--freeze_encoder', type=float, default=0.0,
                        help='Freeze N% of the encoder [0 - 1]')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='The optimizer to use to train the model')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Starting learning rate')

    parser.add_argument('--augment_data', action='store_true',
                        help='Apply affine augmentations to training data')

    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Starting learning rate')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of samples per batch during training')

    parser.add_argument('--imgsz', type=int, default=640,
                        help='Length of the longest edge after resizing input images (must be divisible by 32)')

    args = parser.parse_args()

    # Validate and adjust imgsz to be divisible by 32
    if args.imgsz % 32 != 0:
        original_imgsz = args.imgsz
        args.imgsz = round(args.imgsz / 32) * 32
        print(f"WARNING: imgsz {original_imgsz} is not divisible by 32. Adjusted to {args.imgsz}")

    # Set output directory to "results" folder in the same directory as data.yaml
    output_dir = os.path.join(os.path.dirname(args.data_yaml), "results")

    try:
        # Main function for semantic segmentation training pipeline
        print("\n" + "=" * 60)
        print("🧠 SEMANTIC SEGMENTATION TRAINING PIPELINE")
        print("=" * 60)
        print("🔧 Initializing...")

        # Check for CUDA
        print(f"   • PyTorch: {torch.__version__}")
        print(f"   • CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
        print(f"   • Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print()

        # Whether to run on GPU or CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load and parse data configuration
        print("📂 Loading dataset configuration...")
        data_config = DataConfig(args.data_yaml)
        train_df, valid_df, test_df = data_config.get_split_dataframes()

        print("📊 Dataset Summary:")
        print(f"   • Training samples: {len(train_df)}")
        print(f"   • Validation samples: {len(valid_df)}")
        print(f"   • Test samples: {len(test_df)}")
        print(f"   • Classes: {data_config.class_names}")
        print()

        # Build model
        model_builder = ModelBuilder(
            encoder_name=args.encoder_name,
            decoder_name=args.decoder_name,
            num_classes=len(data_config.class_names),
            freeze_encoder=args.freeze_encoder,
            pre_trained_path=args.pre_trained_path
        )

        # Setup training configuration
        training_config = TrainingConfig(
            model=model_builder.model,
            loss_function_name=args.loss_function,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            metrics_list=args.metrics,
            class_ids=data_config.class_ids,
            device=device
        )

        # Setup experiment management
        experiment_manager = ExperimentManager(
            output_dir=output_dir,
            decoder_name=args.decoder_name,
            encoder_name=args.encoder_name,
            color_map=data_config.color_map,
            metrics=training_config.metrics
        )

        # Save dataframes
        experiment_manager.save_dataframes(train_df, valid_df, test_df)

        # Create datasets and dataloaders
        dataset_manager = DatasetManager(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            class_ids=data_config.class_ids,
            preprocessing_fn=model_builder.preprocessing_fn,
            augment_data=args.augment_data,
            batch_size=args.batch_size,
            imgsz=args.imgsz
        )

        # Visualize training samples
        dataset_manager.visualize_training_samples(
            experiment_manager.logs_dir, data_config.class_colors
        )

        # Train the model
        trainer = Trainer(
            model=model_builder.model,
            loss_function=training_config.loss_function,
            metrics=training_config.metrics,
            optimizer=training_config.optimizer,
            device=device,
            num_epochs=args.num_epochs,
            train_loader=dataset_manager.train_loader,
            valid_loader=dataset_manager.valid_loader,
            valid_dataset=dataset_manager.valid_dataset,
            valid_dataset_vis=dataset_manager.valid_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=data_config.class_ids,
            class_colors=data_config.class_colors
        )

        best_epoch = trainer.train()

        # Load best model for evaluation
        best_weights = os.path.join(experiment_manager.weights_dir, "best.pt")
        model = torch.load(best_weights)
        print(f"📥 Loaded best weights from {best_weights}")

        # Evaluate model
        evaluator = Evaluator(
            model=model,
            loss_function=training_config.loss_function,
            metrics=training_config.metrics,
            device=device,
            test_dataset=dataset_manager.test_dataset,
            test_dataset_vis=dataset_manager.test_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=data_config.class_ids,
            class_colors=data_config.class_colors
        )

        evaluator.evaluate()
        evaluator.visualize_results()

        print(f"💾 Best model saved to {experiment_manager.run_dir}")
        shutil.copyfile(best_weights, os.path.join(experiment_manager.run_dir, "Best_Model_and_Weights.pt"))

        print("✅ Training pipeline completed successfully!\n")

    except Exception as e:
        print(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()

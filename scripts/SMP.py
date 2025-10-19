import os
import sys
import json
import yaml
import glob
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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.amp
import torch.quantization

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.meter import AverageValueMeter

import albumentations as albu

try:
    from ultralytics.engine.results import Results, Boxes, Masks
except ImportError:
    print("Warning: `ultralytics` package not found. Prediction output will be raw numpy arrays.")
    Results, Boxes, Masks = None, None, None

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
            loss_items.append(f"{k:}: {v:.4f}")
        else:
            metric_items.append(f"{k:}: {v:.4f}")
    
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
        if not images_dir.endswith('images'):
            images_dir = os.path.join(images_dir, 'images')
        labels_dir = images_dir.replace('images', 'labels')
        
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
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True, scaler=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scaler = scaler

    def on_epoch_start(self):
        """Set model to training mode."""
        self.model.train()

    def batch_update(self, x, y):
        """Perform training batch update with loss computation and backpropagation."""
        self.optimizer.zero_grad()
        
        # Use autocast for mixed precision if scaler is provided
        if self.scaler is not None:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
            
            # Scale loss and backpropagate
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training without AMP
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


# ----------------------------------------------------------------------------------------------------------------------
# NEW: Main Model Class (Replaces ModelBuilder and integrates Predictor)
# ----------------------------------------------------------------------------------------------------------------------

class SemanticModel:
    """
    A class that encapsulates semantic segmentation model training and prediction,
    mimicking the Ultralytics API.
    """

    def __init__(self, model_path=None):
        """
        Initializes the SemanticModel.

        Args:
            model_path (str, optional): Path to a pre-trained .pt model.
                                      If None, a new model must be built via .train().
        """
        self.model = None
        self.name = None  # Stashed model name
        self.preprocessing_fn = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = 'segment'
        self.data_config = None
        self.class_names = []
        self.class_colors = []
        self.num_classes = 0
        self.imgsz = None
        self.color_map = {}

        # Prediction optimization settings
        self._optimized_model = None
        self._last_pred_config = {}

        if model_path:
            self.load(model_path)

    def load(self, model_path):
        """Loads a pre-trained model and associated metadata."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        print(f"Loading model from {model_path}...")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # --- NEW: Retrieve stashed metadata directly from the model object ---
        self.name = getattr(self.model, 'name', None)
        self.imgsz = getattr(self.model, 'imgsz', None)
        self.class_names = getattr(self.model, 'class_names', [])
        self.class_colors = getattr(self.model, 'class_colors', [])
        self.num_classes = getattr(self.model, 'num_classes', 0)
        self.color_map = getattr(self.model, 'color_map', {})
        # Load class_ids, fallback to a range based on num_classes
        self.class_ids = getattr(self.model, 'class_ids', list(range(self.num_classes)))

        if self.name:
            print(f"   • Retrieved stashed model name: {self.name}")
            print(f"   • Retrieved stashed imgsz: {self.imgsz}")
            print(f"   • Retrieved {self.num_classes} classes.")
        else:
            print("   • No stashed metadata found (likely an older model file).")

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

        # Infer preprocessing function from the stashed model name
        try:
            if not self.name:
                raise Exception("Model name not stashed.")
            encoder_name = self.name.split('-')[1]
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')
            print(f"Inferred preprocessing for encoder: {encoder_name}")
        except Exception as e:
            print(f"Warning: Could not infer preprocessing function from stashed name '{self.name}'. "
                  f"Using no-op. Error: {e}")
            self.preprocessing_fn = lambda x: x  # No-op

    def _build_model(self, encoder_name, decoder_name, num_classes, freeze_encoder, pre_trained_path):
        """Internal method to build a new model. (Logic from original ModelBuilder)"""
        # Validate options
        if encoder_name not in get_segmentation_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_segmentation_encoders()}")
        if decoder_name not in get_segmentation_decoders():
            raise Exception(f"ERROR: Decoder must be one of {get_segmentation_decoders()}")

        # Build model
        encoder_weights = 'imagenet'
        self.model = getattr(smp, decoder_name)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation='softmax2d',
        )

        # Create and stash the model name string
        self.name = f"{decoder_name}-{encoder_name}"
        self.model.name = self.name
        print(f"   • Stashing model name: {self.model.name}")

        print(f"   • Architecture: {encoder_name} → {decoder_name}")
        print(f"   • Output classes: {num_classes}")
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

        # Load pretrained weights
        if pre_trained_path:
            if not os.path.exists(pre_trained_path):
                print("⚠️ Pre-trained encoder path not found, using random initialization")
            else:
                pre_trained_model = torch.load(pre_trained_path, map_location='cpu')
                try:
                    encoder_name_loaded = pre_trained_model.name
                    state_dict = pre_trained_model.encoder.state_dict()
                except:
                    encoder_name_loaded = pre_trained_model.encoder.name
                    state_dict = pre_trained_model.encoder.encoder.state_dict()
                self.model.encoder.load_state_dict(state_dict, strict=True)
                print(f"📥 Loaded pre-trained weights from {encoder_name_loaded}")
        else:
            print("📥 Using ImageNet pretrained weights")

        # Freeze encoder
        num_params = len(list(self.model.encoder.parameters()))
        freeze_params = int(num_params * freeze_encoder)
        print(f"🧊 Freezing {freeze_encoder * 100}% of encoder weights ({freeze_params}/{num_params})")
        for idx, param in enumerate(self.model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False
        
        self.model.to(self.device)

    def train(self, data_yaml, encoder_name='resnet34', decoder_name='Unet',
              pre_trained_path=None, freeze_encoder=0.8,
              metrics=None, loss_function='JaccardLoss',
              optimizer='Adam', learning_rate=0.0001, augment_data=False,
              num_epochs=25, batch_size=8, imgsz=640, amp=False,
              output_dir=None, num_vis_samples=10):
        """
        Train the semantic segmentation model.
        (This method encapsulates the logic from SemanticTrain.py's main())
        """
        if metrics is None:
            metrics = ['iou_score', 'f1_score']

        print("\n" + "=" * 60)
        print("🚀 STARTING SEMANTIC SEGMENTATION TRAINING")
        print("=" * 60)
        
        # Store key info on self
        self.imgsz = imgsz

        # 1. Load Data Config
        print("📂 Loading dataset configuration...")
        self.data_config = DataConfig(data_yaml)
        train_df, valid_df, test_df = self.data_config.get_split_dataframes()
        self.class_names = self.data_config.class_names
        self.class_colors = self.data_config.class_colors
        self.num_classes = self.data_config.nc
        self.color_map = self.data_config.color_map

        print("📊 Dataset Summary:")
        print(f"   • Training samples: {len(train_df)}")
        print(f"   • Validation samples: {len(valid_df)}")
        print(f"   • Test samples: {len(test_df)}")
        print(f"   • Classes ({self.num_classes}): {self.class_names}")

        # 2. Build Model (if not already loaded)
        if self.model is None:
            print("🔧 Building new model...")
            self._build_model(
                encoder_name=encoder_name,
                decoder_name=decoder_name,
                num_classes=self.num_classes,
                freeze_encoder=freeze_encoder,
                pre_trained_path=pre_trained_path
            )
        else:
            print("🔧 Using pre-loaded model for training.")
            self.model.to(self.device)
            
        print("📝 Stamping metadata onto model object for saving...")
        self.model.imgsz = self.imgsz
        self.model.class_names = self.class_names
        self.model.class_colors = self.class_colors
        self.model.num_classes = self.num_classes
        self.model.color_map = self.color_map

        # 3. Setup Training Config
        training_config = TrainingConfig(
            model=self.model,
            loss_function_name=loss_function,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            metrics_list=metrics,
            class_ids=self.data_config.class_ids,
            device=self.device
        )

        # 4. Setup Experiment Manager
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(data_yaml), "results")

        experiment_manager = ExperimentManager(
            output_dir=output_dir,
            decoder_name=decoder_name,
            encoder_name=encoder_name,
            color_map=self.data_config.color_map,
            metrics=training_config.metrics
        )
        experiment_manager.save_dataframes(train_df, valid_df, test_df)

        # 5. Setup Datasets
        dataset_manager = DatasetManager(
            train_df=train_df, valid_df=valid_df, test_df=test_df,
            class_ids=self.data_config.class_ids,
            preprocessing_fn=self.preprocessing_fn,  # Use the one from self
            augment_data=augment_data,
            batch_size=batch_size,
            imgsz=self.imgsz  # Use the one from self
        )
        dataset_manager.visualize_training_samples(
            experiment_manager.logs_dir, self.data_config.class_colors
        )

        # 6. Run Trainer
        trainer = Trainer(
            model=self.model,
            loss_function=training_config.loss_function,
            metrics=training_config.metrics,
            optimizer=training_config.optimizer,
            device=self.device,
            num_epochs=num_epochs,
            train_loader=dataset_manager.train_loader,
            valid_loader=dataset_manager.valid_loader,
            valid_dataset=dataset_manager.valid_dataset,
            valid_dataset_vis=dataset_manager.valid_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=self.data_config.class_ids,
            class_colors=self.data_config.class_colors,
            amp_enabled=amp
        )
        trainer.train()

        # 7. Evaluate
        print("\n📥 Loading best weights for evaluation...")
        best_weights = os.path.join(experiment_manager.weights_dir, "best.pt")
        # Load best model back into self.model (this will also update self.name)
        self.load(best_weights) 
        
        evaluator = Evaluator(
            model=self.model,
            loss_function=training_config.loss_function,
            metrics=training_config.metrics,
            device=self.device,
            test_dataset=dataset_manager.test_dataset,
            test_dataset_vis=dataset_manager.test_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=self.data_config.class_ids,
            class_colors=self.data_config.class_colors,
            num_vis_samples=num_vis_samples
        )
        evaluator.evaluate()
        evaluator.visualize_results()

        print(f"\n✅ Training pipeline completed! Best model at {best_weights}")
        self.model.eval()
        # Clear any optimized model from previous runs
        self._optimized_model = None
        
        return best_weights
    
    # --- Evalutation Methods ---
    
    def eval(self, data_yaml, split='test', num_vis_samples=10, output_dir=None,
             loss_function='JaccardLoss', metrics=None):
        """
        Run standalone evaluation on a trained model.

        Args:
            data_yaml (str): Path to the data.yaml file.
            split (str): Data split to evaluate on ('train', 'val', or 'test').
            num_vis_samples (int): Number of result images to save.
            output_dir (str, optional): Directory to save results.
                If None, creates a new dir next to the model's 'weights' dir.
            loss_function (str): Name of the loss function to use for eval.
            metrics (list[str], optional): List of metric names to calculate.
        """
        if self.model is None:
            raise Exception("Model is not loaded. Load a model first.")
        
        if self.imgsz is None:
            raise Exception("Model `imgsz` is not set. Load a model trained "
                            "with this framework or provide `imgsz` manually.")

        if metrics is None:
            metrics = ['iou_score', 'f1_score']

        print("\n" + "=" * 60)
        print(f"🚀 STARTING STANDALONE EVALUATION ON '{split}' SPLIT")
        print("=" * 60)

        # 1. Load Data
        print(f"📂 Loading dataset from {data_yaml}...")
        data_config = DataConfig(data_yaml)
        all_dfs = data_config.get_split_dataframes()
        
        if split == 'train':
            eval_df = all_dfs[0]
        elif split == 'val':
            eval_df = all_dfs[1]
        elif split == 'test':
            eval_df = all_dfs[2]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        if eval_df.empty:
            print(f"⚠️ No data found for split: {split}. Exiting.")
            return

        print(f"   • Evaluating on {len(eval_df)} samples from '{split}' split.")

        # 2. Setup Datasets
        print("📦 Creating evaluation datasets...")
        eval_dataset = Dataset(
            eval_df,
            augmentation=get_validation_augmentation(self.imgsz),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.class_ids,
        )
        eval_dataset_vis = Dataset(
            eval_df,
            augmentation=get_validation_augmentation(self.imgsz),
            classes=self.class_ids,
        )
        
        # 3. Setup Metrics and Loss
        print("⚙️ Configuring metrics...")
        try:
            loss_fn = getattr(smp.losses, loss_function)(mode='multiclass').to(self.device)
            loss_fn.__name__ = loss_fn._get_name()
            if 'ignore_index' in inspect.signature(loss_fn.__init__).parameters:
                loss_fn.ignore_index = 0  # Ignore background
            
            metric_fns = [TorchMetic(getattr(smp.metrics, m)) for m in metrics]
            print(f"   • Loss: {loss_function}")
            print(f"   • Metrics: {metrics}")
        except Exception as e:
            print(f"❌ Error setting up metrics/loss: {e}")
            return

        # 4. Setup Experiment Manager
        if output_dir is None:
            # Place results in a new 'eval' dir at the same level as 'weights'
            model_base_dir = os.path.dirname(os.path.dirname(self.model_path))
            eval_run_name = f"eval_{split}_{get_now()}"
            output_dir = os.path.join(model_base_dir, eval_run_name)
            print(f"   • No output_dir provided. Saving results to: {output_dir}")

        decoder_name, encoder_name = "Model", "eval" # Defaults
        if self.name and '-' in self.name:
             decoder_name, encoder_name = self.name.split('-', 1)
        
        experiment_manager = ExperimentManager(
            output_dir=output_dir,
            decoder_name=decoder_name,
            encoder_name=encoder_name,
            color_map=self.color_map,
            metrics=metric_fns
        )
        # Save the dataframe used for this eval
        eval_df.to_csv(os.path.join(experiment_manager.logs_dir, f"{split}_data.csv"), index=False)

        # 5. Instantiate and Run Evaluator
        print("🚀 Handing off to Evaluator...")
        evaluator = Evaluator(
            model=self.model,
            loss_function=loss_fn,
            metrics=metric_fns,
            device=self.device,
            test_dataset=eval_dataset,
            test_dataset_vis=eval_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=self.class_ids,
            class_colors=self.class_colors,
            num_vis_samples=num_vis_samples
        )

        evaluator.evaluate()
        evaluator.visualize_results()
        
        # Plotting metrics won't work well here as it's not run over epochs
        # but the CSV with final scores will be saved.
        print(f"\n✅ Evaluation complete! Results saved to {experiment_manager.run_dir}")

    # --- Prediction Methods ---
    
    def __call__(self, source, **kwargs):
        """Alias for self.predict()"""
        return self.predict(source, **kwargs)

    def _load_image(self, path):
        """Loads a single image from a file path."""
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not read image from path: {path}, skipping.")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _tensor_to_numpy(self, tensor):
        """Converts a torch.Tensor to a numpy.ndarray (H, W, C)."""
        tensor = tensor.cpu()
        if tensor.dtype == torch.half:
            tensor = tensor.float()
        
        img_np = tensor.numpy()

        # Handle different tensor formats
        if img_np.ndim == 3:
            if img_np.shape[0] in [1, 3]:  # (C, H, W)
                if img_np.shape[0] == 1:  # Grayscale to RGB
                    img_np = np.concatenate([img_np] * 3, axis=0)
                return img_np.transpose(1, 2, 0)
            elif img_np.shape[2] in [1, 3]:  # (H, W, C)
                if img_np.shape[2] == 1:  # Grayscale to RGB
                    img_np = np.concatenate([img_np] * 3, axis=2)
                return img_np
        
        elif img_np.ndim == 4:
            if img_np.shape[1] in [1, 3]:  # (B, C, H, W)
                if img_np.shape[1] == 1:
                    img_np = np.concatenate([img_np] * 3, axis=1)
                return img_np.transpose(0, 2, 3, 1)
            elif img_np.shape[3] in [1, 3]:  # (B, H, W, C)
                if img_np.shape[3] == 1:
                    img_np = np.concatenate([img_np] * 3, axis=3)
                return img_np

        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    def _normalize_source(self, source):
        """
        Normalizes diverse input sources into a list of (image, path) tuples.
        Returns: (list[np.ndarray], list[str])
        """
        images, paths = [], []
        
        if isinstance(source, str):
            # Single image path
            img = self._load_image(source)
            if img is not None:
                images, paths = [img], [source]
        
        elif isinstance(source, np.ndarray):
            # Handle all numpy array formats
            
            # Ensure images are uint8. Albumentations can be picky.
            if source.dtype != np.uint8:
                print(f"Warning: Numpy array is dtype {source.dtype}, not uint8. "
                      "Attempting to convert. This may cause issues if data is not 0-255.")
                if np.max(source) <= 1.0 and np.min(source) >= 0.0:
                    source = (source * 255).astype(np.uint8)  # Handle float 0-1
                else:
                    source = source.astype(np.uint8)  # Hope for the best

            if source.ndim == 3:
                if source.shape[0] in [1, 3]:  # (C, H, W)
                    if source.shape[0] == 1:
                        source = np.concatenate([source] * 3, axis=0)
                    img = source.transpose(1, 2, 0)  # (H, W, C)
                elif source.shape[2] in [1, 3]:  # (H, W, C)
                    if source.shape[2] == 1:
                        source = np.concatenate([source] * 3, axis=2)
                    img = source
                else:
                    raise ValueError(f"Unsupported 3D numpy array shape: {source.shape}. "
                                     "Expected (H, W, C) or (C, H, W).")
                images, paths = [img], ["image.jpg"]
            
            elif source.ndim == 4:
                if source.shape[1] in [1, 3]:  # (B, C, H, W)
                    if source.shape[1] == 1:
                        source = np.concatenate([source] * 3, axis=1)
                    img_batch = source.transpose(0, 2, 3, 1) # (B, H, W, C)
                elif source.shape[3] in [1, 3]:  # (B, H, W, C)
                    if source.shape[3] == 1:
                        source = np.concatenate([source] * 3, axis=3)
                    img_batch = source
                else:
                    raise ValueError(f"Unsupported 4D numpy array shape: {source.shape}. "
                                     "Expected (B, H, W, C) or (B, C, H, W).")
                
                images = [img_batch[i] for i in range(img_batch.shape[0])]
                paths = [f"image_{i}.jpg" for i in range(len(images))]
            # --- END MODIFIED BLOCK ---
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {source.ndim}. Expected 3 or 4.")
        
        elif isinstance(source, list):
            # List of paths or images
            if not source:
                return [], []
            
            if isinstance(source[0], str):  # List of paths
                for p in source:
                    img = self._load_image(p)
                    if img is not None:
                        images.append(img)
                        paths.append(p)
            
            elif isinstance(source[0], np.ndarray):  # List of images
                # Recursively normalize each image in the list
                normalized_list = [self._normalize_source(img) for img in source]
                images = [item[0][0] for item in normalized_list if item[0]]
                paths = [item[1][0] for item in normalized_list if item[1]]
            
            else:
                raise TypeError(f"Unsupported list element type: {type(source[0])}")
        
        elif isinstance(source, torch.Tensor):
            # Single image or batch as torch.Tensor
            img_np = self._tensor_to_numpy(source)
            # After conversion, img_np is (H,W,C) or (B,H,W,C), so we can
            # recursively call this function to handle it as a numpy array.
            return self._normalize_source(img_np)

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
            
        return images, paths

    def predict(self, source, confidence_threshold=0.5, 
                use_fp16=True, compile_model=True, use_int8=False,
                imgsz=None):
        """
        Run inference on a variety of sources.

        Args:
            source (str, np.ndarray, torch.Tensor, list): Input source. Can be:
                - str: Path to a single image.
                - np.ndarray: A (H, W, 3) image or (B, H, W, 3) batch.
                - torch.Tensor: A (C, H, W), (H, W, C) image or
                                  (B, C, H, W), (B, H, W, C) batch.
                - list[str]: A list of image paths.
                - list[np.ndarray]: A list of (H, W, 3) images.
            confidence_threshold (float): Min confidence to keep a pixel's class.
            use_fp16 (bool): Use FP16 precision.
            compile_model (bool): Use torch.compile.
            use_int8 (bool): Use INT8 quantization.
            imgsz (int, optional): Image size for preprocessing. Uses model's
                                   training size if None.

        Returns:
            (ultralytics.engine.results.Results or list): 
            - A single Results object if the input was a single item.
            - A list of Results objects if the input was a batch or list.
        """
        if self.model is None:
            raise Exception("Model is not loaded. Load a model first.")

        # --- 1. Setup ---
        # Use training imgsz if not provided
        if imgsz is None:
            imgsz = self.imgsz if self.imgsz else 640  # Default to 640

        # Prepare the optimized model (re-optimize if config changed)
        pred_config = (use_fp16, compile_model, use_int8)
        if self._optimized_model is None or self._last_pred_config != pred_config:
            print("Initializing/Optimizing model for prediction...")
            self._prepare_optimized_model(use_fp16, compile_model, use_int8)
            self._last_pred_config = pred_config

        # Prepare data preprocessors
        val_aug = get_validation_augmentation(imgsz)
        if self.preprocessing_fn is None:
            print("Warning: `preprocessing_fn` not set. Using no-op.")
            self.preprocessing_fn = lambda x: x
        preproc = get_preprocessing(self.preprocessing_fn)

        # --- 2. Normalize Input ---
        # `is_batch` checks if the *original* input was a list or batch,
        # to decide whether to return a list or a single item.
        is_batch = isinstance(source, list) or \
                   (isinstance(source, (np.ndarray, torch.Tensor)) and source.ndim == 4)
                   
        images, paths = self._normalize_source(source)
        
        if not images:
            return [] if is_batch else None

        # --- 3. Preprocess Batch ---
        preprocessed_tensors = []
        orig_shapes = []
        for img in images:
            orig_shapes.append(img.shape)
            augmented = val_aug(image=img)
            preprocessed = preproc(image=augmented['image'], mask=augmented['image']) # mask is dummy
            preprocessed_tensors.append(torch.from_numpy(preprocessed['image']))

        batch_tensor = torch.stack(preprocessed_tensors).to(self.device)

        # Handle data type for fp16/int8
        if self._last_pred_config[0] and not self._last_pred_config[2]:  # FP16 on, INT8 off
            batch_tensor = batch_tensor.half()
        
        # --- 4. Run Inference ---
        with torch.no_grad():
            pred_logits = self._optimized_model(batch_tensor)
        
        # Post-process logits
        pred_classes = self._process_logits(pred_logits, confidence_threshold)
        
        # Squeeze batch dim and move to CPU
        mask_arrays_aug = [pred_classes[i].cpu().numpy().astype(np.uint8) for i in range(len(images))]
        
        # --- 5. Post-process Results ---
        results_list = []
        for i, mask_aug in enumerate(mask_arrays_aug):
            h, w = orig_shapes[i][:2]
            # Resize mask from (imgsz, imgsz) back to (h, w)
            mask_array = cv2.resize(mask_aug, (w, h), interpolation=cv2.INTER_NEAREST)
            
            results_list.append(
                self._post_process(mask_array, orig_shapes[i], paths[i], images[i])
            )

        # Return single item or list based on original input type
        return results_list if is_batch else results_list[0]

    def _process_logits(self, pred_logits, confidence_threshold):
        """Applies softmax, max, and thresholding to raw logits."""
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_confidence, pred_class = torch.max(pred_probs, dim=1)
        pred_class = torch.where(
            pred_confidence >= confidence_threshold,
            pred_class,
            torch.zeros_like(pred_class) # Set to background
        )
        return pred_class

    def _post_process(self, mask_array, orig_shape, path, orig_img):
        """Converts a numpy mask array into an Ultralytics Results object."""
        if Results is None:
            print("Warning: `ultralytics` not installed. Returning raw numpy mask.")
            return mask_array

        h, w = orig_shape[:2]
        
        # 1. Create Masks object
        # We need to convert (H, W) class index mask to (NumClasses, H, W) one-hot mask
        one_hot_mask = np.zeros((self.num_classes, h, w), dtype=np.uint8)
        
        present_classes = np.unique(mask_array)
        
        for c in present_classes:
            if c == 0: 
                continue  # Skip background
            if c >= self.num_classes: 
                continue  # Safety check
            
            one_hot_mask[c][mask_array == c] = 1

        # Filter out empty masks
        non_empty_indices = [int(c) for c in present_classes if c != 0 and c < self.num_classes]
        
        if not non_empty_indices:
            # No detections, return empty Results
            return [
                Results(
                    orig_img=orig_img,
                    path=path,
                    names=dict(enumerate(self.class_names)),
                    boxes=Boxes(torch.empty(0, 6), (h, w)) if Boxes is not None else None,
                    masks=Masks(torch.empty(0, h, w), (h, w)) if Masks is not None else None
                )
            ]

        final_masks_tensor = torch.from_numpy(one_hot_mask[non_empty_indices])
        
        ult_masks = Masks(final_masks_tensor, (h, w)) if Masks is not None else None

        # 2. Create Boxes object (dummy boxes from masks)
        ult_boxes = None
        if Boxes is not None and ult_masks is not None:
            try:
                # Use .from_mask (may not exist in all versions, hence the try/except)
                if hasattr(Boxes, 'from_mask'):
                    ult_boxes = Boxes.from_mask(ult_masks.data, orig_shape)
                else: 
                    # Manual fallback to get bounding boxes
                    boxes_xyxy = []
                    for mask in ult_masks.data:
                        pos = torch.where(mask)
                        if pos[0].shape[0] > 0:
                            xmin, ymin = pos[1].min(), pos[0].min()
                            xmax, ymax = pos[1].max(), pos[0].max()
                            boxes_xyxy.append([xmin, ymin, xmax, ymax])
                        else:
                            boxes_xyxy.append([0, 0, 0, 0])
                    ult_boxes = Boxes(torch.tensor(boxes_xyxy), (h, w))
                
                # We need to add class and conf
                box_data = ult_boxes.data
                # We don't have per-instance confidence, so use 1.0
                conf = torch.ones(len(non_empty_indices), dtype=torch.float)
                cls = torch.tensor(non_empty_indices, dtype=torch.float)
                
                # Combine xyxy, conf, cls
                final_box_data = torch.cat([
                    box_data[:, :4],
                    conf.unsqueeze(1),
                    cls.unsqueeze(1)
                ], dim=1)
                
                ult_boxes = Boxes(final_box_data, (h, w))
                
            except Exception as e:
                # Fallback if Boxes.from_mask fails (e.g., empty masks)
                print(f"Warning: Could not generate boxes from masks: {e}")
                ult_boxes = Boxes(torch.empty(0, 6), (h, w))

        return [
            Results(
                orig_img=orig_img,
                path=path,
                names=dict(enumerate(self.class_names)),
                boxes=ult_boxes,
                masks=ult_masks
            )
        ]

    # --- Optimization Methods (from Predictor) ---
    
    def _prepare_optimized_model(self, use_fp16, compile_model, use_int8):
        """Applies optimizations (compile, fp16, int8) to a copy of the model."""
        # Start from a fresh copy of the trained model
        # Use deepcopy to avoid modifying the original model
        self._optimized_model = self.model
        
        # Move to device and set to eval mode
        self._optimized_model.to(self.device)
        self._optimized_model.eval()

        # Compile model
        if compile_model and hasattr(torch, 'compile'):
            try:
                self._optimized_model = torch.compile(self._optimized_model, mode='max-autotune')
                print("🚀 Model compiled for optimized inference")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
        
        # Use INT8 quantization (highest priority)
        if use_int8 and 'cuda' in self.device:
            self._quantize_model()
        # Use half precision if requested (not compatible with INT8)
        elif use_fp16 and 'cuda' in self.device:
            self._optimized_model = self._optimized_model.half()
            print("⚡ Using FP16 precision")

    def _quantize_model(self):
        """Apply INT8 quantization to the `_optimized_model`."""
        try:
            self._optimized_model.eval()
            self._optimized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self._optimized_model, inplace=True)
            self._calibrate_quantization()
            torch.quantization.convert(self._optimized_model, inplace=True)
            print("✅ Model successfully quantized to INT8")
        except Exception as e:
            print(f"⚠️ INT8 quantization failed, falling back: {e}")
            # Reset to non-quantized model
            self._optimized_model = self.model
            self._optimized_model.to(self.device).eval()

    def _calibrate_quantization(self):
        """Calibrate quantization with dummy data."""
        # Use a dummy input based on imgsz
        size = self.imgsz if self.imgsz else 640
        dummy_input = torch.randn(1, 3, size, size).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self._optimized_model(dummy_input)


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
        self.run = f"{get_now()}_{self.encoder_name}_{self.decoder_name}"

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
        train_df.to_csv(os.path.join(self.logs_dir, "Training.csv"), index=False)
        valid_df.to_csv(os.path.join(self.logs_dir, "Validation.csv"), index=False)
        test_df.to_csv(os.path.join(self.logs_dir, "Testing.csv"), index=False)

    def log_metrics(self, epoch, phase, logs):
        """Log metrics to CSV file."""
        row = f"{epoch},{phase},{logs.get('loss', 0)}"
        for metric in self.metrics:
            row += f",{logs.get(metric.__name__, 0)}"
        with open(self.results_csv_path, 'a') as f:
            f.write(row + "\n")
            
    def plot_metrics(self):
        """
        Reads the results.csv file and generates line plots for all
        metrics, saving the plot to the logs directory.
        """
        print(f"\n📈 Generating metrics plot from {self.results_csv_path}...")
        
        try:
            # 1. Read the data
            if not os.path.exists(self.results_csv_path):
                print(f"⚠️ Cannot plot metrics. File not found: {self.results_csv_path}")
                return
                
            data = pd.read_csv(self.results_csv_path)
            
            if data.empty:
                print(f"⚠️ Cannot plot metrics. File is empty: {self.results_csv_path}")
                return

            # 2. Identify metrics to plot
            # Get metric names from the class instance, not just the file
            metric_names = [m.__name__ for m in self.metrics]
            all_metrics_to_plot = ['loss'] + metric_names
            
            # 3. Separate train and valid data
            train_data = data[data['phase'] == 'train']
            valid_data = data[data['phase'] == 'valid']

            # 4. Create a dynamic subplot grid (2 rows, multiple columns)
            num_metrics = len(all_metrics_to_plot)
            num_rows = 2
            num_cols = (num_metrics + num_rows - 1) // num_rows  # Ceiling division
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * num_rows))
            
            # Flatten axes array for easy iteration, handling 1-row case
            if num_rows == 1:
                if num_cols == 1:
                    axes = [axes]  # Make it iterable if 1x1
                else:
                    axes = axes.flatten()  # 1xN
            else:
                axes = axes.flatten()  # NxN

            # 5. Iterate and plot each metric
            for i, metric in enumerate(all_metrics_to_plot):
                ax = axes[i]
                
                # Check if metric exists in the data
                if metric not in train_data.columns or metric not in valid_data.columns:
                    print(f"   - Skipping '{metric}', not found in CSV columns.")
                    continue
                
                # Plot train data
                ax.plot(train_data['epoch'], 
                        train_data[metric], 
                        label='Train', 
                        marker='o', 
                        linestyle='-', 
                        markersize=4)
                
                # Plot valid data
                ax.plot(valid_data['epoch'], 
                        valid_data[metric], 
                        label='Valid', 
                        marker='x', 
                        linestyle='--', 
                        markersize=5)
                
                # Set titles and labels
                ax.set_title(f'{metric.replace("_", " ").title()} Over Epochs')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, linestyle=':')

            # 6. Clean up unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # 7. Save the figure
            plt.tight_layout()
            save_path = os.path.join(self.logs_dir, "metrics_plot.png")
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory
            
            print(f"📊 Metrics plot saved to {save_path}")

        except Exception as e:
            print(f"⚠️ Could not generate metrics plot. Error: {e}")
            print(traceback.format_exc())

    def save_best_model(self, model):
        """Save the best model weights as best.pt."""
        path = os.path.join(self.weights_dir, 'best.pt')
        torch.save(model, path)
        print(f'💾 Best model saved to {path}')
        return path

    def save_last_model(self, model):
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
            augmentation=get_validation_augmentation(self.imgsz),
            classes=self.class_ids,
        )

    def _create_dataloaders(self):
        """Create data loaders for training and validation."""
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=torch.cuda.is_available()
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=1, 
            shuffle=False, 
            pin_memory=torch.cuda.is_available()
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            pin_memory=torch.cuda.is_available()
        )

    def visualize_training_samples(self, logs_dir, class_colors):
        """Visualize training samples and save to logs directory."""
        print("\n" + "=" * 60)
        print("👀 GENERATING TRAINING SAMPLE VISUALIZATIONS")
        print("=" * 60)

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
                          training_sample=colorize_mask(mask, self.class_ids, class_colors))
            except:
                pass
            
            
class Trainer:
    """Handles the training loop with early stopping and learning rate scheduling."""

    def __init__(self, model, loss_function, metrics, optimizer, device, num_epochs,
                 train_loader, valid_loader, valid_dataset, valid_dataset_vis,
                 experiment_manager, class_ids, class_colors, amp_enabled=False):
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
        self.amp_enabled = amp_enabled

        # Initialize AMP scaler if enabled
        self.scaler = torch.amp.GradScaler() if amp_enabled else None

        # Training state
        self.best_score = float('inf')
        self.best_epoch = 0
        self.since_best = 0
        self.since_drop = 0

        # Create training epochs
        self.train_epoch = TrainEpoch(
            model, loss_function, metrics, optimizer, device, verbose=True, scaler=self.scaler
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
        print(f"   • AMP: {'Enabled' if self.amp_enabled else 'Disabled'}")
        print()

        try:
            # Training loop
            for e_idx in range(1, self.num_epochs + 1):
                print(f"\n🦖 Epoch {e_idx}/{self.num_epochs}")
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

                # Check for best model and handle early stopping
                should_continue = self._update_training_state(e_idx, train_logs, valid_logs)
                
                # Save the model for the current epoch
                self.experiment_manager.save_last_model(self.model)
                
                # Check to exit early
                if not should_continue:
                    break

        except KeyboardInterrupt:
            print("⏹️ Training interrupted by user")
        except Exception as e:
            if 'CUDA out of memory' in str(e):
                print("⚠️ Not enough GPU memory for the provided parameters")
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
            self.experiment_manager.save_best_model(self.model)
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
            file.write(f"Caught exception: {str(error)}\n{traceback.format_exc()}\n")


class Evaluator:
    """Handles model evaluation and result visualization."""

    def __init__(self, model, loss_function, metrics, device, test_dataset, test_dataset_vis,
                 experiment_manager, class_ids, class_colors, num_vis_samples=10):
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
        self.num_vis_samples = num_vis_samples

        # Get original image dimensions
        self.original_width, self.original_height = Image.open(
            self.test_dataset_vis.images_fps[0]
        ).size

    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "=" * 60)
        print("🧪 EVALUATING MODEL ON TEST SET")
        print("=" * 60)

        # Create test dataloader
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            pin_memory=torch.cuda.is_available()
        )

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
            print("ERROR: Could not calculate metrics")
            # Likely Memory
            if 'CUDA out of memory' in str(e):
                print("WARNING: Not enough GPU memory for the provided parameters")
            self._log_error(e)

    def visualize_results(self):
        """Visualize test results."""
        print("\n" + "=" * 60)
        print(f"🎨 VISUALIZING {self.num_vis_samples} TEST RESULTS")
        print("=" * 60)
        
        try:
            # Plot metrics after visualizations
            self.experiment_manager.plot_metrics()
        except Exception as e:
            print(f"⚠️ Could not plot metrics: {e}")

        try:
            # Empty cache from testing
            torch.cuda.empty_cache()

            # Loop through samples
            for i in range(self.num_vis_samples):
                # Get a random sample
                n = np.random.choice(len(self.test_dataset))
                # Get the original image and mask without preprocessing
                image_vis, gt_mask_vis = self.test_dataset_vis[n]
                image_vis = image_vis.numpy()
                gt_mask_vis = gt_mask_vis.numpy().squeeze()

                # Get dimensions of the current image
                current_width, current_height = Image.open(self.test_dataset_vis.images_fps[n]).size

                # Colorize the ground truth mask
                gt_mask = colorize_mask(gt_mask_vis, self.class_ids, self.class_colors)

                # Get the preprocessed input for model prediction
                image, _ = self.test_dataset[n]
                x_tensor = image.to(self.device).unsqueeze(0)
                # Make prediction
                pr_mask = self.model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                pr_mask = np.argmax(pr_mask, axis=0)
                # Colorize the predicted mask (no resize needed since image_vis is already at imgsz)
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
            print("ERROR: Could not make predictions")
            # Likely Memory
            if 'CUDA out of memory' in str(e):
                print("WARNING: Not enough GPU memory for the provided parameters")
            self._log_error(e)

    def _log_error(self, error):
        """Log evaluation error to file."""
        print(f"📄 Error details saved to {self.experiment_manager.logs_dir}Error.txt")
        with open(os.path.join(self.experiment_manager.logs_dir, "Error.txt"), 'a') as file:
            file.write(f"Caught exception: {str(error)}\n{traceback.format_exc()}\n")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """Parse command line arguments and run semantic segmentation training."""
    # Get available options for help text
    available_encoders = get_segmentation_encoders()
    available_decoders = get_segmentation_decoders()
    available_losses = get_segmentation_losses()
    available_metrics = get_segmentation_metrics()
    available_optimizers = get_segmentation_optimizers()
    
    # Build help strings safely
    encoder_list = ', '.join(available_encoders)
    decoder_list = ', '.join(available_decoders)
    loss_list = ', '.join(available_losses)
    metrics_list = ', '.join(available_metrics)
    optimizer_list = ', '.join(available_optimizers)
    
    encoder_help = f"Name of the encoder backbone to use ({encoder_list})"
    decoder_help = f"Name of the decoder architecture to use ({decoder_list})"
    loss_help = f"Loss function to use during training ({loss_list})"
    metrics_help = f"List of metrics to use during training ({metrics_list})"
    optimizer_help = f"Optimizer to use during training ({optimizer_list})"
    
    # Build epilog text
    epilog_text = """
    Available Options:
    Encoders: """ + encoder_list + """
    Decoders: """ + decoder_list + """
    Loss Functions: """ + loss_list + """
    Metrics: """ + metrics_list + """
    Optimizers: """ + optimizer_list + """
    """
    
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text
    )

    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to YOLO data.yaml file')

    parser.add_argument('--pre_trained_path', type=str, default=None,
                        help='Path to pre-trained model of the same architecture')

    parser.add_argument('--encoder_name', type=str, default='resnet34',
                        help=encoder_help)

    parser.add_argument('--decoder_name', type=str, default='Unet',
                        help=decoder_help)

    parser.add_argument('--metrics', type=str, nargs='+', default=get_segmentation_metrics(),
                        help=metrics_help)

    parser.add_argument('--loss_function', type=str, default='JaccardLoss',
                        help=loss_help)

    parser.add_argument('--freeze_encoder', type=float, default=0.80,
                        help='Freeze N percent of the encoder [0 - 1]')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help=optimizer_help)

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

    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision training for faster training and reduced memory usage')

    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 quantization for maximum inference speed (applied to saved models)')

    parser.add_argument('--num_vis_samples', type=int, default=10,
                        help='Number of test samples to visualize during evaluation')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # Check imgsz
    if args.imgsz % 32 != 0:
        new_size = (args.imgsz // 32) * 32
        print(f"⚠️ imgsz must be divisible by 32. Adjusting from {args.imgsz} to {new_size}.")
        args.imgsz = new_size

    # Validate AMP
    if args.amp and not torch.cuda.is_available():
        print("⚠️ AMP requires CUDA. Disabling AMP.")
        args.amp = False

    try:
        # Initialize the model (it will be built inside .train())
        model = SemanticModel()
        
        # Start training
        model.train(
            data_yaml=args.data_yaml,
            encoder_name=args.encoder_name,
            decoder_name=args.decoder_name,
            pre_trained_path=args.pre_trained_path,
            freeze_encoder=args.freeze_encoder,
            metrics=args.metrics,
            loss_function=args.loss_function,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            augment_data=args.augment_data,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            amp=args.amp,
            num_vis_samples=args.num_vis_samples
        )

    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()

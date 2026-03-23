import os
import sys
import uuid
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
import torch.amp
import torch.quantization
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

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


def parse_ignore_index(ignore_index_input):
    """
    Parse ignore_index parameter from various input formats.
    
    Args:
        ignore_index_input (str, int, or None): Index to ignore, can be string, int, or None
    
    Returns:
        int or None: Single integer to ignore, or None if no index to ignore
    """
    if ignore_index_input is None:
        return None
    
    # If already an integer, return it directly
    if isinstance(ignore_index_input, int):
        return ignore_index_input if ignore_index_input >= 0 else None
    
    # If it's a string, try to parse it
    if isinstance(ignore_index_input, str):
        ignore_index_str = ignore_index_input.strip()
        if ignore_index_str == "":
            return None
            
        try:
            # Convert to integer
            ignore_index = int(ignore_index_str)
            return ignore_index if ignore_index >= 0 else None
        except ValueError as e:
            print(f"⚠️ Warning: Failed to parse ignore_index '{ignore_index_str}': {e}")
            return None
    
    # For any other type, return None
    print(f"⚠️ Warning: Unsupported ignore_index type: {type(ignore_index_input)}")
    return None


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
    

def validate_augmentation(dataframe, class_ids, imgsz, save_path=None):
    """
    Validate that augmentations are working correctly by checking a sample.
    Shows original and augmented image/mask pairs to verify alignment.
    Runs three times and saves separate files.
    """
    print("🔍 Validating augmentation pipeline...")
    
    # Create datasets with and without augmentation
    aug_transform = get_training_augmentation(imgsz)
    
    # Run validation three times with different samples or random augmentations
    for run_idx in range(3):
        # Use different samples for each run, cycling through available samples
        sample_idx = run_idx % len(dataframe)
        img_path = dataframe.iloc[sample_idx]['Image']
        mask_path = dataframe.iloc[sample_idx]['Mask']
        
        # Load original
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if run_idx == 0:  # Only print details for first run to avoid spam
            print(f"   • Original image shape: {image.shape}")
            print(f"   • Original mask shape: {mask.shape}")
            print(f"   • Mask unique values: {np.unique(mask)}")
        
        # Apply training augmentation
        aug_result = aug_transform(image=image, mask=mask)
        
        if run_idx == 0:  # Only print details for first run
            print(f"   • Processed image shape: {aug_result['image'].shape}")
            print(f"   • Processed mask shape: {aug_result['mask'].shape}")
            print(f"   • Processed mask unique values: {np.unique(aug_result['mask'])}")
        
        # Check that mask values are preserved
        original_classes = set(np.unique(mask))
        processed_classes = set(np.unique(aug_result['mask']))
        
        if run_idx == 0:  # Only check for first run
            if original_classes == processed_classes:
                print("   ✅ Mask class values preserved correctly")
            else:
                print(f"   ⚠️ Mask classes changed! Original: {original_classes}, Processed: {processed_classes}")
        
        # Visualize if path provided
        if save_path:
            # Create filename with run number
            if save_path.endswith('.png'):
                save_file = save_path.replace('.png', f'{run_idx + 1}.png')
            else:
                save_file = f"{save_path}{run_idx + 1}.png"
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # Create a custom colormap where background (0) is transparent
            from matplotlib.colors import ListedColormap
            import matplotlib.cm as cm
            
            # Get the tab10 colormap but make the first color (index 0) transparent
            tab10 = cm.get_cmap('tab10', 10)
            colors = tab10(np.linspace(0, 1, 10))
            colors[0] = [0, 0, 0, 0]  # Make background transparent (RGBA)
            custom_cmap = ListedColormap(colors)
            
            # For displaying masks without background, create masked arrays
            original_mask_display = np.ma.masked_where(mask == 0, mask)
            augmented_mask_display = np.ma.masked_where(aug_result['mask'] == 0, aug_result['mask'])
            
            # Original
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            axes[0, 0].set_aspect('equal')
            
            axes[0, 1].imshow(image)  # Show image as background
            axes[0, 1].imshow(original_mask_display, cmap=custom_cmap, alpha=0.7)
            axes[0, 1].set_title('Original Mask')
            axes[0, 1].axis('off')
            axes[0, 1].set_aspect('equal')
            
            # Augmented
            axes[1, 0].imshow(aug_result['image'])
            axes[1, 0].set_title('Augmented Image')
            axes[1, 0].axis('off')
            axes[1, 0].set_aspect('equal')
            
            axes[1, 1].imshow(aug_result['image'])  # Show image as background
            axes[1, 1].imshow(augmented_mask_display, cmap=custom_cmap, alpha=0.7)
            axes[1, 1].set_title('Augmented Mask')
            axes[1, 1].axis('off')
            axes[1, 1].set_aspect('equal')

            plt.tight_layout()
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   📊 Validation plot {run_idx + 1} saved to: {save_file}")
    
    print("✅ Augmentation validation complete!")


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
        # Basic spatial transformations - these work well with masks
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.2),
        
        # Geometric transformations with proper border handling
        albu.OneOf([
            albu.Rotate(
                limit=45, 
                p=1.0, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=0,
                mask_value=0
            ),
            albu.Affine(
                scale=(0.8, 1.2), 
                rotate=(-45, 45), 
                shear=(-10, 10), 
                p=1.0, 
                mode=cv2.BORDER_CONSTANT, 
                cval=0,
                mask_mode=cv2.BORDER_CONSTANT,
                cval_mask=0
            ),
        ], p=0.5),
        
        # Elastic deformation - good for both image and mask
        albu.ElasticTransform(
            alpha=1, 
            sigma=50, 
            alpha_affine=50, 
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        
        # Grid distortion - works well for semantic segmentation
        albu.GridDistortion(
            num_steps=5, 
            distort_limit=0.3, 
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        
        # Resize and padding - critical for maintaining aspect ratios
        albu.LongestMaxSize(max_size=imgsz),
        albu.PadIfNeeded(
            min_height=imgsz, 
            min_width=imgsz, 
            always_apply=True, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0,
            mask_value=0
        ),
        
        # Color and intensity augmentations - ONLY applied to image, not mask
        albu.OneOf([
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=1.0
            ),
            albu.CLAHE(
                clip_limit=4.0, 
                tile_grid_size=(8, 8), 
                p=1.0
            ),
            albu.RandomGamma(
                gamma_limit=(80, 120), 
                p=1.0
            ),
        ], p=0.8),
        
        # Hue/Saturation/Value adjustments - image only
        albu.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=30, 
            val_shift_limit=20, 
            p=0.5
        ),
        
        # Noise and artifacts - image only (masks should not be noisy)
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 50), p=1.0),
            albu.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Blur and sharpening - image only
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1.0),
            albu.MotionBlur(blur_limit=3, p=1.0), 
            albu.GaussianBlur(blur_limit=3, p=1.0),
            albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.4),
        
        # Dropout and cutout - can be problematic for masks, reducing probability
        albu.OneOf([
            albu.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                p=1.0,
                fill_value=0,
                mask_fill_value=0
            ),
        ], p=0.2),  # Reduced from 0.3
    ]

    return albu.Compose(train_transform)


def get_validation_augmentation(imgsz):
    """Get data augmentation pipeline for validation."""
    test_transform = [
        albu.LongestMaxSize(max_size=imgsz),
        albu.PadIfNeeded(
            min_height=imgsz, 
            min_width=imgsz, 
            always_apply=True, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0,
            mask_value=0
        ),
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
    process_split(val_dir, 'valid')
    process_split(test_dir, 'test')
    
    if not data_rows:
        raise Exception("ERROR: No valid image-mask pairs found in YOLO directories")
    
    return pd.DataFrame(data_rows)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Epoch:
    def __init__(self, model, loss, stage_name, device="cpu", verbose=True, metrics=None):
        self.model = model
        self.loss = loss
        if metrics is None:
            metrics = [TorchMetic(getattr(smp.metrics, m)) for m in get_segmentation_metrics()]
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

        # Get ignore_index from loss function
        # smp.metrics.functional._get_stats_multiclass supports a single int or None
        ignore_index = getattr(self.loss, 'ignore_index', None)

        # Initialize tensors for epoch-wide statistics
        epoch_tp = None
        epoch_fp = None
        epoch_fn = None
        epoch_tn = None
        num_classes = None

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

                # --- Stat calculation and aggregation ---
                
                # Get num_classes from the first batch output
                if num_classes is None:
                    num_classes = y_pred.shape[1]
                    # Initialize epoch stats tensors on the correct device
                    epoch_tp = torch.zeros(num_classes, dtype=torch.long, device=self.device)
                    epoch_fp = torch.zeros(num_classes, dtype=torch.long, device=self.device)
                    epoch_fn = torch.zeros(num_classes, dtype=torch.long, device=self.device)
                    epoch_tn = torch.zeros(num_classes, dtype=torch.long, device=self.device)

                # Convert y_pred logits/probs to class indices
                y_pred_classes = torch.argmax(y_pred, axis=1)

                # Calculate the stats for this batch
                tp, fp, fn, tn = smp.metrics.functional._get_stats_multiclass(
                    output=y_pred_classes,
                    target=y,
                    num_classes=num_classes,
                    ignore_index=ignore_index  # <-- Use derived ignore_index
                )
                
                # If stats are (B, C), sum over batch dim (0) to get (C,)
                if tp.ndim > 1:
                    tp = torch.sum(tp, dim=0)
                    fp = torch.sum(fp, dim=0)
                    fn = torch.sum(fn, dim=0)
                    tn = torch.sum(tn, dim=0)
                    
                tp = tp.to(self.device)
                fp = fp.to(self.device)
                fn = fn.to(self.device)
                tn = tn.to(self.device)

                # Sum stats for the epoch
                epoch_tp += tp
                epoch_fp += fp
                epoch_fn += fn
                epoch_tn += tn
        
                if self.verbose:
                    # Only log the running loss, as metrics are calculated at the end
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        # --- Compute metrics *after* the epoch loop ---
        metrics_logs = {}
        
        # Ensure stats were initialized (i.e., dataloader wasn't empty)
        if num_classes is not None:
            for metric_fn in self.metrics:
                # Compute the metric from the *total* epoch stats
                metric_values_per_class = smp.metrics.functional._compute_metric(
                    metric_fn, epoch_tp, epoch_fp, epoch_fn, epoch_tn
                )
                
                # Average the metric across classes, ignoring NaNs
                # (e.g., if a class was ignored or had 0/0)
                final_metric_value = torch.nanmean(metric_values_per_class)
                
                # Handle case where all values were NaN
                if torch.isnan(final_metric_value):
                    final_metric_value = 0.0
                
                metrics_logs[metric_fn.__name__] = final_metric_value.item()

        logs.update(metrics_logs)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
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
        
        # Standard training without AMP
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        
        return loss, prediction
        

class ValidEpoch(Epoch):
    def __init__(self, model, loss, device="cpu", verbose=True, metrics=None):
        super().__init__(
            model=model,
            loss=loss,
            stage_name="valid",
            device=device,
            verbose=verbose,
            metrics=metrics,
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

    def __init__(self, data_yaml_path, class_mapping_path=None):
        """Initialize DataConfig with path to data.yaml file and optional class_mapping.json."""
        self.data_yaml_path = data_yaml_path
        self.class_mapping_path = class_mapping_path
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
        self.class_names = [self.names[i] for i in range(self.nc)]
        self.class_ids = list(range(self.nc))  # This is the *integer index* map (0, 1, 2...)
        
        self.class_mapping = {}  # This will store the *application-level* map
        self.class_colors = []  # This will be built in order
        
        existing_mapping_data = {}
        if self.class_mapping_path and os.path.exists(self.class_mapping_path):
            try:
                with open(self.class_mapping_path, 'r') as f:
                    existing_mapping_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing class_mapping.json: {e}")

        for i, name in enumerate(self.class_names):
            # Check if this class 'name' is in the provided mapping file
            if name in existing_mapping_data:
                # Use existing info
                info = existing_mapping_data[name]
                class_uuid = info.get('id', str(uuid.uuid4()))  # Get UUID, or generate if missing
                color = info.get('color', [255, 0, 255])[:3]  # Get color, or default
            else:
                # Generate new info
                class_uuid = str(uuid.uuid4())
                # Generate random color
                np.random.seed(hash(name) % (2**32))
                color = np.random.randint(0, 256, 3).tolist()

            self.class_colors.append(color)  # Add to the ordered color list
            
            # Build the mapping with a proper UUID
            self.class_mapping[name] = {
                'id': class_uuid, 
                'short_label_code': name,
                'long_label_code': name,
                'color': color
            }

    def _build_dataframe(self):
        """Build dataframe from YOLO directory structure."""
        self.dataframe = build_dataframe_from_yolo(
            self.base_path, self.train_dir, self.val_dir, self.test_dir
        )

    def get_split_dataframes(self):
        """Return train, validation, and test dataframes."""
        train_df = self.dataframe[self.dataframe['Split'] == 'train'].copy()
        valid_df = self.dataframe[self.dataframe['Split'] == 'valid'].copy()
        test_df = self.dataframe[self.dataframe['Split'] == 'test'].copy()

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, valid_df, test_df


# ----------------------------------------------------------------------------------------------------------------------
# Main Model Class (Replaces ModelBuilder and integrates Predictor)
# ----------------------------------------------------------------------------------------------------------------------

class SemanticModel:
    """
    A class that encapsulates semantic segmentation model training and prediction,
    mimicking the Ultralytics API.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initializes the SemanticModel.

        Args:
            model_path (str, optional): Path to a pre-trained .pt model.
                                      If None, a new model must be built via .train().
        """
        self.model = None
        self.name = None  # Stashed model name
        self.preprocessing_fn = None
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.task = 'semantic'
        self.data_config = None
        self.class_names = []
        self.class_colors = []
        self.num_classes = 0
        self.imgsz = None
        self.class_mapping = {}

        # Prediction optimization settings
        self._optimized_model = None
        self._last_pred_config = {}

        if model_path:
            self.load(model_path)

    def load(self, model_path, device=None):
        """Loads a pre-trained model and associated metadata."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if device is not None:
            self.device = device

        # Determine the correct map_location
        if not torch.cuda.is_available():
            map_location = 'cpu'
        else:
            map_location = self.device
        
        # Load the model
        try:
            print(f"Loading model from {model_path}...")
            self.model = torch.load(model_path, map_location=map_location, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True, trying weights_only=False: {e}")
            try:
                self.model = torch.load(model_path, map_location=map_location, weights_only=False)
            except Exception as e2:
                raise Exception(f"Failed to load model from {model_path}: {e2}")

        # --- Retrieve stashed metadata directly from the model object ---
        self.name = getattr(self.model, 'name', None)
        self.imgsz = getattr(self.model, 'imgsz', None)
        self.class_names = getattr(self.model, 'class_names', [])
        self.class_colors = getattr(self.model, 'class_colors', [])
        self.num_classes = getattr(self.model, 'num_classes', 0)
        self.class_mapping = getattr(self.model, 'class_mapping', {})
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
            encoder_name = self.name.split('-', 1)[1]
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')
            print(f"Inferred preprocessing for encoder: {encoder_name}")
        except Exception as e:
            print(f"Warning: Could not infer preprocessing function from stashed name '{self.name}'. "
                  f"Using no-op. Error: {e}")
            self.preprocessing_fn = lambda x: x  # No-op

    def _build_model(self, encoder_name, decoder_name, num_classes, freeze, 
                     pre_trained_path, pretrained=True, device=None):
        """Internal method to build a new model. (Logic from original ModelBuilder)"""
        # Validate options
        if encoder_name not in get_segmentation_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_segmentation_encoders()}")
        if decoder_name not in get_segmentation_decoders():
            raise Exception(f"ERROR: Decoder must be one of {get_segmentation_decoders()}")

        if device is not None:
            self.device = device

        # Build model
        encoder_weights = 'imagenet'  # if pretrained else None
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
        freeze_params = int(num_params * freeze)
        print(f"🧊 Freezing {freeze * 100}% of encoder weights ({freeze_params}/{num_params})")
        for idx, param in enumerate(self.model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False
        
        self.model.to(self.device)

    def train(self, data_yaml, encoder_name=None, decoder_name=None,
              pre_trained_path=None, freeze=0.8, ignore_index=None, **kwargs):
        """
        Train the semantic segmentation model.
        """
        print(f"🚀 Training parameters: { {k: v for k, v in locals().items() if k != 'self'} }")
        
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = get_segmentation_metrics()

        print("\n" + "=" * 60)
        print("🚀 STARTING SEMANTIC SEGMENTATION TRAINING")
        print("=" * 60)
        
        # Store key info on self
        self.imgsz = kwargs.get('imgsz')
        val = kwargs.get('val', True)

        # Set device
        if kwargs.get('device') is not None:
            self.device = kwargs.get('device')
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Handle device types
        if isinstance(self.device, int):
            self.device = f'cuda:{self.device}'
        elif isinstance(self.device, list):
            # Multi-GPU
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
            self.device = f'cuda:{self.device[0]}'
        elif self.device == -1:
            # Auto-select most idle GPU (simple: use GPU 0)
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'

        # Handle pretrained
        if isinstance(kwargs.get('pretrained'), str):
            pre_trained_path = kwargs.get('pretrained')
            pretrained = True
        else:
            pretrained = kwargs.get('pretrained')

        # Handle output directory
        if kwargs.get('project') is None:
            kwargs['project'] = os.path.dirname(data_yaml)
        
        # Determine if user provided a custom name
        user_provided_name = kwargs.get('name') if kwargs.get('name') else None
        
        # Set base output directory
        if user_provided_name:
            # User provided a name, use it as the final directory name
            base_output_dir = os.path.join(kwargs['project'], user_provided_name)
        else:
            # No name provided, will use encoder_decoder naming in ExperimentManager
            base_output_dir = kwargs['project']

        # 1. Load Data Config
        print("📂 Loading dataset configuration...")
        class_mapping_path = kwargs.get('class_mapping')
        self.data_config = DataConfig(data_yaml, class_mapping_path)
        train_df, valid_df, test_df = self.data_config.get_split_dataframes()
        self.class_names = self.data_config.class_names
        self.class_colors = self.data_config.class_colors
        self.num_classes = self.data_config.nc
        self.class_mapping = self.data_config.class_mapping

        print("📊 Dataset Summary:")
        print(f"   • Training samples: {len(train_df)}")
        print(f"   • Validation samples: {len(valid_df)}")
        print(f"   • Test samples: {len(test_df)}")
        print(f"   • Classes ({self.num_classes}): {self.class_names}")
    
        # 2. Build Model (if not already loaded)
        if self.model is None:
            print("🔧 Building new model...")
            
            try:
                if pre_trained_path:
                    # Load existing model completely
                    self.load(pre_trained_path)
                    print(f"Loaded pre-trained model from: {pre_trained_path}")
                else:
                    # Build new model from encoder/decoder
                    if not encoder_name or not decoder_name:
                        raise ValueError("Must provide either pre_trained_path OR both encoder_name and decoder_name")
                    
                    self._build_model(
                        encoder_name=encoder_name,
                        decoder_name=decoder_name,
                        num_classes=self.num_classes,
                        freeze=freeze,
                        pre_trained_path=None,  # This is for encoder weights only
                        pretrained=pretrained,
                        device=self.device
                    )
            except Exception as e:
                print(f"❌ Failed to load or build model: {e}")
                raise
    
        # --- Automatically stamp metadata onto model ---
        print("📝 Stamping metadata onto model object for saving...")
        self.model.imgsz = self.imgsz
        self.model.class_names = self.class_names
        self.model.class_colors = self.class_colors
        self.model.num_classes = self.num_classes
        self.model.class_mapping = self.class_mapping
        
        # Parse ignore_index
        parsed_ignore_index = parse_ignore_index(ignore_index)

        # 3. Setup Training Config
        training_config = TrainingConfig(
            model=self.model,
            loss_function_name=kwargs.get('loss_function'),
            optimizer_name=kwargs.get('optimizer'),
            lr=kwargs.get('lr'),
            class_ids=self.data_config.class_ids,
            device=self.device,
            ignore_index=parsed_ignore_index
        )

        # 4. Setup Experiment Manager
        experiment_manager = ExperimentManager(
            output_dir=base_output_dir,
            decoder_name=decoder_name,
            encoder_name=encoder_name,
            class_mapping=self.data_config.class_mapping,
            exist_ok=kwargs.get('exist_ok'),
            user_provided_name=user_provided_name
        )
        experiment_manager.save_dataframes(train_df, valid_df, test_df)

        # 5. Setup Datasets
        dataset_manager = DatasetManager(
            train_df=train_df, valid_df=valid_df, test_df=test_df,
            class_ids=self.data_config.class_ids,
            preprocessing_fn=self.preprocessing_fn,  # Use the one from self
            augment_data=kwargs.get('augment_data'),
            batch=kwargs.get('batch'),
            imgsz=self.imgsz  # Use the one from self
        )
        
        # Validate augmentations, if augmentation is enabled
        if kwargs.get('augment_data'):
            validate_augmentation(
                train_df, 
                self.data_config.class_ids, 
                self.imgsz,
                os.path.join(experiment_manager.run_dir, 'augmentations')
            )
        
        # If validation is disabled, set valid loaders to None
        if not val:
            dataset_manager.valid_loader = None
            dataset_manager.valid_dataset = None
            dataset_manager.valid_dataset_vis = None
        
        dataset_manager.visualize_training_samples(
            experiment_manager.run_dir, self.data_config.class_colors
        )

        # 6. Run Trainer
        trainer = Trainer(
            model=self.model,
            loss_function=training_config.loss_function,
            optimizer=training_config.optimizer,
            device=self.device,
            epochs=kwargs.get('epochs'),
            train_loader=dataset_manager.train_loader,
            valid_loader=dataset_manager.valid_loader,
            valid_dataset=dataset_manager.valid_dataset,
            valid_dataset_vis=dataset_manager.valid_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=self.data_config.class_ids,
            class_colors=self.data_config.class_colors,
            patience=kwargs.get('patience')
        )
        trainer.train()

        # Plot training metrics
        experiment_manager.plot_metrics()

        # 7. Evaluate
        print("\n📥 Loading best weights for evaluation...")
        best_weights = os.path.join(experiment_manager.weights_dir, "best.pt")
        # Load best model back into self.model (this will also update self.name)
        self.load(best_weights) 
        
        # Call eval method for evaluation (same as standalone)
        self.eval(
            data_yaml=data_yaml,
            split='valid',
            num_vis_samples=kwargs.get('num_vis_samples'),
            output_dir=experiment_manager.run_dir,  # Integrate into training dir
            loss_function=kwargs.get('loss_function'),
            metrics=[m.__name__ for m in training_config.metrics],
            device=self.device,
            ignore_index=ignore_index
        )

        print(f"✅ Training pipeline completed! Best model at {best_weights}")
        self.model.eval()
        # Clear any optimized model from previous runs
        self._optimized_model = None
        
        return best_weights
    
    # --- Evaluation Methods ---
    
    def eval(self, data_yaml, split='test', num_vis_samples=10, output_dir=None,
             loss_function='JaccardLoss', metrics=None, device=None, ignore_index=None,
             imgsz=None, batch=1, confidence=0.5):
        """
        Run standalone evaluation on a trained model.

        Args:
            data_yaml (str): Path to the data.yaml file.
            split (str): Data split to evaluate on ('train', 'valid', or 'test').
            num_vis_samples (int): Number of result images to save.
            output_dir (str, optional): Directory to save results.
                If None, creates a new dir next to the model's 'weights' dir.
            loss_function (str): Name of the loss function to use for eval.
            metrics (list[str], optional): List of metric names to calculate.
            device (str, optional): Device to use for evaluation. If None, uses self.device.
            ignore_index (list[int], optional): Class indices to ignore during loss calculation.
            imgsz (int, optional): Image size for preprocessing. If None, uses model's training size.
            batch (int): Batch size for evaluation DataLoader.
            confidence (float): Confidence threshold for predictions in visualization.
        """
        if self.model is None:
            raise Exception("Model is not loaded. Load a model first.")
        
        if self.imgsz is None:
            raise Exception("Model `imgsz` is not set. Load a model trained "
                            "with this framework or provide `imgsz` manually.")

        if device is not None:
            self.device = device

        if metrics is None:
            metrics = get_segmentation_metrics()

        print("\n" + "=" * 60)
        print(f"🚀 STARTING STANDALONE EVALUATION ON '{split}' SPLIT")
        print("=" * 60)

        # 1. Load Data
        print(f"📂 Loading dataset from {data_yaml}...")
        data_config = DataConfig(data_yaml)
        all_dfs = data_config.get_split_dataframes()
        
        if split == 'train':
            eval_df = all_dfs[0]
        elif split == 'valid':
            eval_df = all_dfs[1]
        elif split == 'test':
            eval_df = all_dfs[2]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'valid', or 'test'.")

        if eval_df.empty:
            print(f"⚠️ No data found for split: {split}. Exiting.")
            return

        print(f"   • Evaluating on {len(eval_df)} samples from '{split}' split.")

        # Use provided imgsz or default to self.imgsz
        eval_imgsz = imgsz if imgsz is not None else self.imgsz

        # 2. Setup Datasets
        print("📦 Creating evaluation datasets...")
        eval_dataset = Dataset(
            eval_df,
            augmentation=get_validation_augmentation(eval_imgsz),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.class_ids,
        )
        eval_dataset_vis = Dataset(
            eval_df,
            augmentation=get_validation_augmentation(eval_imgsz),
            classes=self.class_ids,
        )
        
        # 3. Setup Metrics and Loss
        print("⚙️ Configuring loss...")
        try:
            loss_fn = getattr(smp.losses, loss_function)(mode='multiclass').to(self.device)
            loss_fn.__name__ = loss_fn._get_name()
            
            # Set ignore_index if supported and provided by user
            ignore_msg = "no indices ignored"
            if 'ignore_index' in inspect.signature(loss_fn.__init__).parameters and ignore_index is not None:
                try:
                    if hasattr(loss_fn, 'ignore_index'):
                        if len(ignore_index) == 1:
                            loss_fn.ignore_index = ignore_index[0]
                            ignore_msg = f"ignoring class {ignore_index[0]}"
                        else:
                            # Try setting as list first, fallback to first element
                            try:
                                loss_fn.ignore_index = ignore_index
                                ignore_msg = f"ignoring classes {ignore_index}"
                            except (TypeError, ValueError, AttributeError):
                                loss_fn.ignore_index = ignore_index[0]
                                ignore_msg = f"ignoring class {ignore_index[0]} (loss supports single index only)"
                except Exception as e:
                    print(f"⚠️ Warning: Could not set ignore_index on loss function: {e}")
            
            print(f"   • Loss: {loss_function} ({ignore_msg})")
        except Exception as e:
            print(f"❌ Error setting up loss: {e}")
            return

        # 4. Setup Experiment Manager
        if output_dir is None:
            # Place results in a new 'eval' dir at the same level as 'weights'
            model_base_dir = os.path.dirname(os.path.dirname(self.model_path))
            output_dir = os.path.join(model_base_dir, 'logs', split)
            print(f"   • No output_dir provided. Saving results to: {output_dir}")

        decoder_name, encoder_name = "Model", "eval"  # Defaults
        if self.name and '-' in self.name:
            decoder_name, encoder_name = self.name.split('-', 1)
        
        experiment_manager = ExperimentManager(
            output_dir=output_dir,
            decoder_name=decoder_name,
            encoder_name=encoder_name,
            class_mapping=self.class_mapping,
            exist_ok=True,  # For eval, we typically want to allow overwriting
            is_eval=True,
            user_provided_name=split  # Use split name as user provided name for eval
        )

        # 5. Instantiate and Run Evaluator
        evaluator = Evaluator(
            model=self.model,
            loss_function=loss_fn,
            device=self.device,
            test_dataset=eval_dataset,
            test_dataset_vis=eval_dataset_vis,
            experiment_manager=experiment_manager,
            class_ids=self.class_ids,
            class_colors=self.class_colors,
            num_vis_samples=num_vis_samples,
            batch_size=batch,
            confidence_threshold=confidence
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
        
        elif type(source) is np.ndarray:
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
            
            elif type(source[0]) is np.ndarray:  # List of images
                # Recursively normalize each image in the list
                normalized_list = [self._normalize_source(img) for img in source]
                images = [item[0][0] for item in normalized_list if item[0]]
                paths = [item[1][0] for item in normalized_list if item[1]]
            
            else:
                raise TypeError(f"Unsupported list element type: {type(source[0])}")

        elif type(source) is torch.Tensor:
            # Single image or batch as torch.Tensor
            img_np = self._tensor_to_numpy(source)
            # After conversion, img_np is (H,W,C) or (B,H,W,C), so we can
            # recursively call this function to handle it as a numpy array.
            return self._normalize_source(img_np)

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
            
        return images, paths

    def predict(self, source, confidence_threshold=0.5, imgsz=None):
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

        # Prepare the vanilla model (puts on device and in eval mode)
        if self._optimized_model is None:
            self._prepare_optimized_model()

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
            preprocessed = preproc(image=augmented['image'], mask=augmented['image'])  # mask is dummy
            preprocessed_tensors.append(torch.from_numpy(preprocessed['image']))

        batch_tensor = torch.stack(preprocessed_tensors).to(self.device)
        
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
            h_orig, w_orig = orig_shapes[i][:2]

            # 1. Re-calculate the intermediate size (after LongestMaxSize)
            # This logic mimics albu.LongestMaxSize
            if h_orig > w_orig:
                new_h = imgsz
                new_w = int(w_orig * (imgsz / h_orig))
            else:
                new_w = imgsz
                new_h = int(h_orig * (imgsz / w_orig))

            # 2. Re-calculate the padding (from PadIfNeeded)
            pad_h = imgsz - new_h
            pad_w = imgsz - new_w

            # Albumentations centers the padding
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad

            # 3. Crop the padding from the predicted mask
            # mask_aug is (imgsz, imgsz)
            # This gives a mask of shape (new_h, new_w)
            cropped_mask = mask_aug[top_pad: imgsz - bottom_pad, left_pad: imgsz - right_pad]

            # 4. Resize the *cropped* mask back to the original image size
            # cv2.resize expects dsize as (width, height)
            mask_array = cv2.resize(cropped_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

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
            torch.zeros_like(pred_class)  # Set to background
        )
        return pred_class

    def _post_process(self, mask_array, orig_shape, path, orig_img):
        """Converts a numpy mask array into an Ultralytics Results object."""
        if Results is None:
            print("Warning: `ultralytics` not installed. Returning raw numpy mask.")
            return mask_array

        h, w = orig_shape[:2]
        
        # 1. Create one-hot mask representation
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
            # No detections, return empty Results in a list
            return [
                Results(
                    orig_img=orig_img,
                    path=path,
                    names=dict(enumerate(self.class_names)),
                    boxes=torch.empty(0, 6),  # Raw tensor: (N, 6) where 6 = [x1, y1, x2, y2, conf, cls]
                    masks=torch.empty(0, h, w)  # Raw tensor: (N, H, W)
                )
            ]

        # Create masks tensor for non-empty classes
        final_masks_tensor = torch.from_numpy(one_hot_mask[non_empty_indices]).float()
        
        # 2. Create bounding boxes from masks
        try:
            boxes_xyxy = []
            for mask in final_masks_tensor:
                pos = torch.where(mask > 0)
                if len(pos[0]) > 0:
                    xmin = pos[1].min().item()
                    ymin = pos[0].min().item() 
                    xmax = pos[1].max().item()
                    ymax = pos[0].max().item()
                    boxes_xyxy.append([xmin, ymin, xmax, ymax])
                else:
                    boxes_xyxy.append([0, 0, 0, 0])
        
            # Convert to tensor
            boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
        
            # Add confidence and class columns
            conf = torch.ones(len(non_empty_indices), dtype=torch.float32)
            cls = torch.tensor(non_empty_indices, dtype=torch.float32)
        
            # Combine into final boxes format: [x1, y1, x2, y2, conf, cls]
            boxes_data = torch.cat([
                boxes_xyxy,
                conf.unsqueeze(1),
                cls.unsqueeze(1)
            ], dim=1)
        
        except Exception as e:
            print(f"Warning: Could not generate boxes from masks: {e}")
            # Fallback to empty boxes and masks
            boxes_data = torch.empty(0, 6)
            final_masks_tensor = torch.empty(0, h, w)

        # Return Results in a list to match original API
        return [
            Results(
                orig_img=orig_img,
                path=path,
                names=dict(enumerate(self.class_names)),
                boxes=boxes_data,  # Raw tensor data
                masks=final_masks_tensor  # Raw tensor data
            )
        ]

    # --- Optimization Methods (from Predictor) ---
    
    def _prepare_optimized_model(self):
        """Assigns the vanilla model for prediction."""
        print("Using vanilla PyTorch model for prediction.")
        self._optimized_model = self.model
        self._optimized_model.to(self.device)
        self._optimized_model.eval()

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

    def __init__(self, model, loss_function_name, optimizer_name, lr, class_ids, device, ignore_index=None):
        """Initialize TrainingConfig with training components."""
        self.model = model
        self.loss_function_name = loss_function_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.metrics_list = get_segmentation_metrics()
        self.class_ids = class_ids
        self.device = device
        self.ignore_index = ignore_index

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

        # Set ignore_index if supported and provided by user
        ignore_msg = "no indices ignored"
        if 'ignore_index' in params and self.ignore_index is not None:
            try:
                if hasattr(self.loss_function, 'ignore_index'):
                    self.loss_function.ignore_index = self.ignore_index
                    ignore_msg = f"ignoring class {self.ignore_index}"
            except Exception as e:
                print(f"⚠️ Warning: Could not set ignore_index on loss function: {e}")

        print(f"   • Loss: {self.loss_function_name} ({ignore_msg})")

    def _setup_optimizer(self):
        """Setup the optimizer."""
        assert self.optimizer_name in get_segmentation_optimizers()
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), self.lr)
        print(f"   • Optimizer: {self.optimizer_name} (lr={self.lr})")

    def _setup_metrics(self):
        """Setup the evaluation metrics."""
        self.metrics = [getattr(smp.metrics, m) for m in self.metrics_list]

        # Include at least one metric regardless
        if not self.metrics:
            self.metrics.append(smp.metrics.iou_score)

        # Convert to torch metric so can be used on CUDA
        self.metrics = [TorchMetic(m) for m in self.metrics]
        print(f"   • Metrics: {self.metrics_list}")


class ExperimentManager:
    """Manages experiment directories, logging, and result tracking."""

    def __init__(self, output_dir, decoder_name, encoder_name, class_mapping, 
                 exist_ok=False, is_eval=False, user_provided_name=None):
        """Initialize ExperimentManager with experiment configuration."""
        self.output_dir = output_dir
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.class_mapping = class_mapping
        self.metrics = [TorchMetic(getattr(smp.metrics, m)) for m in get_segmentation_metrics()]
        self.exist_ok = exist_ok
        self.is_eval = is_eval
        self.user_provided_name = user_provided_name

        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self):
        """Create experiment directories with proper naming and conflict resolution."""
        # Determine the run name and directory
        if self.user_provided_name:
            # User provided a name, use output_dir directly (which is already project/name)
            self.run = self.user_provided_name
            base_run_dir = self.output_dir
        else:
            # No user name provided, create encoder_decoder subfolder
            self.run = f"{self.encoder_name}_{self.decoder_name}"
            base_run_dir = os.path.join(self.output_dir, self.run)
        
        # Handle existing directories by adding numbers if exist_ok is False
        if not self.exist_ok and os.path.exists(base_run_dir):
            counter = 1
            original_run = self.run
            while True:
                if self.user_provided_name:
                    # For user-provided names, add number to the name itself
                    self.run = f"{original_run} {counter}"
                    test_run_dir = os.path.join(os.path.dirname(base_run_dir), self.run)
                else:
                    # For auto-generated names, add number to the encoder_decoder name
                    self.run = f"{original_run} {counter}"
                    test_run_dir = os.path.join(self.output_dir, self.run)
                
                if not os.path.exists(test_run_dir):
                    base_run_dir = test_run_dir
                    break
                counter += 1
        
        self.run_dir = base_run_dir
        self.weights_dir = os.path.join(self.run_dir, "weights") if not self.is_eval else None

        # Make the directories
        os.makedirs(self.run_dir, exist_ok=True)
        if self.weights_dir:
            os.makedirs(self.weights_dir, exist_ok=True)

        print(f"📁 Experiment: {self.run}")
        print(f"📁 Run Directory: {self.run_dir}")
        if self.weights_dir:
            print(f"📁 Weights Directory: {self.weights_dir}")

    def _setup_logging(self):
        """Setup logging files and CSV results tracking."""
        # Save the class_mapping to JSON
        class_mapping_path = os.path.join(self.run_dir, "class_mapping.json")
        with open(class_mapping_path, 'w') as f:
            json.dump(self.class_mapping, f, indent=4)

        # Create results CSV file
        self.results_csv_path = os.path.join(self.run_dir, "results.csv")
        with open(self.results_csv_path, 'w') as f:
            f.write("epoch,phase,loss")
            for metric in self.metrics:
                f.write(f",{metric.__name__}")
            f.write("\n")

    def save_dataframes(self, train_df, valid_df, test_df):
        """Save dataframes to CSV files in logs directory."""
        train_df.to_csv(os.path.join(self.run_dir, "training_split.csv"), index=False)
        valid_df.to_csv(os.path.join(self.run_dir, "validation_split.csv"), index=False)
        test_df.to_csv(os.path.join(self.run_dir, "testing_split.csv"), index=False)

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

            # 4. Create a fixed 2x5 subplot grid for the 10 metrics
            num_rows = 2
            num_cols = 5
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
            axes = axes.flatten()  # Flatten for easy iteration

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
                
                # Set y-axis limits to 0-1 for all metrics
                ax.set_ylim(0, 1)

            # 6. Save the figure
            plt.tight_layout()
            save_path = os.path.join(self.run_dir, "results.png")
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
                 augment_data=False, batch=8, imgsz=640):
        """Initialize DatasetManager with dataframes and configuration."""
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.class_ids = class_ids
        self.preprocessing_fn = preprocessing_fn
        self.augment_data = augment_data
        self.batch = batch
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
            batch_size=self.batch, 
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

    def visualize_training_samples(self, run_dir, class_colors):
        """Visualize training samples in three 3x3 grids and save to separate images."""
        print("\n" + "=" * 60)
        print("👀 GENERATING TRAINING SAMPLE VISUALIZATIONS")
        print("=" * 60)

        # Create a sample version dataset
        sample_dataset = Dataset(self.train_df,
                                 augmentation=get_training_augmentation(self.imgsz),
                                 classes=self.class_ids)

        # Generate 3 grids
        for grid_idx in range(3):
            # Create a 3x3 grid figure
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()  # Flatten to easily iterate

            # Loop through 9 samples
            for i in range(9):
                try:
                    # Get a random sample from dataset
                    image_tensor, mask_tensor = sample_dataset[np.random.randint(0, len(self.train_df))]
                    
                    # Convert tensors to numpy arrays with proper handling
                    if isinstance(image_tensor, torch.Tensor):
                        image = image_tensor.numpy()
                    else:
                        image = image_tensor
                        
                    if isinstance(mask_tensor, torch.Tensor):
                        mask = mask_tensor.numpy()
                    else:
                        mask = mask_tensor
                    
                    # Handle different image tensor shapes
                    if image.ndim == 3:
                        if image.shape[0] in [1, 3]:  # (C, H, W) format
                            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
                            # If grayscale, convert to RGB
                            if image.shape[2] == 1:
                                image = np.concatenate([image] * 3, axis=2)
                    # else already in (H, W, C) format
                    elif image.ndim == 4:
                        # Remove batch dimension if present
                        image = image.squeeze(0)
                        if image.shape[0] in [1, 3]:  # (C, H, W) format
                            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
                            if image.shape[2] == 1:
                                image = np.concatenate([image] * 3, axis=2)
                    
                    # Handle mask tensor shapes
                    if mask.ndim >= 2:
                        mask = mask.squeeze()  # Remove any extra dimensions
                    
                    # Ensure image values are in proper range for display
                    if image.dtype != np.uint8:
                        if np.max(image) <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = np.clip(image, 0, 255).astype(np.uint8)
                    
                    # Ensure mask is in proper format
                    if mask.dtype != np.uint8:
                        mask = mask.astype(np.uint8)

                    # Colorize the mask
                    colored_mask = colorize_mask(mask, self.class_ids, class_colors)
                    
                    # Plot in the subplot
                    axes[i].imshow(image)
                    axes[i].imshow(colored_mask, alpha=0.5)
                    axes[i].set_title(f'Sample {i+1}')
                    axes[i].axis('off')
                    
                except Exception as e:
                    print(f"⚠️ Could not visualize sample {i+1} in grid {grid_idx+1}: {e}")
                    # Add debug information
                    try:
                        print(f"   Debug info - Image shape: {image.shape if 'image' in locals() else 'N/A'}")
                        print(f"   Debug info - Mask shape: {mask.shape if 'mask' in locals() else 'N/A'}")
                        print(f"   Debug info - Image dtype: {image.dtype if 'image' in locals() else 'N/A'}")
                        print(f"   Debug info - Mask dtype: {mask.dtype if 'mask' in locals() else 'N/A'}")
                    except:
                        pass
                    axes[i].axis('off')  # Hide empty subplots

            # Save the grid to a single image
            save_path = os.path.join(run_dir, f'train_batch{grid_idx}.png')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"🖼️ Training samples grid {grid_idx+1} saved to {save_path}")
            
            
class Trainer:
    """Handles the training loop with early stopping and learning rate scheduling."""

    def __init__(self, model, loss_function, optimizer, device, epochs,
                 train_loader, valid_loader, valid_dataset, valid_dataset_vis,
                 experiment_manager, class_ids, class_colors, patience=100):
        """Initialize Trainer with training components."""
        self.model = model
        self.loss_function = loss_function
        self.metrics = [TorchMetic(getattr(smp.metrics, m)) for m in get_segmentation_metrics()]
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_dataset = valid_dataset
        self.valid_dataset_vis = valid_dataset_vis
        self.experiment_manager = experiment_manager
        self.class_ids = class_ids
        self.class_colors = class_colors
        self.patience = patience

        # Training state
        self.best_score = float('inf')
        self.best_epoch = 0
        self.since_best = 0
        self.since_drop = 0

        # Create training epochs
        self.train_epoch = TrainEpoch(
            model, loss_function, optimizer, device, verbose=True
        )
        self.valid_epoch = ValidEpoch(
            model, loss_function, device, verbose=True
        )

    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 60)
        print("🚀 STARTING SEMANTIC SEGMENTATION TRAINING")
        print("=" * 60)

        print("📋 Training Configuration:")
        print(f"   • Epochs: {self.epochs}")
        print(f"   • Device: {self.device}")
        print(f"   • Model: {type(self.model).__name__}")
        print(f"   • Loss: {self.loss_function.__name__}")
        print(f"   • Metrics: {[m.__name__ for m in self.metrics]}")
        print()

        try:
            # Training loop
            for e_idx in range(1, self.epochs + 1):
                print(f"\n🦖 Epoch {e_idx}/{self.epochs}")
                print("-" * 40)

                # Go through an epoch for train, valid
                train_logs = self.train_epoch.run(self.train_loader)
                if self.valid_loader is not None:
                    valid_logs = self.valid_epoch.run(self.valid_loader)
                else:
                    valid_logs = {}

                # Print training metrics
                print(f"  📈 Train: {format_logs_pretty(train_logs)}")
                if valid_logs:
                    print(f"  ✅ Valid: {format_logs_pretty(valid_logs)}")

                # Log training metrics to CSV
                self.experiment_manager.log_metrics(e_idx, "train", train_logs)
                if valid_logs:
                    self.experiment_manager.log_metrics(e_idx, "valid", valid_logs)

                # Visualize a validation sample
                if self.valid_dataset_vis is not None:
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
        """Visualize validation samples for the current epoch in a 3x2 grid."""
        try:
            # Select up to 3 random samples (or fewer if dataset is small)
            num_samples = min(3, len(self.valid_dataset_vis))
            if num_samples == 0:
                print(f"⚠️ No validation samples available for visualization in epoch {epoch}")
                return
            sample_indices = np.random.choice(len(self.valid_dataset_vis), num_samples, replace=False)
            
            # Create a grid with appropriate number of rows
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
            
            # Handle single sample case (axes is 1D)
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i, n in enumerate(sample_indices):
                # Get the original image without preprocessing
                image_vis = self.valid_dataset_vis[n][0].numpy()
                # Get the preprocessed input for model prediction
                image, gt_mask = self.valid_dataset[n]
                gt_mask = gt_mask.squeeze().numpy()
                x_tensor = image.to(self.device).unsqueeze(0)
                # Perform inference
                with torch.inference_mode():
                    output_mask = self.model(x_tensor)
                
                # Apply confidence threshold (using default 0.5 for training visualization)
                pred_probs = torch.softmax(output_mask, dim=1)
                pred_confidence, pred_class = torch.max(pred_probs, dim=1)
                pred_class = torch.where(
                    pred_confidence >= 0.1,
                    pred_class,
                    torch.zeros_like(pred_class)
                )
                pr_mask = pred_class[0].cpu().numpy().astype(np.uint8)
                
                # Colorize masks
                gt_colored = colorize_mask(gt_mask, self.class_ids, self.class_colors)
                pr_colored = colorize_mask(pr_mask, self.class_ids, self.class_colors)
                
                # Left column: Actual image with ground truth mask
                axes[i, 0].imshow(image_vis)
                axes[i, 0].imshow(gt_colored, alpha=0.5)
                axes[i, 0].set_title(f'Sample {i+1}: Ground Truth')
                axes[i, 0].axis('off')
                
                # Right column: Actual image with predicted mask
                axes[i, 1].imshow(image_vis)
                axes[i, 1].imshow(pr_colored, alpha=0.5)
                axes[i, 1].set_title(f'Sample {i+1}: Prediction')
                axes[i, 1].axis('off')
            
            # Save the grid to a single image
            save_path = os.path.join(self.experiment_manager.run_dir, f'val_batch{epoch}.png')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"🖼️ Validation samples grid saved to {save_path}")
        except Exception as e:
            print(f"⚠️ Could not visualize validation samples for epoch {epoch}: {e}")

    def _update_training_state(self, epoch, train_logs, valid_logs):
        """Update training state and handle early stopping."""
        # Get the loss values
        train_loss = [v for k, v in train_logs.items() if 'loss' in k.lower()][0]
        if valid_logs:
            loss_value = [v for k, v in valid_logs.items() if 'loss' in k.lower()][0]
        else:
            loss_value = train_loss

        if loss_value < self.best_score:
            # Update best
            self.best_score = loss_value
            self.best_epoch = epoch
            self.since_best = 0
            print(f"🏆 New best epoch {epoch}")
        else:
            # Increment the counters
            self.since_best += 1
            self.since_drop += 1
            print(f"📉 Model did not improve after epoch {self.best_epoch}")

        # Overfitting indication (only if validation is available)
        if valid_logs and train_loss < loss_value:
            print(f"⚠️ Overfitting detected in epoch {epoch}")

        # Check if it's time to decrease the learning rate
        if (self.since_best >= 5 or (valid_logs and train_loss <= loss_value)) and self.since_drop >= 5:
            self.since_drop = 0
            new_lr = self.optimizer.param_groups[0]['lr'] * 0.75
            self.optimizer.param_groups[0]['lr'] = new_lr
            print(f"🔄 Decreased learning rate to {new_lr:.6f} after epoch {epoch}")

        # Exit early if progress stops (only if validation is available)
        if valid_logs and self.since_best >= self.patience and train_loss < loss_value and self.since_drop >= 5:
            print("🛑 Training plateaued; stopping early")
            return False
        
        if self.since_best == 0:
            # Save the best model
            self.experiment_manager.save_best_model(self.model)

        return True

    def _log_error(self, error):
        """Log training error to file."""
        print(f"📄 Error details saved to {self.experiment_manager.run_dir}Error.txt")
        with open(os.path.join(self.experiment_manager.run_dir, "Error.txt"), 'a') as file:
            file.write(f"Caught exception: {str(error)}\n{traceback.format_exc()}\n")


class Evaluator:
    """Handles model evaluation and result visualization."""

    def __init__(self, model, loss_function, device, test_dataset, test_dataset_vis,
                 experiment_manager, class_ids, class_colors, num_vis_samples=10,
                 batch_size=1, confidence_threshold=0.5):
        """Initialize Evaluator with evaluation components."""
        self.model = model
        self.loss_function = loss_function
        self.metrics = [TorchMetic(getattr(smp.metrics, m)) for m in get_segmentation_metrics()]
        self.device = device
        self.test_dataset = test_dataset
        self.test_dataset_vis = test_dataset_vis
        self.experiment_manager = experiment_manager
        self.class_ids = class_ids
        self.class_colors = class_colors
        self.num_vis_samples = num_vis_samples
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        # Get original image dimensions
        self.original_width, self.original_height = Image.open(
            self.test_dataset_vis.images_fps[0]
        ).size

    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "=" * 60)
        print("🧪 EVALUATING MODEL")
        print("=" * 60)

        # Create test dataloader
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
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
        print(f"🎨 VISUALIZING {self.num_vis_samples} RESULTS")
        print("=" * 60)
        
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

                # Get the preprocessed input for model prediction
                image, _ = self.test_dataset[n]
                x_tensor = image.to(self.device).unsqueeze(0)
                # Perform inference
                with torch.inference_mode():
                    output_mask = self.model(x_tensor)
                
                # Apply confidence threshold
                pred_probs = torch.softmax(output_mask, dim=1)
                pred_confidence, pred_class = torch.max(pred_probs, dim=1)
                pred_class = torch.where(
                    pred_confidence >= 0.1,
                    pred_class,
                    torch.zeros_like(pred_class)
                )
                pr_mask = pred_class[0].cpu().numpy().astype(np.uint8)
                # Colorize the predicted mask (no resize needed since image_vis is already at imgsz)
                pr_mask = colorize_mask(pr_mask, self.class_ids, self.class_colors)
                # Colorize the ground truth mask
                gt_mask_vis = colorize_mask(gt_mask_vis, self.class_ids, self.class_colors)

                try:
                    # Visualize the colorized results locally
                    save_path = os.path.join(self.experiment_manager.run_dir, f'eval_batch{i}.png')
                    visualize(save_path=save_path,
                              save_figure=True,
                              image=image_vis,
                              ground_truth_mask=gt_mask_vis,
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
        print(f"📄 Error details saved to {self.experiment_manager.run_dir}Error.txt")
        with open(os.path.join(self.experiment_manager.run_dir, "Error.txt"), 'a') as file:
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

    parser.add_argument('--class_mapping', type=str, default=None,
                        help='Path to class_mapping.json file with class colors. '
                             'If not provided, random colors will be generated.')

    parser.add_argument('--pre_trained_path', type=str, default=None,
                        help='Path to pre-trained model of the same architecture')

    parser.add_argument('--encoder_name', type=str, default='mit_b0',
                        help=encoder_help)

    parser.add_argument('--decoder_name', type=str, default='Unet',
                        help=decoder_help)

    parser.add_argument('--loss_function', type=str, default='JaccardLoss',
                        help=loss_help)

    parser.add_argument('--freeze', type=float, default=0.80,
                        help='Freeze N percent of the encoder [0 - 1]')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help=optimizer_help)

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Starting learning rate')

    parser.add_argument('--augment_data', action='store_true',
                        help='Apply affine augmentations to training data')

    parser.add_argument('--epochs', type=int, default=25,
                        help='Starting learning rate')

    parser.add_argument('--batch', type=int, default=8,
                        help='Number of samples per batch during training')

    parser.add_argument('--imgsz', type=int, default=640,
                        help='Length of the longest edge after resizing input images (must be divisible by 32)')

    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of test samples to visualize during evaluation')

    parser.add_argument('--patience', type=int, default=30,
                        help='Number of epochs to wait without improvement in validation metrics before early stopping')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading')

    parser.add_argument('--device', type=str, default=None,
                        help='Specifies the computational device(s) for training')

    parser.add_argument('--project', type=str, default=None,
                        help='Name of the project directory where training outputs are saved')

    parser.add_argument('--name', type=str, default=None,
                        help='Name of the training run.')

    parser.add_argument('--exist_ok', action='store_true',
                        help='If True, allows overwriting of an existing project/name directory')

    parser.add_argument('--pretrained', type=str, default='True',
                        help='Determines whether to start training from a pretrained model')

    parser.add_argument('--val', action='store_true',
                        help='Enable validation during training if a validation set is provided')

    parser.add_argument('--ignore_index', type=int, default=None,
                        help='Class index to ignore during loss calculation. '
                             'Specify as a single integer, e.g., 0 or 255')
    
    args = parser.parse_args()
    
    # Parse device
    device = None
    if args.device is not None:
        if args.device == 'cpu':
            device = 'cpu'
        elif args.device == 'mps':
            device = 'mps'
        elif args.device.isdigit():
            device = int(args.device)
        elif args.device == '-1':
            device = -1
        elif args.device.startswith('[') and args.device.endswith(']'):
            device = eval(args.device)
        else:
            device = args.device
    
    # Parse pretrained
    pretrained = args.pretrained
    if pretrained.lower() == 'true':
        pretrained = True
    elif pretrained.lower() == 'false':
        pretrained = False
    # else keep as str
    
    # Check imgsz
    if args.imgsz % 32 != 0:
        new_size = (args.imgsz // 32) * 32
        print(f"⚠️ imgsz must be divisible by 32. Adjusting from {args.imgsz} to {new_size}.")
        args.imgsz = new_size

    # Parse ignore_index
    ignore_index = parse_ignore_index(args.ignore_index)
    if ignore_index is not None:
        print(f"📋 Parsed ignore_index: {ignore_index}")

    try:
        # Initialize the model (it will be built inside .train())
        model = SemanticModel()
        
        # Start training
        model.train(
            data_yaml=args.data_yaml,
            class_mapping=args.class_mapping,
            encoder_name=args.encoder_name,
            decoder_name=args.decoder_name,
            pre_trained_path=args.pre_trained_path,
            freeze=args.freeze,
            ignore_index=ignore_index,
            metrics=args.metrics,
            loss_function=args.loss_function,
            optimizer=args.optimizer,
            lr=args.lr,
            augment_data=args.augment_data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            num_vis_samples=args.num_vis_samples,
            patience=args.patience,
            device=device,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            pretrained=pretrained,
            val=args.val
        )

    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()

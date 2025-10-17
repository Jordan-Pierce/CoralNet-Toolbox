import os
import sys
import json
import glob
import warnings
import argparse
import traceback
from tqdm import tqdm

import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torcheval.metrics import functional as torch_metrics

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.meter import AverageValueMeter

import albumentations as albu

from coralnet_toolshed.Common import console_user

torch.cuda.empty_cache()

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------


def get_now():
    """Get current date and time as a formatted string."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


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


def get_classifier_encoders():
    """
    Lists all the models available
    """
    encoder_options = []

    try:
        options = smp.encoders.get_encoder_names()
        encoder_options = options

    except Exception as e:
        # Fail silently
        pass

    return encoder_options


def get_classifier_losses():
    """
    Lists all the losses available
    """
    loss_options = []
    try:
        import torch.nn as nn
        options = [_ for _ in dir(nn) if callable(getattr(nn, _))]
        options = [_ for _ in options if 'Loss' in getattr(nn, _).__name__]
        loss_options = options

    except Exception as e:
        # Fail silently
        pass

    return loss_options


def get_classifier_metrics():
    """
    Lists all the metrics available
    """
    metric_options = ['binary_accuracy',
                      'binary_f1_score',
                      'binary_precision',
                      'binary_recall',
                      'multiclass_accuracy',
                      'multiclass_f1_score',
                      'multiclass_precision',
                      'multiclass_recall']

    return metric_options


def get_classifier_optimizers():
    """
    Lists all the optimizers available
    """
    optimizer_options = []

    try:
        options = [attr for attr in dir(torch.optim) if callable(getattr(torch.optim, attr))]
        optimizer_options = options

    except Exception as e:
        # Fail silently
        pass

    return optimizer_options


def get_training_augmentation(height=224, width=224):
    """
    Training augmentation techniques; very light, if any.
    """
    train_transform = []

    train_transform.extend([

        # Center crop 112x112 1/4 the time, otherwise full patch,
        albu.OneOf([
            albu.CenterCrop(height=112, width=112, p=0.5),
            albu.NoOp(p=1.0)
        ], p=0.5),

        # Always resize to 224x224
        albu.Resize(height=height, width=width),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=0),

        # Flips
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.GaussNoise(p=0.1),
        albu.PixelDropout(p=0.1, dropout_prob=0.05),

        # Small amounts of brightness
        albu.OneOf(
            [
                albu.CLAHE(p=0.1),
                albu.RandomBrightness(p=0.1),
                albu.RandomGamma(p=0.1),
            ],
            p=0.9,
        ),
        # Small amounts of blur
        albu.OneOf(
            [
                albu.Sharpen(p=0.1),
                albu.Blur(blur_limit=3, p=0.1),
                albu.MotionBlur(blur_limit=3, p=0.1),
            ],
            p=0.9,
        ),
        # Small amounts of contrast / hue
        albu.OneOf(
            [
                albu.RandomContrast(p=0.1),
                albu.HueSaturationValue(p=0.1),
            ],
            p=0.9,
        ),
    ])

    return albu.Compose(train_transform)


def get_validation_augmentation(height=224, width=224):
    """
    Validation augmentation techniques
    """
    test_transform = []

    test_transform.extend([
        albu.Resize(height=height, width=width),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=0),
    ])

    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Convert image to tensor
    """
    if len(x.shape) == 2:
        return x
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """
    Preprocessing
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def downsample_majority_classes(df, about=0.1):
    """
    Function to downsample majority classes

    :param df:
    :param about:
    :return:
    """
    label_counts = df['Label'].value_counts()
    minority_count = label_counts.min()
    target_range = (minority_count * (1 - about), minority_count * (1 + about))

    downsampled_df = pd.DataFrame()

    for label, count in label_counts.items():
        if count > target_range[1]:
            sampled_df = df[df['Label'] == label].sample(n=int(target_range[1]), random_state=42)
        else:
            sampled_df = df[df['Label'] == label]

        downsampled_df = pd.concat([downsampled_df, sampled_df])

    return downsampled_df


def compute_class_weights(df, mu=0.15):
    """
    Compute class weights for the given dataframe.
    """
    # Compute the value counts for each class
    class_categories = sorted(df['Label'].unique())
    class_values = [df['Label'].value_counts()[c] for c in class_categories]
    value_counts = dict(zip(class_categories, class_values))

    total = sum(value_counts.values())
    keys = value_counts.keys()

    # To store the class weights
    class_weight = dict()

    # Compute the class weights for each class
    for key in keys:
        score = math.log(mu * total / float(value_counts[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def plot_confusion_matrix(matrix, writer, epoch, class_names, mode='Train', save_dir=None):
    """
    Plots the confusion matrix, saves it locally (if save_dir is provided), and uploads it to TensorBoard.
    """
    # Calculate the figure size dynamically based on the number of classes
    num_classes = len(class_names)
    figsize = (int(num_classes * 0.5), int(num_classes * 0.5))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the confusion matrix
    plot_matrix(ax, matrix, class_names, title=f'{mode} Confusion Matrix')

    # Save the plot locally if save_dir is provided
    if save_dir:
        save_path = f"{save_dir}/{mode}_Confusion_Matrix_Epoch_{epoch}.jpg"
        fig.savefig(save_path)

        # Log the visualization to TensorBoard if writer is provided
        if writer:
            figure = np.array(Image.open(save_path))
            writer.add_image(f'{mode} Confusion Matrix', figure, dataformats="HWC", global_step=epoch)

    # Log the plot to TensorBoard if writer is provided
    if writer:
        writer.add_figure(f'{mode} Confusion Matrix', fig, global_step=epoch, close=True)
    else:
        plt.close(fig)


def plot_matrix(ax, matrix, class_names, title):
    """
    Plots a confusion matrix on a given axis.
    """
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax)
    plt.subplots_adjust(bottom=0.2)


def plot_gridded_predictions(model, data_loader, num_rows=5):
    """
    Plots a grid of images with their actual and predicted labels.
    Adds a green border for correct predictions and a red border for incorrect predictions.

    Args:
        model (torch.nn.Module): The model to use for predictions.
        data_loader (torch.utils.data.DataLoader): The data loader containing the images and labels.
        num_rows (int): The number of rows in the grid.
    """
    model.eval()

    # Get validation samples
    samples = iter(data_loader)
    images, labels, paths = next(samples)
    num_samples = len(images)

    # Calculate the number of columns and total grids based on the square of num_rows
    num_grids = num_rows ** 2
    num_cols = num_rows

    # Set the num samples to num_grids if more
    num_samples = min(num_samples, num_grids)

    # Move data to the same device as the model
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        preds = model(images[:num_samples])

    preds = torch.argmax(preds, dim=1)
    labels = torch.argmax(labels, dim=1)

    class_names = list(data_loader.dataset.class_map.keys())

    # Create a grid for plotting the images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))

    for i in range(num_samples):
        ax = axes[i // num_cols, i % num_cols]

        actual_label = class_names[labels[i]]
        predicted_label = class_names[preds[i]]

        # Set title color based on prediction correctness
        title_color = 'green' if labels[i] == preds[i] else 'red'

        # Display the image
        img = plt.imread(paths[i])
        ax.imshow(img)

        # Get the image dimensions
        height, width, _ = img.shape

        # Create a Rectangle patch for the border
        rect = plt.Rectangle((-0.5, -0.5), width, height, fill=False, edgecolor=title_color, linewidth=5)
        ax.add_patch(rect)

        ax.set_title(f'Actual: {actual_label}\n'
                     f'Prediction: {predicted_label}',
                     fontsize=8,
                     color=title_color)

        ax.axis('off')

    plt.tight_layout()

    return fig


def build_dataframe_from_yolo_classification(data_dir):
    """
    Build a dataframe from YOLO image classification directory structure.
    
    Args:
        data_dir (str): Root directory containing train/, val/, test/ folders
        
    Returns:
        pd.DataFrame: Dataframe with columns 'Image Path', 'Label', 'Split'
    """
    data_rows = []
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ {split} directory not found, skipping")
            continue
        
        # Get all class subfolders
        class_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        
        for class_name in class_folders:
            class_dir = os.path.join(split_dir, class_name)
            image_files = []
            image_files.extend(glob.glob(os.path.join(class_dir, '*.jpg')))
            image_files.extend(glob.glob(os.path.join(class_dir, '*.png')))
            image_files.extend(glob.glob(os.path.join(class_dir, '*.jpeg')))
            image_files.extend(glob.glob(os.path.join(class_dir, '*.JPG')))
            image_files.extend(glob.glob(os.path.join(class_dir, '*.PNG')))
            
            for img_path in image_files:
                data_rows.append({
                    'Image Path': img_path,
                    'Label': class_name,
                    'Split': split
                })
    
    if not data_rows:
        raise Exception("❌ No valid images found in YOLO directories")
    
    return pd.DataFrame(data_rows)


# ------------------------------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------------------------------
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, log_dir, device="cpu", verbose=True):

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.log_dir = log_dir
        self.verbose = verbose
        self.device = device

        self.class_names = self.model.class_names
        self.num_classes = self.model.num_classes

        self.all_preds = torch.tensor([]).to(self.device)
        self.all_labels = torch.tensor([]).to(self.device)

        self.loss_meter = AverageValueMeter()
        self.micro_metrics_meters = {f"micro_{m.__name__}": AverageValueMeter() for m in self.metrics}
        self.macro_metrics_meters = {f"macro_{m.__name__}": AverageValueMeter() for m in self.metrics}

        self.matrix = None
        self.report = None

        self._to_device()

    def batch_update(self, x, y):
        raise NotImplementedError

    def _to_device(self):
        """

        :return:
        """
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        """

        :param logs:
        :return:
        """
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def on_epoch_start(self):
        """

        :return:
        """
        self.all_preds = torch.tensor([]).to(self.device)
        self.all_labels = torch.tensor([]).to(self.device)

        if self.stage_name == 'Train':
            self.model.train()
        elif self.stage_name == 'Valid':
            self.model.eval()
        else:
            raise NotImplementedError

    def calculate_metrics(self):
        """

        :return:
        """
        # Calculate each of the metrics
        for metric_fn in self.metrics:
            # Micro averaged (classes are weighted based on frequency)
            micro_value = metric_fn(self.all_preds,
                                    self.all_labels,
                                    average='micro',
                                    num_classes=self.num_classes)

            self.micro_metrics_meters[f'micro_{metric_fn.__name__}'].add(micro_value.item())

            # Macro averaged (classes are treated equally, regardless of frequency)
            macro_value = metric_fn(self.all_preds,
                                    self.all_labels,
                                    average='macro',
                                    num_classes=self.num_classes)

            self.macro_metrics_meters[f'macro_{metric_fn.__name__}'].add(macro_value.item())

    def calculate_classification_report(self):
        """

        :return:
        """
        # Convert to numpy
        all_preds = self.all_preds.cpu().numpy().astype(int)
        all_labels = self.all_labels.cpu().numpy().astype(int)

        # Calculate the report
        self.report = classification_report(all_labels,
                                            all_preds,
                                            target_names=self.class_names,
                                            digits=4)

    def calculate_confusion_matrix(self):
        """

        :return:
        """
        if self.num_classes == 2:
            matrix = torch_metrics.binary_confusion_matrix(self.all_preds,
                                                           self.all_labels,
                                                           normalize='all')
        else:
            matrix = torch_metrics.multiclass_confusion_matrix(self.all_preds,
                                                               self.all_labels,
                                                               num_classes=self.num_classes,
                                                               normalize='all')
        # Return as np for plotting
        self.matrix = matrix.cpu().numpy()

    def plot_gridded_results(self, dataloader, epoch_num):
        """

        :param dataloader:
        :return:
        """
        # Plot gridded predictions samples
        save_path = f'{self.log_dir}/{self.stage_name}_Result_{epoch_num}.jpg'
        figure = plot_gridded_predictions(self.model, dataloader)
        figure.savefig(save_path)
        plt.close(figure)

    def log_results(self, logs, epoch_num):
        """

        :param logs:
        :param epoch_num:
        :return:
        """
        # Format the metrics
        micro_metrics_logs = {k: v.mean for k, v in self.micro_metrics_meters.items()}
        macro_metrics_logs = {k: v.mean for k, v in self.macro_metrics_meters.items()}

        # Update metrics in log
        logs.update(micro_metrics_logs)
        logs.update(macro_metrics_logs)

        # Plot and log the confusion matrix
        plot_confusion_matrix(self.matrix,
                              None,  # No writer
                              epoch_num,
                              self.class_names,
                              mode=self.stage_name,
                              save_dir=f"{self.log_dir}")

        # Plot and log the classification report
        with open(f"{self.log_dir}/{self.stage_name}_Report_{epoch_num}.txt", 'w') as f:
            f.write(self.report)

        return logs

    def run(self, dataloader, epoch_num):
        """

        :param dataloader:
        :param epoch_num:
        :return:
        """
        self.on_epoch_start()

        logs = {}

        with tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch", file=sys.stdout) as iterator:
            for idx, (x, y, _) in iterator:
                # Pass forward
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # Update loss logs
                loss_value = loss.cpu().detach().numpy()
                self.loss_meter.add(loss_value)
                loss_logs = {'loss': self.loss_meter.mean}
                logs.update(loss_logs)
                # Output to console
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)

                # Reshape the prediction and ground-truth
                y_pred = torch.argmax(y_pred, axis=1)
                y = torch.argmax(y, axis=1)

                # Store predictions and ground-truths
                self.all_preds = torch.cat([self.all_preds, y_pred], dim=0).long()
                self.all_labels = torch.cat([self.all_labels, y], dim=0).long()

                # Check if this is the last step of the epoch
                if idx == len(dataloader) - 1:
                    # Calculate metrics
                    self.calculate_metrics()
                    # Calculate confusion matrix
                    self.calculate_confusion_matrix()
                    # Calculate report
                    self.calculate_classification_report()
                    # Plot gridded results
                    self.plot_gridded_results(dataloader, epoch_num)
                    # Log all the results
                    logs = self.log_results(logs, epoch_num)

                    # Print to console
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, log_dir, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="Train",
            log_dir=log_dir,
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, log_dir, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="Valid",
            log_dir=log_dir,
            device=device,
            verbose=verbose,
        )

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


class Dataset(BaseDataset):
    """

    """

    def __init__(
            self,
            dataframe,
            class_map,
            augmentation=None,
            preprocessing=None,
            log_dir=None,
    ):
        assert 'Name' in dataframe.columns, "ERROR: 'Name' not found in Patches file"
        assert 'Path' in dataframe.columns, "ERROR: 'Path' not found in Patches file"
        assert 'Label' in dataframe.columns, "ERROR: 'Label' not found in Patches file"

        self.df = dataframe
        self.ids = dataframe['Name'].to_list()
        self.patches = dataframe['Path'].to_list()
        self.labels = dataframe['Label'].to_list()
        self.class_map = class_map
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.log_dir = log_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """

        :param i:
        :return:
        """
        # read data
        image = cv2.imread(self.patches[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[i]

        # apply augmentations
        if self.augmentation:
            image = self.augmentation(image=image)['image']

        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']

        # Convert the image
        image = torch.from_numpy(image)
        # Convert the label
        label = torch.tensor([self.class_map[label]])
        label = F.one_hot(label.view(-1), len(self.class_map.keys()))
        label = label.squeeze().to(torch.float)

        return image, label, self.patches[i]

    def plot_samples(self, n=1):
        """

        :param n:
        :return:
        """
        save_paths = []

        for idx in range(n):
            # Get a random sample from dataset
            image, label, _ = self[np.random.randint(0, len(self.df))]
            image = image.numpy()
            label = np.argmax(label.numpy())
            # Visualize and save to logs dir
            save_path = f'{self.log_dir}/Sample_{idx}.jpg'
            plt.figure()
            plt.imshow(image)
            plt.title(f"{list(self.class_map.keys())[label]}")
            plt.savefig(save_path)
            plt.tight_layout()
            plt.close()

            save_paths.append(save_path)

        return save_paths


class CustomModel(torch.nn.Module):
    def __init__(self, encoder_name, weights, class_names, dropout_rate):
        super(CustomModel, self).__init__()

        # Classes
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Model Name
        self.name = encoder_name

        # Pre-trained encoder
        self.encoder = smp.encoders.get_encoder(name=encoder_name, weights=weights)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(self.encoder.out_channels[-1], self.num_classes)

        # Dropout layer
        if isinstance(dropout_rate, int):
            dropout_rate = dropout_rate / 100

        self.dropout = torch.nn.Dropout(dropout_rate)

    # Add a method to get the name attribute
    def get_name(self):
        return self.name

    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)
        x = x[-1]
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # Dropout layer
        x = self.dropout(x)
        # Fully connected layer for classification
        x = self.fc(x)
        # Softmax activation
        x = F.softmax(x, dim=1)

        return x


# ------------------------------------------------------------------------------------------------------------------
# New Classes for Structured Training
# ------------------------------------------------------------------------------------------------------------------

class DataConfig:
    """Handles loading and parsing of patch data from YOLO directory structure."""

    def __init__(self, data_dir):
        """Initialize DataConfig with path to data directory."""
        self.data_dir = data_dir
        self._load_data()
        self._build_class_map()

    def _load_data(self):
        """Load data from YOLO directory structure."""
        self.patches_df = build_dataframe_from_yolo_classification(self.data_dir)

    def _build_class_map(self):
        """Build class name and ID mappings."""
        class_names = sorted(self.patches_df['Label'].unique())
        self.num_classes = len(class_names)
        self.class_map = {class_names[_]: _ for _ in range(self.num_classes)}
        print(f"🏷️ Classes Found: {self.num_classes} ({class_names})")

    def get_dataframes(self):
        """Return the loaded dataframe and class map."""
        return self.patches_df, self.class_map


class ModelBuilder:
    """Handles construction and configuration of classification models."""

    def __init__(self, encoder_name, num_classes, freeze_encoder=0.0, dropout_rate=0.5, pre_trained_path=None):
        """Initialize ModelBuilder with model configuration."""
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.dropout_rate = dropout_rate
        self.pre_trained_path = pre_trained_path

        self._validate_encoder()
        self._build_model()
        self._load_pretrained_weights()
        self._freeze_encoder()

    def _validate_encoder(self):
        """Validate that encoder is available."""
        if self.encoder_name not in get_classifier_encoders():
            raise ValueError(f"❌ Encoder {self.encoder_name} not available")

    def _build_model(self):
        """Build the classification model."""
        self.model = CustomModel(
            encoder_name=self.encoder_name,
            weights='imagenet',
            class_names=list(range(self.num_classes)),  # Placeholder
            dropout_rate=self.dropout_rate
        )
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder_name, 'imagenet')
        print(f"   • Encoder: {self.encoder_name}")
        print(f"   • Classes: {self.num_classes}")

    def _load_pretrained_weights(self):
        """Load pretrained weights if provided."""
        if self.pre_trained_path and os.path.exists(self.pre_trained_path):
            self.model.load_state_dict(torch.load(self.pre_trained_path))
            print(f"   • Loaded Pretrained Weights: {self.pre_trained_path}")

    def _freeze_encoder(self):
        """Freeze a percentage of the encoder weights."""
        if self.freeze_encoder > 0:
            num_params = len(list(self.model.encoder.parameters()))
            freeze_params = int(num_params * self.freeze_encoder)
            for idx, param in enumerate(self.model.encoder.parameters()):
                if idx < freeze_params:
                    param.requires_grad = False
            print(f"   • Frozen Encoder: {self.freeze_encoder*100}%")


class TrainingConfig:
    """Handles configuration of loss functions, optimizers, and metrics for training."""

    def __init__(self, model, loss_function_name, optimizer_name, learning_rate, metrics_list, class_weights=None):
        """Initialize TrainingConfig with training components."""
        self.model = model
        self.loss_function_name = loss_function_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.metrics_list = metrics_list
        self.class_weights = class_weights

        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_metrics()

    def _setup_loss_function(self):
        """Setup the loss function."""
        if self.loss_function_name not in get_classifier_losses():
            raise ValueError(f"❌ Loss {self.loss_function_name} not available")

        loss_class = getattr(torch.nn, self.loss_function_name)
        if self.class_weights is not None:
            self.loss_function = loss_class(weight=self.class_weights)
        else:
            self.loss_function = loss_class()
        print(f"   • Loss: {self.loss_function_name}")

    def _setup_optimizer(self):
        """Setup the optimizer."""
        if self.optimizer_name not in get_classifier_optimizers():
            raise ValueError(f"❌ Optimizer {self.optimizer_name} not available")
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), self.learning_rate)
        print(f"   • Optimizer: {self.optimizer_name} (lr={self.learning_rate})")

    def _setup_metrics(self):
        """Setup the evaluation metrics."""
        available_metrics = get_classifier_metrics()
        self.metrics = []
        for m in self.metrics_list:
            if m in available_metrics:
                if 'accuracy' in m:
                    self.metrics.append(getattr(torch_metrics, m))
                else:
                    self.metrics.append(getattr(torch_metrics, m))
        if not self.metrics:
            self.metrics = [torch_metrics.multiclass_accuracy]
        print(f"   • Metrics: {[m.__name__ for m in self.metrics]}")


class ExperimentManager:
    """Manages experiment directories, logging, and result tracking."""

    def __init__(self, output_dir, encoder_name, class_map):
        self.output_dir = output_dir
        self.encoder_name = encoder_name
        self.class_map = class_map
        self.run = f"{get_now()}_{encoder_name}"
        self._setup_directories()

    def _setup_directories(self):
        """Setup experiment directories."""
        self.run_dir = f"{self.output_dir}/classification/{self.run}/"
        self.weights_dir = self.run_dir + "weights/"
        self.logs_dir = self.run_dir + "logs/"

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        with open(f'{self.run_dir}Class_Map.json', 'w') as f:
            json.dump(self.class_map, f, indent=3)

        print(f"📂 Run: {self.run}")
        print(f"📂 Run Directory: {self.run_dir}")

    def save_best_model(self, model):
        """Save the best model."""
        best_model_path = os.path.join(self.weights_dir, "best.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 Best model saved to {best_model_path}")

    def save_last_model(self, model):
        """Save the last model."""
        last_model_path = os.path.join(self.weights_dir, "last.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"💾 Last model saved to {last_model_path}")


class DatasetManager:
    """Manages creation and configuration of training datasets."""

    def __init__(self, patches_df, class_map, preprocessing_fn, augment_data=False, batch_size=128):
        self.patches_df = patches_df
        self.class_map = class_map
        self.preprocessing_fn = preprocessing_fn
        self.augment_data = augment_data
        self.batch_size = batch_size

        self._split_data()
        self._create_datasets()
        self._create_dataloaders()

    def _split_data(self):
        """Split data into train/val/test based on Split column."""
        self.train_df = self.patches_df[self.patches_df['Split'] == 'train'].copy()
        self.valid_df = self.patches_df[self.patches_df['Split'] == 'val'].copy()
        self.test_df = self.patches_df[self.patches_df['Split'] == 'test'].copy()

        # Reset indices
        self.train_df.reset_index(drop=True, inplace=True)
        self.valid_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)

        print(f"📊 Train: {len(self.train_df)}, Valid: {len(self.valid_df)}, Test: {len(self.test_df)}")

    def _create_datasets(self):
        """Create dataset objects."""
        training_augmentation = get_training_augmentation() if self.augment_data else get_validation_augmentation()

        self.train_dataset = Dataset(
            self.train_df,
            augmentation=training_augmentation,
            preprocessing=get_preprocessing(self.preprocessing_fn),
            class_map=self.class_map,
            log_dir=None  # Will set later
        )

        self.valid_dataset = Dataset(
            self.valid_df,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            class_map=self.class_map,
            log_dir=None
        )

        self.test_dataset = Dataset(
            self.test_df,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            class_map=self.class_map,
            log_dir=None
        )

    def _create_dataloaders(self):
        """Create data loaders."""
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class Trainer:
    """Handles the training loop with early stopping and learning rate scheduling."""

    def __init__(self, model, loss_function, metrics, optimizer, train_loader, valid_loader, 
                 num_epochs, logs_dir, experiment_manager):
        self.model = model
        self.loss_function = loss_function
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.logs_dir = logs_dir
        self.experiment_manager = experiment_manager

        self.train_epoch = TrainEpoch(self.model, self.loss_function, self.metrics, self.optimizer, self.logs_dir)
        self.valid_epoch = ValidEpoch(self.model, self.loss_function, self.metrics, self.logs_dir)

        # Training state
        self.best_score = float('inf')
        self.best_epoch = 0
        self.since_best = 0
        self.since_drop = 0

    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 60)
        print("🚀 STARTING CLASSIFICATION TRAINING")
        print("=" * 60)

        print("📋 Training Configuration:")
        print(f"   • Epochs: {self.num_epochs}")
        print(f"   • Model: {type(self.model).__name__}")
        print(f"   • Loss: {self.loss_function.__name__}")
        print(f"   • Metrics: {[m.__name__ for m in self.metrics]}")
        print()

        try:
            # Training loop
            for e_idx in range(1, self.num_epochs + 1):
                print(f"\n� Epoch {e_idx}/{self.num_epochs}")
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

                # Save the model for the current epoch
                self.experiment_manager.save_last_model(self.model)

                # Check for best model and handle early stopping
                should_continue = self._update_training_state(e_idx, train_logs, valid_logs)
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
            file.write(str(error))


class Evaluator:
    """Handles model evaluation and result visualization."""

    def __init__(self, model, loss_function, metrics, test_loader, logs_dir):
        self.model = model
        self.loss_function = loss_function
        self.metrics = metrics
        self.test_loader = test_loader
        self.logs_dir = logs_dir

        self.test_epoch = ValidEpoch(self.model, self.loss_function, self.metrics, self.logs_dir)

    def evaluate(self):
        """Evaluate the model on test set."""
        print("\n📈 Evaluating on test set")
        test_logs = self.test_epoch.run(self.test_loader, 0)
        print(f"📊 Test Results: {test_logs}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Train an Image Classifier')

    parser.add_argument('--data_dir', required=True,
                        help='Path to YOLO classification dataset directory (with train/val/test subdirs)')

    parser.add_argument('--pre_trained_path', type=str, default=None,
                        help='Path to pre-trained model of the same architecture')

    parser.add_argument('--encoder_name', type=str, default='resnet34',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--freeze_encoder', type=float, default=0.0,
                        help='Freeze N% of the encoder [0 - 1]')

    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss',
                        help='The loss function to use to train the model')

    parser.add_argument('--weighted_loss', action='store_true',
                        help='Use a weighted loss function; good for imbalanced datasets')

    parser.add_argument('--metrics', nargs="+", default=[],
                        help='The metrics used to evaluate the model (default is all applicable)')

    parser.add_argument('--optimizer', default="Adam",
                        help='Optimizer for training the model')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')

    parser.add_argument('--augment_data', action='store_true',
                        help='Apply affine augmentations to training data')

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Amount of dropout in model (augmentation)')

    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')

    parser.add_argument('--imgsz', type=int, default=224,
                        help='Length of the longest edge after resizing input images')

    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision training for faster training and reduced memory usage')

    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 quantization for maximum inference speed (applied to saved models)')

    parser.add_argument('--num_vis_samples', type=int, default=10,
                        help='Number of test samples to visualize during evaluation')

    args = parser.parse_args()

    # Set output dir to results in data directory
    args.output_dir = os.path.join(args.data_dir, 'results')

    # Validate AMP compatibility
    if args.amp and not torch.cuda.is_available():
        print("⚠️ AMP requires CUDA. Disabling AMP and using CPU training.")
        args.amp = False

    try:
        print("\n" + "=" * 60)
        print("🧠 IMAGE CLASSIFICATION TRAINING PIPELINE")
        print("=" * 60)
        print("🔧 Initializing...")

        # Check for CUDA
        print(f"   • PyTorch: {torch.__version__}")
        print(f"   • CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   • Device: {device}")
        print()

        # Load data
        print("📂 Loading dataset configuration...")
        data_config = DataConfig(args.data_dir)
        patches_df, class_map = data_config.get_dataframes()

        print("📊 Dataset Summary:")
        print(f"   • Training samples: {len(patches_df[patches_df['Split'] == 'train'])}")
        print(f"   • Validation samples: {len(patches_df[patches_df['Split'] == 'val'])}")
        print(f"   • Test samples: {len(patches_df[patches_df['Split'] == 'test'])}")
        print(f"   • Classes: {list(class_map.keys())}")
        print()

        # Build model
        print("🤖 Model Summary:")
        model_builder = ModelBuilder(
            encoder_name=args.encoder_name,
            num_classes=len(class_map),
            freeze_encoder=args.freeze_encoder,
            dropout_rate=args.dropout_rate,
            pre_trained_path=args.pre_trained_path
        )
        model = model_builder.model
        preprocessing_fn = model_builder.preprocessing_fn

        # Setup experiment
        print("📁 Experiment Setup:")
        print(f"   • Output Directory: {args.output_dir}")
        experiment_manager = ExperimentManager(args.output_dir, args.encoder_name, class_map)

        # Setup datasets
        print("📦 Dataset Configuration:")
        print(f"   • Augmentation: {'Enabled' if args.augment_data else 'Disabled'}")
        print(f"   • Batch Size: {args.batch_size}")
        dataset_manager = DatasetManager(
            patches_df=patches_df,
            class_map=class_map,
            preprocessing_fn=preprocessing_fn,
            augment_data=args.augment_data,
            batch_size=args.batch_size
        )

        # Compute class weights if needed
        class_weights = None
        if args.weighted_loss:
            class_weights = compute_class_weights(dataset_manager.train_df)
            class_weights = torch.tensor(list(class_weights.values())).to(device)

        # Setup training config
        training_config = TrainingConfig(
            model=model,
            loss_function_name=args.loss_function,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            metrics_list=args.metrics,
            class_weights=class_weights
        )

        # Train
        trainer = Trainer(
            model=model,
            loss_function=training_config.loss_function,
            metrics=training_config.metrics,
            optimizer=training_config.optimizer,
            train_loader=dataset_manager.train_loader,
            valid_loader=dataset_manager.valid_loader,
            num_epochs=args.num_epochs,
            logs_dir=experiment_manager.logs_dir,
            experiment_manager=experiment_manager
        )

        best_epoch = trainer.train()
        print(f"🏆 Best epoch: {best_epoch}")

        # Evaluate
        evaluator = Evaluator(
            model=model,
            loss_function=training_config.loss_function,
            metrics=training_config.metrics,
            test_loader=dataset_manager.test_loader,
            logs_dir=experiment_manager.logs_dir
        )

        evaluator.evaluate()
        print("✅ Training completed successfully\n")

    except Exception as e:
        print(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()

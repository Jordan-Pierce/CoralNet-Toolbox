import os
import sys
import json
import glob
import time
import shutil
import inspect
import warnings
import argparse
import traceback
import subprocess
from tqdm import tqdm

import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import functional as torch_metrics

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.meter import AverageValueMeter

import albumentations as albu

from Common import get_now

torch.cuda.empty_cache()

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------------------------------
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
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
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
            for x, y, _, in iterator:
                # Pass forward
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # Update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss._get_name(): loss_meter.mean}
                logs.update(loss_logs)

                # Reshape the prediction and ground-truth
                num_classes = y_pred.shape[1]
                y_pred = torch.argmax(y_pred, axis=1)
                y = torch.argmax(y, axis=1)

                # Update metrics logs
                for metric_fn in self.metrics:
                    metric_values = metric_fn(y_pred, y, average=None, num_classes=num_classes)
                    metrics_meters[metric_fn.__name__].add(torch.mean(metric_values))

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
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
        self.model.train()

    def batch_update(self, x, y):
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
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


class PytorchMetric(torch.nn.Module):
    """

    """

    def __init__(self, func):
        super(PytorchMetric, self).__init__()
        self.func = func  # The custom function to be wrapped

    def forward(self, *args, **kwargs):
        # Check if a device is specified in the keyword arguments
        device = kwargs.get('device', 'cpu')

        # Move any input tensors to the specified device
        args = [arg.to(device) if torch.is_tensor(arg) else arg for arg in args]

        # Execute the custom function on the specified device
        result = self.func(*args)

        return result

    @property
    def __name__(self):
        return self.func.__name__


class Dataset(BaseDataset):
    """

    """

    def __init__(
            self,
            dataframe,
            class_map,
            augmentation=None,
            preprocessing=None,
    ):
        assert 'Name' in dataframe.columns, print(f"ERROR: 'Name' not found in Patches file")
        assert 'Path' in dataframe.columns, print(f"ERROR: 'Path' not found in Patches file")
        assert 'Label' in dataframe.columns, print(f"ERROR: 'Label' not found in Patches file")

        self.ids = dataframe['Name'].to_list()
        self.patches = dataframe['Path'].to_list()
        self.labels = dataframe['Label'].to_list()
        self.class_map = class_map
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

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

    def __len__(self):
        return len(self.ids)


class CustomModel(torch.nn.Module):
    def __init__(self, encoder_name, weights, num_classes, dropout_rate):
        super(CustomModel, self).__init__()

        # Name
        self.name = encoder_name

        # Pre-trained encoder
        self.encoder = smp.encoders.get_encoder(name=encoder_name,
                                                weights=weights)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(self.encoder.out_channels[-1],
                                  num_classes)

        # Dropout layer
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
# Functions
# ------------------------------------------------------------------------------------------------------------------
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

    """
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.Resize(height=height, width=width),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=0),

        albu.GaussNoise(p=0.2),
        albu.PixelDropout(p=1.0, dropout_prob=0.1),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),

    ]

    return albu.Compose(train_transform)


def get_validation_augmentation(height=224, width=224):
    """

    """
    test_transform = [
        albu.Resize(height=height, width=width),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=0),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    if len(x.shape) == 2:
        return x
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


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


def plot_samples(model, data_loader, num_samples=100):
    """

    """
    model.eval()

    # Get validation samples
    samples = iter(data_loader)
    images, labels, paths = next(samples)

    # Move data to the same device as the model
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        preds = model(images)

    labels = np.argmax(labels.cpu().numpy(), axis=1)
    preds = np.argmax(preds.cpu().numpy(), axis=1)

    class_names = list(data_loader.dataset.class_map.keys())

    # Create a grid for plotting the images
    rows = int(np.ceil(num_samples / 10))
    fig, axes = plt.subplots(rows, 10, figsize=(20, 2 * rows))

    for i in range(num_samples):
        ax = axes[i // 10, i % 10] if num_samples > 10 else axes[i]

        actual_label = class_names[labels[i]]
        predicted_label = class_names[preds[i]]

        # Set title color based on prediction correctness
        title_color = 'green' if labels[i] == preds[i] else 'red'

        ax.imshow(plt.imread(paths[i]))
        ax.set_title(f'Actual: {actual_label}\n'
                     f'Prediction: {predicted_label}', fontsize=8, color=title_color)
        ax.axis('off')

    plt.tight_layout()

    return fig


# ------------------------------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------------------------------

def classification(args):
    """

    """

    print("\n###############################################")
    print(f"Classification")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------------------------------------------------------
    # Check input data
    # ------------------------------------------------------------------------------------------------------------------

    # If the user provides multiple patch dataframes
    patches_df = pd.DataFrame()

    for patches_path in args.patches:
        if os.path.exists(patches_path):
            # Patch dataframe
            patches = pd.read_csv(patches_path, index_col=0)
            patches = patches.dropna()
            patches_df = pd.concat((patches_df, patches))
        else:
            raise Exception(f"ERROR: Patches dataframe {patches_path} does not exist")

    class_names = sorted(patches_df['Label'].unique())
    num_classes = len(class_names)
    class_map = {f"{class_names[_]}": _ for _ in range(num_classes)}

    # ------------------------------------------------------------------------------------------------------------------
    # Building Model
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Building Model\n"
          f"#########################################\n")

    try:
        encoder_weights = 'imagenet'

        if args.encoder_name not in get_classifier_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_classifier_encoders()}")

        # Building model using user's input
        model = CustomModel(encoder_name=args.encoder_name,
                            weights=encoder_weights,
                            num_classes=num_classes,
                            dropout_rate=args.dropout_rate)

        # Freezing percentage of the encoder
        num_params = len(list(model.encoder.parameters()))
        freeze_params = int(num_params * args.freeze_encoder)

        # Give users the ability to freeze N percent of the encoder
        print(f"NOTE: Freezing {args.freeze_encoder}% of encoder weights")
        for idx, param in enumerate(model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False

        preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, encoder_weights)

        print(f"NOTE: Using {args.encoder_name} encoder")

    except Exception as e:
        print(f"ERROR: Could not build model\n{e}")
        sys.exit(1)

    # ---------------------------------------------------------------------------------------
    # Source directory setup
    # ---------------------------------------------------------------------------------------
    print("\n###############################################")
    print("Logging")
    print("###############################################\n")

    output_dir = f"{args.output_dir}\\"

    # Run Name
    run = f"{get_now()}_{args.encoder_name}"

    # We'll also create folders in this source to hold results of the model
    run_dir = f"{output_dir}classification\\{run}\\"
    weights_dir = run_dir + "weights\\"
    logs_dir = run_dir + "logs\\"
    tensorboard_dir = logs_dir + "tensorboard\\"

    # Make the directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    with open(f'{run_dir}Class_Map.json', 'w') as json_file:
        json.dump(class_map, json_file, indent=3)

    print(f"NOTE: Model Run - {run}")
    print(f"NOTE: Model Directory - {run_dir}")
    print(f"NOTE: Log Directory - {logs_dir}")
    print(f"NOTE: Tensorboard Directory - {tensorboard_dir}")

    # Create a SummaryWriter for logging to tensorboard
    train_writer = SummaryWriter(log_dir=tensorboard_dir + "train")
    valid_writer = SummaryWriter(log_dir=tensorboard_dir + "valid")
    test_writer = SummaryWriter(log_dir=tensorboard_dir + "test")

    # Open tensorboard
    if args.tensorboard:
        print(f"\n#########################################\n"
              f"Tensorboard\n"
              f"#########################################\n")

        # Create a subprocess that opens tensorboard
        tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', tensorboard_dir],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)

        print("NOTE: View Tensorboard at 'http://localhost:6006'")

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data, creating datasets
    # ------------------------------------------------------------------------------------------------------------------
    print("\n###############################################")
    print("Creating Datasets")
    print("###############################################\n")
    # Names of all images; sets to be split based on images
    image_names = patches_df['Image Name'].unique()

    # Split the Images into training, validation, and test sets (70/20/10)
    # We split based on the image names, so that we don't have the same image in multiple sets.
    training_images, temp_images = train_test_split(image_names, test_size=0.3, random_state=42)
    validation_images, testing_images = train_test_split(temp_images, test_size=0.33, random_state=42)

    # Create training, validation, and test dataframes
    train_df = patches_df[patches_df['Image Name'].isin(training_images)]
    valid_df = patches_df[patches_df['Image Name'].isin(validation_images)]
    test_df = patches_df[patches_df['Image Name'].isin(testing_images)]

    train_classes = len(set(train_df['Label'].unique()))
    valid_classes = len(set(valid_df['Label'].unique()))
    test_classes = len(set(test_df['Label'].unique()))

    # If there isn't one class sample in each data sets
    # Keras will throw an error; hacky way of fixing this.
    if not (train_classes == valid_classes == test_classes):
        print("NOTE: Sampling one of each class category")
        # Holds one sample of each class category
        sample = pd.DataFrame()
        # Gets one sample from patches_df
        for label in patches_df['Label'].unique():
            one_sample = patches_df[patches_df['Label'] == label].sample(n=1)
            sample = pd.concat((sample, one_sample))

        train_df = pd.concat((sample, train_df))
        valid_df = pd.concat((sample, valid_df))
        test_df = pd.concat((sample, test_df))

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Output to logs
    train_df.to_csv(f"{logs_dir}Training_Set.csv", index=False)
    valid_df.to_csv(f"{logs_dir}Validation_Set.csv", index=False)
    test_df.to_csv(f"{logs_dir}Testing_Set.csv", index=False)

    # The number of class categories
    print(f"NOTE: Number of classes in training set is {len(train_df['Label'].unique())}")
    print(f"NOTE: Number of classes in validation set is {len(valid_df['Label'].unique())}")
    print(f"NOTE: Number of classes in testing set is {len(test_df['Label'].unique())}")

    # ------------------------------------------------------------------------------------------------------------------
    # Data Exploration
    # ------------------------------------------------------------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    # Set the same y-axis limits for all subplots
    ymin = 0
    ymax = train_df['Label'].value_counts().max() + 10

    # Plotting the train data
    plt.subplot(1, 3, 1)
    plt.title(f"Train: {len(train_df)} Classes: {len(train_df['Label'].unique())}")
    ax = train_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Plotting the valid data
    plt.subplot(1, 3, 2)
    plt.title(f"Valid: {len(valid_df)} Classes: {len(valid_df['Label'].unique())}")
    ax = valid_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Plotting the test data
    plt.subplot(1, 3, 3)
    plt.title(f"Test: {len(test_df)} Classes: {len(test_df['Label'].unique())}")
    ax = test_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Saving and displaying the figure
    plt.savefig(logs_dir + "DatasetSplit.png")
    plt.close()

    if os.path.exists(logs_dir + "DatasetSplit.png"):
        print(f"NOTE: Datasplit Figure saved in {logs_dir}")

    # ------------------------------------------------------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------------------------------------------------------

    # Whether to include training augmentation
    if args.augment_data:
        training_augmentation = get_training_augmentation()
    else:
        training_augmentation = get_validation_augmentation()

    train_dataset = Dataset(
        train_df,
        augmentation=training_augmentation,
        preprocessing=get_preprocessing(preprocessing_fn),
        class_map=class_map,
    )

    valid_dataset = Dataset(
        valid_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_map=class_map,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ------------------------------------------------------------------------------------------------------------------
    # Show sample of training data
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Viewing Training Samples\n"
          f"#########################################\n")

    # Create a sample version dataset
    sample_dataset = Dataset(train_df,
                             augmentation=training_augmentation,
                             class_map=class_map)

    # Loop through a few samples
    for i in range(5):
        # Get a random sample from dataset
        image, label, _ = sample_dataset[np.random.randint(0, len(train_df))]
        image = image.numpy()
        label = np.argmax(label.numpy())
        # Visualize and save to logs dir
        save_path = f'{tensorboard_dir}train\\TrainingSample_{i}.png'
        plt.figure()
        plt.imshow(image)
        plt.title(f"{class_names[label]}")
        plt.savefig(save_path)
        plt.tight_layout()
        plt.close()

        # Write to tensorboard
        train_writer.add_image(f'Training_Samples', np.array(Image.open(save_path)), dataformats="HWC", global_step=i)

    # ------------------------------------------------------------------------------------------------------------------
    # Start of parameter setting
    # ------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Setting Parameters\n"
          f"#########################################\n")

    # Calculate weights
    if args.weighted_loss:
        print(f"NOTE: Calculating weights for weighted loss function")
        class_weight = compute_class_weights(train_df)
        print(f"NOTE: {class_weight}")
    else:
        class_weight = {c: 1.0 for c in range(num_classes)}

    # Reformat for training
    class_weight = torch.tensor(list(class_weight.values()))

    try:
        # Get the loss function
        assert args.loss_function in get_classifier_losses()

        # Get the loss function
        loss_function = getattr(torch.nn, args.loss_function)().to(device)

        # Get the parameters of the DiceLoss class using inspect.signature
        params = inspect.signature(loss_function.__init__).parameters

        # Check if the 'classes' or 'ignore_index' parameters exist
        if 'weight' in params and args.weighted_loss:
            loss_function.weight = class_weight

        print(f"NOTE: Using loss function {args.loss_function}")

    except Exception as e:
        print(f"ERROR: Could not get loss function {args.loss_function}")
        print(f"NOTE: Choose one of the following: {get_classifier_losses()}")
        sys.exit(1)

    try:
        # Get the optimizer
        assert args.optimizer in get_classifier_optimizers()
        optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), args.learning_rate)

        print(f"NOTE: Using optimizer function {args.optimizer}")

    except Exception as e:
        print(f"ERROR: Could not get optimizer {args.optimizer}")
        print(f"NOTE: Choose one of the following: {get_classifier_optimizers()}")
        sys.exit(1)

    try:
        # Get the metrics
        if not any(m in get_classifier_metrics() for m in args.metrics):
            args.metrics = get_classifier_metrics()

        if len(class_names) == 2:
            metrics = [m for m in args.metrics if "binary" in m]
        elif len(class_names) > 2:
            metrics = [m for m in args.metrics if "multiclass" in m]
        else:
            raise Exception("ERROR: There is only one class present in dataset, cannot train model")

        print(f"NOTE: Using metrics {metrics}")
        metrics = [getattr(torch_metrics, m) for m in metrics]

        # Convert for CUDA
        metrics = [PytorchMetric(m) for m in metrics]

    except Exception as e:
        print(f"ERROR: Could not get metric(s): {args.metrics}")
        print(f"NOTE: Choose one or more of the following: {get_classifier_metrics()}")
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------------------------------------------------
    try:

        print(f"\n#########################################\n"
              f"Training\n"
              f"#########################################\n")

        print("NOTE: Starting Training")
        train_epoch = TrainEpoch(
            model,
            loss=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        valid_epoch = ValidEpoch(
            model,
            loss=loss_function,
            metrics=metrics,
            device=device,
            verbose=True,
        )

        best_score = float('inf')
        best_epoch = 0
        since_best = 0
        since_drop = 0

        # Training loop
        for e_idx in range(1, args.num_epochs + 1):

            print(f"\nEpoch: {e_idx} / {args.num_epochs}")
            # Go through an epoch for train, valid
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # Log training metrics
            for key, value in train_logs.items():
                train_writer.add_scalar(key, value, global_step=e_idx)

            # Log validation metrics
            for key, value in valid_logs.items():
                valid_writer.add_scalar(key, value, global_step=e_idx)

            # Plot validation samples
            save_path = f'{tensorboard_dir}valid\\ValidResult_{e_idx}.png'
            figure = plot_samples(model, valid_loader)
            figure.savefig(save_path)

            # Log the visualization to TensorBoard
            figure = np.array(Image.open(save_path))
            valid_writer.add_image(f'Valid_Results', figure, dataformats="HWC", global_step=e_idx)

            # Get the loss values
            train_loss = [v for k, v in train_logs.items() if 'loss' in k.lower()][0]
            valid_loss = [v for k, v in valid_logs.items() if 'loss' in k.lower()][0]

            if valid_loss < best_score:
                # Update best
                best_score = valid_loss
                best_epoch = e_idx
                since_best = 0
                print(f"NOTE: Current best epoch {e_idx}")

                # Save the model
                prefix = f'{weights_dir}model-{str(e_idx)}-'
                suffix = f'{str(np.around(train_loss, 4))}-{str(np.around(valid_loss, 4))}'
                path = prefix + suffix
                torch.save(model, f'{path}.pth')
                print(f'NOTE: Model saved to {path}')
            else:
                # Increment the counters
                since_best += 1
                since_drop += 1
                print(f"NOTE: Model did not improve after epoch {e_idx}")

            # Overfitting indication
            if train_loss < valid_loss:
                print(f"NOTE: Overfitting occurred in epoch {e_idx}")

            # Check if it's time to decrease the learning rate
            if (since_best >= 5 or train_loss <= valid_loss) and since_drop >= 5:
                since_drop = 0
                new_lr = optimizer.param_groups[0]['lr'] * 0.75
                optimizer.param_groups[0]['lr'] = new_lr
                print(f"NOTE: Decreased learning rate to {new_lr} after epoch {e_idx}")

            # Exit early if progress stops
            if since_best >= 10 and train_loss < valid_loss and since_drop >= 5:
                print("NOTE: Model training plateaued; exiting training loop")
                break

    except KeyboardInterrupt:
        print("NOTE: Exiting training loop")

    except Exception as e:
        print(f"ERROR: There was an issue with training!\n{e}")

        if 'CUDA out of memory' in str(e):
            print(f"WARNING: Not enough GPU memory for the provided parameters")

        # Write the error to text file
        print(f"NOTE: Please see {logs_dir}Error.txt")
        with open(f"{logs_dir}Error.txt", 'a') as file:
            file.write(f"Caught exception: {str(traceback.print_exc())}\n")

        # Exit early
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # Load best model
    # ------------------------------------------------------------------------------------------------------------------
    weights = sorted(glob.glob(weights_dir + "*.pth"))
    best_weights = [w for w in weights if f'model-{str(best_epoch)}' in w][0]

    # Load into the model
    model = torch.load(best_weights)
    print(f"NOTE: Loaded best weights {best_weights}")

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate model on test set
    # ------------------------------------------------------------------------------------------------------------------

    # Create test dataset
    test_dataset = Dataset(
        test_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_map=class_map,
    )

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate on the test set
    test_epoch = ValidEpoch(
        model=model,
        loss=loss_function,
        metrics=metrics,
        device=device,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Calculating Metrics\n"
          f"#########################################\n")

    try:
        # Empty cache from training
        torch.cuda.empty_cache()

        # Score on test set
        test_logs = test_epoch.run(test_loader)

        # Log test metrics
        for key, value in test_logs.items():
            test_writer.add_scalar(key, value, global_step=best_epoch)

    except Exception as e:
        print(f"ERROR: Could not calculate metrics")

        if 'CUDA out of memory' in str(e):
            print(f"WARNING: Not enough GPU memory for the provided parameters")

        # Write the error to text file
        print(f"NOTE: Please see {logs_dir}Error.txt")
        with open(f"{logs_dir}Error.txt", 'a') as file:
            file.write(f"Caught exception: {str(e)}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize results
    # ------------------------------------------------------------------------------------------------------------------

    try:
        # Empty cache from testing
        torch.cuda.empty_cache()

        # Visualize the colorized results locally
        save_path = f'{tensorboard_dir}test\\TestResults.png'
        figure = plot_samples(model, test_loader)
        figure.savefig(save_path)

        # Log the visualization to TensorBoard
        figure = np.array(Image.open(save_path))
        test_writer.add_image(f'Test_Results', figure, dataformats="HWC", global_step=i)

    except Exception as e:
        print(f"ERROR: Could not make predictions")

        if 'CUDA out of memory' in str(e):
            print(f"WARNING: Not enough GPU memory for the provided parameters")

        # Write the error to text file
        print(f"NOTE: Please see {logs_dir}Error.txt")
        with open(f"{logs_dir}Error.txt", 'a') as file:
            file.write(f"Caught exception: {str(e)}\n")

    print(f"NOTE: Saving best weights in {run_dir}")
    shutil.copyfile(best_weights, f"{run_dir}Best_Model_and_Weights.pth")

    # Close tensorboard writers
    for writer in [train_writer, valid_writer, test_writer]:
        writer.close()

    # Close tensorboard
    if args.tensorboard:
        print("NOTE: Closing Tensorboard in 60 seconds")
        time.sleep(60)
        tensorboard_process.terminate()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train an Image Classifier')

    parser.add_argument('--patches', required=True, nargs="+",
                        help='The path to the patch labels csv file output the Patches tool')

    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--freeze_encoder', type=float,
                        help='Freeze N% of the encoder [0 - 1]')

    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss',
                        help='The loss function to use to train the model')

    parser.add_argument('--weighted_loss', type=bool, default=True,
                        help='Use a weighted loss function; good for imbalanced datasets')

    parser.add_argument('--metrics', nargs="+", default=[],
                        help='The metrics used to evaluate the model (default is all applicable)')

    parser.add_argument('--optimizer', default="Adam",
                        help='Optimizer for training the model')

    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')

    parser.add_argument('--augment_data', action='store_true',
                        help='Apply affine augmentations to training data')

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Amount of dropout in model (augmentation)')

    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Starting learning rate')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Starting learning rate')

    parser.add_argument('--tensorboard', action='store_true',
                        help='Display training on Tensorboard')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save updated label csv file.')

    args = parser.parse_args()

    try:
        classification(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
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
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.meter import AverageValueMeter

from tensorboard import program

import albumentations as albu

from src.Common import get_now
from src.Common import console_user

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


class TorchMetic(torch.nn.Module):
    """

    """

    def __init__(self, func):
        super(TorchMetic, self).__init__()
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
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        assert 'Name' in dataframe.columns, print(f"ERROR: 'Name' not found in mask file")
        assert 'Image Path' in dataframe.columns, print(f"ERROR: 'Image Path' not found in mask file")
        assert 'Semantic Path' in dataframe.columns, print(f"ERROR: 'Semantic Path' not found in mask file")

        self.ids = dataframe['Name'].to_list()
        self.masks_fps = dataframe['Semantic Path'].to_list()
        self.images_fps = dataframe['Image Path'].to_list()

        # convert str names to class values on masks
        self.class_ids = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

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
        return len(self.ids)


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def get_segmentation_encoders():
    """

    """
    encoder_options = []

    try:
        options = smp.encoders.get_encoder_names()
        encoder_options = options

    except Exception as e:
        # Fail silently
        pass

    return encoder_options


def get_segmentation_decoders():
    """

    """
    decoder_options = []
    try:

        options = [_ for _, obj in inspect.getmembers(smp) if inspect.isclass(obj)]
        decoder_options = options

    except Exception as e:
        # Fail silently
        pass

    return decoder_options


def get_segmentation_losses():
    """

    """
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
    """

    """
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


def visualize(save_path=None, save_figure=False, image=None, **masks):
    """

    """
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
        print(f"NOTE: Figure saved to {save_path}")

    # Show the figure
    plt.close()


def colorize_mask(mask, class_ids, class_colors):
    """

    """
    # Initialize the RGB mask with zeros
    height, width = mask.shape[0:2]

    rgb_mask = np.full((height, width, 3), fill_value=0, dtype=np.uint8)

    # dict with index as key, rgb as value
    cmap = {i: class_colors[i_idx] for i_idx, i in enumerate(class_ids)}

    # Loop through all index values
    # Set rgb color in colored mask
    for val in np.unique(mask):
        if val in class_ids:
            color = np.array(cmap[val])
            rgb_mask[mask == val, :] = color.astype(np.uint8)

    return rgb_mask.astype(np.uint8)


def get_training_augmentation(height, width):
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


def get_validation_augmentation(height, width):
    """
    Padding so divisible by 32
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


def segmentation(args):
    """

    """
    print("\n###############################################")
    print("Semantic Segmentation")
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
    # Color mapping file
    if os.path.exists(args.color_map):
        with open(args.color_map, 'r') as json_file:
            color_map = json.load(json_file)

        # Modify color map format
        class_names = list(color_map.keys())
        class_ids = [color_map[c]['id'] for c in class_names]
        class_colors = [color_map[c]['color'] for c in class_names]

    else:
        raise Exception(f"ERROR: Color Mapping JSON file provided doesn't exist; check input provided")

    # Data
    if os.path.exists(args.masks):
        dataframe = pd.read_csv(args.masks)
    else:
        raise Exception(f"ERROR: Mask file provided does not exist; please check input")

    # ------------------------------------------------------------------------------------------------------------------
    # Model building, parameters
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Building Model\n"
          f"#########################################\n")

    try:
        # Make sure it's a valid choice
        if args.encoder_name not in get_segmentation_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_segmentation_encoders()}")

        if args.decoder_name not in get_segmentation_decoders():
            raise Exception(f"ERROR: Decoder must be one of {get_segmentation_decoders()}")

        # Building model using user's input
        encoder_weights = 'imagenet'

        model = getattr(smp, args.decoder_name)(
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            classes=len(class_names),
            activation='softmax2d',
        )

        print(f"NOTE: Using {args.encoder_name} encoder with a {args.decoder_name} decoder")

        # Get the weights of the pre-trained encoder, if provided
        if args.pre_trained_path:
            pre_trained_model = torch.load(args.pre_trained_path, map_location='cpu')

            try:
                # Getting the encoder name (preprocessing), and encoder state
                encoder_name = pre_trained_model.name
                state_dict = pre_trained_model.encoder.state_dict()
            except:
                encoder_name = pre_trained_model.encoder.name
                state_dict = pre_trained_model.encoder.encoder.state_dict()

            model.encoder.load_state_dict(state_dict, strict=True)
            print(f"NOTE: Loaded pre-trained weights from {encoder_name}")

        else:
            print("WARNING: Path to pre-trained encoder does not exist, skipping")

        # Freezing percentage of the encoder
        num_params = len(list(model.encoder.parameters()))
        freeze_params = int(num_params * args.freeze_encoder)

        # Give users the ability to freeze N percent of the encoder
        print(f"NOTE: Freezing {args.freeze_encoder}% of encoder weights")
        for idx, param in enumerate(model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False

        preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, encoder_weights)

    except Exception as e:
        raise Exception(f"ERROR: Could not build model\n{e}")

    try:
        # Get the loss function
        assert args.loss_function in get_segmentation_losses()

        # Specify the mode
        mode = 'binary' if len(class_ids) == 2 else 'multiclass'
        loss_function = getattr(smp.losses, args.loss_function)(mode=mode).to(device)
        loss_function.__name__ = loss_function._get_name()

        # Get the parameters of the DiceLoss class using inspect.signature
        params = inspect.signature(loss_function.__init__).parameters

        # Check if the 'classes' or 'ignore_index' parameters exist
        if 'classes' in params and 'ignore_index' in params:
            loss_function.classes = [i for i in class_ids if i != 0]
            loss_function.ignore_index = 0
        elif 'classes' in params:
            loss_function.classes = [i for i in class_ids if i != 0]
        elif 'ignore_index' in params:
            loss_function.ignore_index = 0
        else:
            pass

        print(f"NOTE: Using loss function {args.loss_function}")

    except Exception as e:
        raise Exception(f"ERROR: Could not get loss function {args.loss_function}\n"
                        f"NOTE: Choose one of the following: {get_segmentation_losses()}")

    try:
        # Get the optimizer
        assert args.optimizer in get_segmentation_optimizers()
        optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), args.learning_rate)
        print(f"NOTE: Using optimizer {args.optimizer}")

    except Exception as e:
        raise Exception(f"ERROR: Could not get optimizer {args.optimizer}\n"
                        f"NOTE: Choose one of the following: {get_segmentation_optimizers()}")

    try:
        # Get the metrics
        assert any(m in get_segmentation_metrics() for m in args.metrics)
        metrics = [getattr(smp.metrics, m) for m in args.metrics]

        # Include at least one metric regardless
        if not metrics:
            metrics.append(smp.metrics.iou_score)

        # Convert to torch metric so can be used on CUDA
        metrics = [TorchMetic(m) for m in metrics]
        print(f"NOTE: Using metrics {args.metrics}")

    except Exception as e:
        raise Exception(f"ERROR: Could not get metric(s): {args.metrics}\n"
                        f"NOTE: Choose one or more of the following: {get_segmentation_metrics()}")

    # ------------------------------------------------------------------------------------------------------------------
    # Source directory setup
    # ------------------------------------------------------------------------------------------------------------------
    print("\n###############################################")
    print("Logging")
    print("###############################################\n")
    output_dir = f"{args.output_dir}/"

    # Run Name
    run = f"{get_now()}_{args.decoder_name}_{args.encoder_name}"

    # We'll also create folders in this source to hold results of the model
    run_dir = f"{output_dir}segmentation/{run}/"
    weights_dir = run_dir + "weights/"
    logs_dir = run_dir + "logs/"
    tensorboard_dir = logs_dir + "tensorboard/"

    # Make the directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Copy over the color map file to the run directory for users
    shutil.copyfile(args.color_map, f"{run_dir}{os.path.basename(args.color_map)}")

    print(f"NOTE: Model Run - {run}")
    print(f"NOTE: Model Directory - {run_dir}")
    print(f"NOTE: Weights Directory - {weights_dir}")
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

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_dir])
        url = tb.launch()

        print(f"NOTE: View Tensorboard at {url}")

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data, creating datasets
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Loading Data\n"
          f"#########################################\n")

    # Names of all images; sets to be split based on images
    image_names = dataframe['Name'].unique()

    print(f"NOTE: Found {len(image_names)} samples in dataset")

    # Split the Images into training, validation, and test sets.
    # We split based on the image names, so that we don't have the same image in multiple sets.
    training_images, testing_images = train_test_split(image_names, test_size=0.35, random_state=42)
    validation_images, testing_images = train_test_split(testing_images, test_size=0.5, random_state=42)

    # Create training, validation, and test dataframes
    train_df = dataframe[dataframe['Name'].isin(training_images)]
    valid_df = dataframe[dataframe['Name'].isin(validation_images)]
    test_df = dataframe[dataframe['Name'].isin(testing_images)]

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Output to logs
    train_df.to_csv(f"{logs_dir}Training_Set.csv", index=False)
    valid_df.to_csv(f"{logs_dir}Validation_Set.csv", index=False)
    test_df.to_csv(f"{logs_dir}Testing_Set.csv", index=False)

    print(f"NOTE: Number of samples in training set is {len(train_df['Name'].unique())}")
    print(f"NOTE: Number of samples in validation set is {len(valid_df['Name'].unique())}")
    print(f"NOTE: Number of samples in testing set is {len(test_df['Name'].unique())}")

    # ------------------------------------------------------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------------------------------------------------------
    # Sample augmentation techniques
    height, width = 736, 1280

    # Whether to include training augmentation
    if args.augment_data:
        training_augmentation = get_training_augmentation(height, width)
    else:
        training_augmentation = get_validation_augmentation(height, width)

    train_dataset = Dataset(
        train_df,
        augmentation=training_augmentation,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=class_ids,
    )

    valid_dataset = Dataset(
        valid_df,
        augmentation=get_validation_augmentation(height, width),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=class_ids,
    )

    # For visualizing progress
    valid_dataset_vis = Dataset(
        valid_df,
        augmentation=get_validation_augmentation(height, width),
        classes=class_ids,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------------------------------------------------------
    # Show sample of training data
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Viewing Training Samples\n"
          f"#########################################\n")

    # Create a sample version dataset
    sample_dataset = Dataset(train_df,
                             augmentation=training_augmentation,
                             classes=class_ids)

    # Loop through a few samples
    for i in range(5):

        try:
            # Get a random sample from dataset
            image, mask = sample_dataset[np.random.randint(0, len(train_df))]
            # Visualize and save to logs dir
            save_path = f'{tensorboard_dir}train/TrainingSample_{i}.png'
            visualize(save_path=save_path,
                      save_figure=True,
                      image=image,
                      mask=colorize_mask(mask, class_ids, class_colors))

            # Write to tensorboard
            train_writer.add_image(f'Training_Samples',
                                   np.array(Image.open(save_path)),
                                   dataformats="HWC",
                                   global_step=i)

        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------------------------------------------------

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

    try:

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

            # Visualize a validation sample on tensorboard
            n = np.random.choice(len(valid_dataset))
            # Get the image original image without preprocessing
            image_vis = valid_dataset_vis[n][0].numpy()
            # Get the expected input for model
            image, gt_mask = valid_dataset[n]
            gt_mask = gt_mask.squeeze().numpy()
            x_tensor = image.to(device).unsqueeze(0)
            # Make prediction
            pr_mask = model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            pr_mask = np.argmax(pr_mask, axis=0)

            try:
                # Visualize the colorized results locally
                save_path = f'{tensorboard_dir}valid/ValidResult_{e_idx}.png'
                visualize(save_path=save_path,
                          save_figure=True,
                          image=image_vis,
                          ground_truth_mask=colorize_mask(gt_mask, class_ids, class_colors),
                          predicted_mask=colorize_mask(pr_mask, class_ids, class_colors))

                # Log the visualization to TensorBoard
                figure = np.array(Image.open(save_path))
                valid_writer.add_image(f'Valid_Results', figure, dataformats="HWC", global_step=e_idx)

            except:
                pass

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

        if 'CUDA out of memory' in str(e):
            print(f"WARNING: Not enough GPU memory for the provided parameters")

        # Write the error to text file
        print(f"NOTE: Please see {logs_dir}Error.txt")
        with open(f"{logs_dir}Error.txt", 'a') as file:
            file.write(f"Caught exception: {str(traceback.print_exc())}\n")

        # Exit early
        raise Exception(f"ERROR: There was an issue with training!\n{e}")

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
    # Open an image using PIL to get the original dimensions
    original_width, original_height = Image.open(test_df.loc[0, 'Image Path']).size

    # Create test dataset
    test_dataset = Dataset(
        test_df,
        augmentation=get_validation_augmentation(height, width),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=class_ids,
    )

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

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

        # Catch the error
        print(f"ERROR: Could not calculate metrics")

        # Likely Memory
        if 'CUDA out of memory' in str(e):
            print(f"WARNING: Not enough GPU memory for the provided parameters")

        # Write the error to text file
        print(f"NOTE: Please see {logs_dir}Error.txt")
        with open(f"{logs_dir}Error.txt", 'a') as file:
            file.write(f"Caught exception: {str(e)}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize results
    # ------------------------------------------------------------------------------------------------------------------
    # Test dataset without preprocessing
    test_dataset_vis = Dataset(
        test_df,
        classes=class_ids,
    )

    try:
        # Empty cache from testing
        torch.cuda.empty_cache()

        # Loop through some samples
        for i in range(25):
            # Get a random sample
            n = np.random.choice(len(test_dataset))
            # Get the image original image without preprocessing
            image_vis = test_dataset_vis[n][0].numpy()
            # Get the expected input for model
            image, gt_mask = test_dataset[n]
            gt_mask = gt_mask.squeeze().numpy()
            gt_mask = cv2.resize(gt_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            gt_mask = colorize_mask(gt_mask, class_ids, class_colors)
            # Prepare sample
            x_tensor = image.to(device).unsqueeze(0)
            # Make prediction
            pr_mask = model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            pr_mask = np.argmax(pr_mask, axis=0)
            pr_mask = cv2.resize(pr_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            pr_mask = colorize_mask(pr_mask, class_ids, class_colors)

            try:
                # Visualize the colorized results locally
                save_path = f'{tensorboard_dir}test/TestResult_{i}.png'

                visualize(save_path=save_path,
                          save_figure=True,
                          image=image_vis,
                          ground_truth_mask=gt_mask,
                          predicted_mask=pr_mask)

                # Log the visualization to TensorBoard
                test_writer.add_image(f'Test_Results',
                                      np.array(Image.open(save_path)),
                                      dataformats="HWC",
                                      global_step=i)

            except:
                pass
    except Exception as e:

        # Catch the error
        print(f"ERROR: Could not make predictions")

        # Likely Memory
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


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument('--masks', type=str, required=True,
                        help='The path to the masks csv file')

    parser.add_argument('--color_map', type=str,
                        help='Path to Color Map JSON file')

    parser.add_argument('--pre_trained_path', type=str, default=None,
                        help='Path to pre-trained model of the same architecture')

    parser.add_argument('--encoder_name', type=str, default='mit_b0',
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

    parser.add_argument('--tensorboard', action='store_true',
                        help='Display training on Tensorboard')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store results')

    args = parser.parse_args()

    try:
        segmentation(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
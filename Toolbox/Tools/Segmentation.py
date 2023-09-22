import os
import sys
import json
import glob
import shutil
import inspect
import warnings
import argparse
import traceback
import subprocess

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
import segmentation_models_pytorch.utils

import albumentations as albu

from Common import log
from Common import get_now

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ------------------------------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------------------------------

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
        assert 'Name' in dataframe.columns, log(f"ERROR: 'Name' not found in mask file")
        assert 'Seg Path' in dataframe.columns, log(f"ERROR: 'Name' not found in mask file")
        assert 'Image Path' in dataframe.columns, log(f"ERROR: 'Seg Path' not found in mask file")

        self.ids = dataframe['Name'].to_list()
        self.masks_fps = dataframe['Seg Path'].to_list()
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

        # One hot encoded
        masks = [(mask == v) for v in self.class_ids]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

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
        options = smp.encoders.encoders.keys()
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
        import segmentation_models_pytorch.utils.losses as losses

        options = [attr for attr in dir(losses) if callable(getattr(losses, attr))]
        options = [o for o in options if o != 'Activation']
        loss_options = options

    except Exception as e:
        # Fail silently
        pass

    return loss_options


def get_segmentation_metrics():
    """

    """
    metric_options = []

    try:
        import segmentation_models_pytorch.utils.metrics as metrics

        options = [attr for attr in dir(metrics) if callable(getattr(metrics, attr))]
        options = [o for o in options if o != 'Activation']
        metric_options = options

    except Exception as e:
        # Fail silently
        pass

    return metric_options


def get_segmentation_optimizers():
    """

    """
    optimizer_options = []

    try:
        import torch.optim as optimizers

        options = [attr for attr in dir(optimizers) if callable(getattr(optimizers, attr))]
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
        log(f"NOTE: Figure saved to {save_path}")

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
        albu.Resize(height=height, width=width),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0, value=0),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=0),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

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
    """

    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def seg(args):
    """

    """
    log("\n###############################################")
    log("Semantic Segmentation")
    log("###############################################\n")

    # Check for CUDA
    log(f"NOTE: PyTorch version - {torch.__version__}")
    log(f"NOTE: Torchvision version - {torchvision.__version__}")
    log(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

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

        # Add Unlabeled class
        class_names.insert(0, "Unlabeled")
        class_ids.insert(0, 0)
        class_colors.insert(0, [0, 0, 0])

    else:
        log(f"ERROR: Color Mapping JSON file provided doesn't exist; check input provided")
        sys.exit(1)

    # Data
    if os.path.exists(args.masks):
        dataframe = pd.read_csv(args.masks)
    else:
        log(f"ERROR: Mask file provided does not exist; please check input")
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # Model building, parameters
    # ------------------------------------------------------------------------------------------------------------------
    log(f"\n#########################################\n"
        f"Building Model\n"
        f"#########################################\n")

    try:
        encoder_weights = 'imagenet'

        if args.encoder_name not in get_segmentation_encoders():
            raise Exception(f"ERROR: Encoder must be one of {get_segmentation_encoders()}")

        if args.decoder_name not in get_segmentation_decoders():
            raise Exception(f"ERROR: Decoder must be one of {get_segmentation_decoders()}")

        # Building model using user's input
        model = getattr(smp, args.decoder_name)(
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            classes=len(class_names),
            activation='softmax2d',
        )

        if args.freeze_encoder:
            log(f"NOTE: Freezing encoder weights")
            for param in model.encoder.parameters():
                param.requires_grad = False

        preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, encoder_weights)

        log(f"NOTE: Using {args.encoder_name} encoder with a {args.decoder_name} decoder")

    except Exception as e:
        log(f"ERROR: Could not build model\n{e}")
        sys.exit(1)

    try:
        # Get the loss function
        assert args.loss_function in get_segmentation_losses()
        loss_function = getattr(smp.utils.losses, args.loss_function)(ignore_channels=[0])

        log(f"NOTE: Using loss function {args.loss_function}")

    except Exception as e:
        log(f"ERROR: Could not get loss function\n{e}")
        sys.exit(1)

    try:
        # Get the optimizer
        assert args.optimizer in get_segmentation_optimizers()
        optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), args.learning_rate)

        log(f"NOTE: Using optimizer {args.optimizer}")

    except Exception as e:
        log(f"ERROR: Could not get optimizer\n{e}")
        sys.exit(1)

    try:
        # Get the metrics
        if not all(m in get_segmentation_metrics() for m in args.metrics):
            raise Exception

        metrics = [getattr(smp.utils.metrics, m)(threshold=0.5) for m in args.metrics]

        if 'IoU' not in args.metrics:
            metrics.append(smp.utils.metrics.IoU(threshold=0.5))

        log(f"NOTE: Using metrics {args.metrics}")

    except Exception as e:
        log(f"ERROR: Could not get metrics\n{e}")
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # Source directory setup
    # ------------------------------------------------------------------------------------------------------------------
    log("\n###############################################")
    log("Logging")
    log("###############################################\n")
    output_dir = f"{args.output_dir}\\"

    # Run Name
    run = f"{get_now()}_{args.decoder_name}_{args.encoder_name}"

    # We'll also create folders in this source to hold results of the model
    run_dir = f"{output_dir}segmentation\\{run}\\"
    samples_dir = run_dir + "samples\\"
    weights_dir = run_dir + "weights\\"
    logs_dir = run_dir + "logs\\"
    tensorboard_dir = logs_dir + "tensorboard\\"

    # Make the directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    log(f"NOTE: Model Run - {run}")
    log(f"NOTE: Model Directory - {run_dir}")
    log(f"NOTE: Samples Directory - {samples_dir}")
    log(f"NOTE: Weights Directory - {weights_dir}")
    log(f"NOTE: Log Directory - {logs_dir}")
    log(f"NOTE: Tensorboard Directory - {tensorboard_dir}")

    # Create a SummaryWriter for logging to tensorboard
    train_writer = SummaryWriter(log_dir=tensorboard_dir + "train")
    valid_writer = SummaryWriter(log_dir=tensorboard_dir + "valid")
    test_writer = SummaryWriter(log_dir=tensorboard_dir + "test")

    # Open tensorboard
    if args.tensorboard:

        log(f"\n#########################################\n"
            f"Tensorboard\n"
            f"#########################################\n")

        # Create a subprocess that opens tensorboard
        tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', tensorboard_dir],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)

        log("NOTE: View Tensorboard at 'http://localhost:6006'")

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data, creating datasets
    # ------------------------------------------------------------------------------------------------------------------
    log(f"\n#########################################\n"
        f"Loading Data\n"
        f"#########################################\n")

    # Names of all images; sets to be split based on images
    image_names = dataframe['Name'].unique()

    log(f"NOTE: Found {len(image_names)} samples in dataset")

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

    log(f"NOTE: Number of classes in training set is {len(train_df['Name'].unique())}")
    log(f"NOTE: Number of classes in validation set is {len(valid_df['Name'].unique())}")
    log(f"NOTE: Number of classes in testing set is {len(test_df['Name'].unique())}")

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # ------------------------------------------------------------------------------------------------------------------
    # Show sample of training data
    # ------------------------------------------------------------------------------------------------------------------
    log(f"\n#########################################\n"
        f"Viewing Training Samples\n"
        f"#########################################\n")

    # Create a sample version dataset
    sample_dataset = Dataset(train_df,
                             augmentation=training_augmentation,
                             classes=class_ids)

    # Loop through a few samples
    for i in range(5):
        # Get a random sample from dataset
        image, mask = sample_dataset[np.random.randint(0, len(train_df))]
        # Visualize and save to logs dir
        save_path = f'{tensorboard_dir}train\\TrainingSample_{i}.png'
        visualize(save_path=save_path,
                  save_figure=True,
                  image=image,
                  mask=colorize_mask(np.argmax(mask, axis=-1), class_ids, class_colors))

        # Write to tensorboard
        train_writer.add_image(f'Training_Samples', np.array(Image.open(save_path)), dataformats="HWC", global_step=i)

    # ------------------------------------------------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------------------------------------------------
    try:

        log(f"\n#########################################\n"
            f"Training\n"
            f"#########################################\n")

        log("NOTE: Starting Training")
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss_function,
            metrics=metrics,
            device=device,
            verbose=True,
        )

        best_score = float('inf')
        best_epoch = 0
        since_best = 0

        # Training loop
        for e_idx in range(1, args.num_epochs + 1):

            log(f"\nEpoch: {e_idx} / {args.num_epochs}")
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
            image_vis = valid_dataset_vis[n][0].astype('uint8')
            # Get the expected input for model
            image, gt_mask = valid_dataset[n]
            gt_mask = gt_mask.squeeze()
            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            # Make prediction
            pr_mask = model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())

            # Visualize the colorized results locally
            save_path = f'{tensorboard_dir}valid\\ValidResult_{e_idx}.png'

            visualize(save_path=save_path,
                      save_figure=True,
                      image=image_vis,
                      ground_truth_mask=colorize_mask(np.argmax(gt_mask, axis=-0), class_ids, class_colors),
                      predicted_mask=colorize_mask(np.argmax(pr_mask, axis=0), class_ids, class_colors))

            figure = np.array(Image.open(save_path))

            # Log the visualization to TensorBoard
            valid_writer.add_image(f'Valid_Results', figure, dataformats="HWC", global_step=e_idx)

            # Get the loss values
            train_loss = [v for k, v in train_logs.items() if 'loss' in k][0]
            valid_loss = [v for k, v in valid_logs.items() if 'loss' in k][0]

            # Update best
            if valid_loss < best_score:
                best_score = valid_loss
                best_epoch = e_idx
                since_best = 0
                log(f"NOTE: Current best epoch {e_idx}")

                # Save the model
                prefix = f'{weights_dir}model-{str(e_idx)}-'
                suffix = f'{str(np.around(train_logs["iou_score"], 4))}-{str(np.around(valid_logs["iou_score"], 4))}'
                path = prefix + suffix
                torch.save(model, f'{path}.pth')
                log(f'NOTE: Model saved to {path}')
            else:
                since_best += 1  # Increment the counter
                log(f"NOTE: Model did not improve after epoch {e_idx}")

            # Overfitting indication
            if train_loss < valid_loss:
                log(f"NOTE: Overfitting occurred in epoch {e_idx}")

            # Check if it's time to decrease the learning rate
            if since_best >= 3 and train_loss <= valid_loss:
                new_lr = optimizer.param_groups[0]['lr'] * 0.5
                optimizer.param_groups[0]['lr'] = new_lr
                log(f"NOTE: Decreased learning rate to {new_lr} after epoch {e_idx}")

            # Exit early if progress stops
            if since_best >= 7 and train_loss < valid_loss:
                log("NOTE: Model training plateaued; exiting training loop")
                break

    except KeyboardInterrupt:
        log("NOTE: Exiting training loop")

    except Exception as e:
        log(f"ERROR: There was an issue with training!")
        if 'CUDA out of memory' in str(e):
            log(f"WARNING: Not enough GPU memory for the provided parameters")

        log(f"NOTE: Please see {logs_dir}Error.txt")
        # Write the error to text file
        with open(f"{logs_dir}Error.txt", 'a') as file:
            file.write(f"Caught exception: {str(e)}\n")

        # Exit early
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # Load best model
    # ------------------------------------------------------------------------------------------------------------------
    weights = sorted(glob.glob(weights_dir + "*.pth"))
    best_weights = [w for w in weights if f'model-{str(best_epoch)}' in w][0]

    # Load into the model
    model = torch.load(best_weights)
    log(f"NOTE: Loaded best weights {best_weights}")

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate model on test set
    # ------------------------------------------------------------------------------------------------------------------
    # Open an image using PIL to get the original dimensions
    original_width, original_height = Image.open(test_df.loc[0, 'Image Path']).size

    # Calculate new width and height that are divisible by 32
    new_width = (original_width // 32) * 32
    new_height = (original_height // 32) * 32

    # Create test dataset
    test_dataset = Dataset(
        test_df,
        augmentation=get_validation_augmentation(new_height, new_width),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=class_ids,
    )

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Evaluate on the test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss_function,
        metrics=metrics,
        device=device,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------------------------------------------------------
    log(f"\n#########################################\n"
        f"Calculating Metrics\n"
        f"#########################################\n")

    # Score on test set
    test_logs = test_epoch.run(test_loader)

    # Log test metrics
    for key, value in test_logs.items():
        test_writer.add_scalar(key, value, global_step=best_epoch)

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize results
    # ------------------------------------------------------------------------------------------------------------------
    # Test dataset without preprocessing
    test_dataset_vis = Dataset(
        test_df,
        classes=class_ids,
    )

    # Loop through some samples
    for i in range(10):
        # Get a random sample
        n = np.random.choice(len(test_dataset))
        # Get the image original image without preprocessing
        image_vis = test_dataset_vis[n][0].astype('uint8')
        # Get the expected input for model
        image, gt_mask = test_dataset[n]
        gt_mask = gt_mask.squeeze()
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        # Make prediction
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # Visualize the colorized results locally
        save_path = f'{tensorboard_dir}test\\TestResult_{i}.png'

        visualize(save_path=save_path,
                  save_figure=True,
                  image=image_vis,
                  ground_truth_mask=colorize_mask(np.argmax(gt_mask, axis=-0), class_ids, class_colors),
                  predicted_mask=colorize_mask(np.argmax(pr_mask, axis=0), class_ids, class_colors))

        # Log the visualization to TensorBoard
        test_writer.add_image(f'Test_Results', np.array(Image.open(save_path)), dataformats="HWC", global_step=i)

    log(f"NOTE: Saving best weights in {run_dir}")
    shutil.copyfile(best_weights, f"{run_dir}Best_Model_and_Weights.pth")

    # Close tensorboard writers
    for writer in [train_writer, valid_writer, test_writer]:
        writer.close()

    # Close tensorboard
    if args.tensorboard:
        log("NOTE: Closing Tensorboard")
        tensorboard_process.terminate()


def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument('--masks', type=str, required=True,
                        help='The path to the masks csv file')

    parser.add_argument('--color_map', type=str,
                        help='Path to Color Map JSON file')

    parser.add_argument('--encoder_name', type=str, default='mit_b0',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--decoder_name', type=str, default='Unet',
                        help='The convolutional decoder')

    parser.add_argument('--metrics', type=str, nargs='+', default=get_segmentation_metrics(),
                        help='The metrics to evaluate the model')

    parser.add_argument('--loss_function', type=str, default='JaccardLoss',
                        help='The loss function to use to train the model')

    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Only train the decoder weights during training')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='The optimizer to use to train the model')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Starting learning rate')

    parser.add_argument('--augment_data', action='store_true',
                        help='Apply affine augmentations to training data')

    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Starting learning rate')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of samples per batch during training')

    parser.add_argument('--tensorboard', action='store_true',
                        help='Display training on Tensorboard')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store results')

    args = parser.parse_args()

    try:
        seg(args)
        log("Done.\n")

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())


if __name__ == '__main__':
    main()

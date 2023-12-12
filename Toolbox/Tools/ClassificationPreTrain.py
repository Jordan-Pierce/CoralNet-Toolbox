import os
import sys
import argparse
import traceback

import cv2
import pandas as pd
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp

from simclr import SimCLR
from simclr.modules import NT_Xent
from simclr.modules.transformations import TransformsSimCLR

import albumentations as albu

from tensorboard import program

from Common import get_now
from Classification import get_classifier_optimizers


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

class CustomModel(torch.nn.Module):
    def __init__(self, encoder_name, weights):
        super(CustomModel, self).__init__()

        # Name
        self.name = encoder_name
        # Pre-trained encoder
        self.encoder = smp.encoders.get_encoder(name=encoder_name,
                                                weights=weights)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(self.encoder.out_channels[-1], 64)

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
        # Fully connected layer for classification
        x = self.fc(x)

        return x


class Dataset(BaseDataset):
    """

    """

    def __init__(
            self,
            dataframe,
            transform=None,
            preprocessing=None,
            device=None,
    ):
        assert 'Name' in dataframe.columns, print(f"ERROR: 'Name' not found in Patches file")
        assert 'Path' in dataframe.columns, print(f"ERROR: 'Path' not found in Patches file")

        self.ids = dataframe['Name'].to_list()
        self.patches = dataframe['Path'].to_list()

        self.transform = transform
        self.preprocessing = preprocessing

        self.device = device

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.patches[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transform
        image = Image.fromarray(image)
        x_i, x_j = self.transform(image)

        # apply preprocessing
        if self.preprocessing:
            # Rescale to [0, 255]
            x_i_rescaled = (x_i * 255).to(torch.uint8)
            x_j_rescaled = (x_j * 255).to(torch.uint8)

            # Transpose dimensions to match the desired shape (224, 224, 3)
            x_i_rescaled = x_i_rescaled.permute(1, 2, 0)
            x_j_rescaled = x_j_rescaled.permute(1, 2, 0)

            x_i = self.preprocessing(x_i_rescaled).permute(2, 0, 1).float()
            x_j = self.preprocessing(x_j_rescaled).permute(2, 0, 1).float()

        if self.device != 'cpu':
            x_i, x_j = x_i.to(self.device), x_j.to(self.device)

        return x_i, x_j

    def __len__(self):
        return len(self.ids)


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def save_model(model, epoch, weights_dir):
    """

    """
    output_path = f"{weights_dir}checkpoint_{epoch}.pth"
    torch.save(model, output_path)


def train_epoch(args, train_loader, model, criterion, optimizer, writer):
    """

    """
    loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):

        optimizer.zero_grad()

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        # calculate loss
        loss = criterion(z_i, z_j)
        loss.backward()

        # back prop
        optimizer.step()

        if args.nr == 0 and step % 10 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()

    return loss_epoch


def classification_pretrain(args):
    """

    """
    print("\n###############################################")
    print(f"Pre-Train")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------------------------------------------------------
    # Building Model
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Building Model\n"
          f"#########################################\n")

    try:
        # Get encoder
        encoder_weights = 'imagenet'
        model = CustomModel(encoder_name=args.encoder_name, weights=encoder_weights)
        print(f"NOTE: Using {args.encoder_name} encoder")

        # Processing function for encoder
        preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, encoder_weights)

        # Freezing percentage of the encoder
        num_params = len(list(model.encoder.parameters()))
        freeze_params = int(num_params * args.freeze_encoder)

        # Give users the ability to freeze N percent of the encoder
        print(f"NOTE: Freezing {args.freeze_encoder}% of encoder weights")
        for idx, param in enumerate(model.encoder.parameters()):
            if idx < freeze_params:
                param.requires_grad = False

        # Initialize model
        n_features = model.fc.in_features
        model = SimCLR(model, args.projection_dim, n_features)
        model = model.to(device)

    except Exception as e:
        raise Exception(f"ERROR: Could not build model\n{e}")

    try:

        # Optimizer / loss
        args.temperature = 0.5

        # Get the optimizer
        assert args.optimizer in get_classifier_optimizers()
        optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), args.learning_rate)

        print(f"NOTE: Using optimizer function {args.optimizer}")

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.num_epochs, eta_min=0, last_epoch=-1
        )

        criterion = NT_Xent(args.batch_size, args.temperature, 1)

    except Exception as e:
        raise Exception(f"ERROR: Could not load optimizer\n{e}")

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
    run_dir = f"{output_dir}pretrain\\{run}\\"
    weights_dir = run_dir + "weights\\"
    logs_dir = run_dir + "logs\\"
    tensorboard_dir = logs_dir + "tensorboard\\"

    # Make the directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    print(f"NOTE: Model Run - {run}")
    print(f"NOTE: Model Directory - {run_dir}")
    print(f"NOTE: Log Directory - {logs_dir}")
    print(f"NOTE: Tensorboard Directory - {tensorboard_dir}")

    # Create a SummaryWriter for logging to tensorboard
    writer = SummaryWriter(log_dir=tensorboard_dir + "train")

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
    print("\n###############################################")
    print("Creating Dataset")
    print("###############################################\n")

    # If the user provides multiple patch dataframes
    train_df = pd.DataFrame()

    for patches_path in args.patches:
        if os.path.exists(patches_path):
            # Patch dataframe
            patches = pd.read_csv(patches_path, index_col=0)
            patches = patches.dropna()
            train_df = pd.concat((train_df, patches))
        else:
            raise Exception(f"ERROR: Patches dataframe {patches_path} does not exist")

    # Dataset
    train_dataset = Dataset(
        train_df,
        transform=TransformsSimCLR(224),
        preprocessing=preprocessing_fn,
        device=device,
    )

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    print(f"NOTE: Number of images {len(train_df)}")

    args.nr = 0
    args.global_step = 0
    args.current_epoch = 0

    # ------------------------------------------------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n#########################################\n"
          f"Pre-Training\n"
          f"#########################################\n")

    for epoch in range(args.num_epochs):

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train_epoch(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(model, epoch, weights_dir)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.num_epochs}]\t "
                f"Loss: {loss_epoch / len(train_loader)}\t "
                f"lr: {round(lr, 5)}\n"
            )
            args.current_epoch += 1

    # end training
    save_model(model, args.num_epochs, weights_dir)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pre-train an Encoder')

    parser.add_argument('--patches', required=True, nargs="+",
                        help='The path to the patch labels csv file output the Patches tool')

    parser.add_argument('--encoder_name', type=str, default='resnet18',
                        help='The convolutional encoder to fine-tune; pretrained on Imagenet')

    parser.add_argument('--freeze_encoder', type=float, default=0.5,
                        help='Freeze N% of the encoder [0 - 1]')

    parser.add_argument('--projection_dim', type=int, default=64,
                        help='Projection head dimensionality into latent space')

    parser.add_argument('--optimizer', default="Adam",
                        help='Optimizer for training the model')

    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Starting learning rate')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Starting learning rate')

    parser.add_argument('--tensorboard', action='store_true',
                        help='Display training on Tensorboard')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')

    args = parser.parse_args()

    try:
        classification_pretrain(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
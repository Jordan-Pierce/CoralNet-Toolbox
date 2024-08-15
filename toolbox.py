import os
import signal
import traceback

from gooey import Gooey, GooeyParser

from Toolbox.Tools.API import api
from Toolbox.Tools.Download import download
from Toolbox.Tools.Labelset import labelset
from Toolbox.Tools.Upload import upload

from Toolbox.Tools.Patches import patches
from Toolbox.Tools.Visualize import visualize
from Toolbox.Tools.Points import points
from Toolbox.Tools.Projector import projector
from Toolbox.Tools.Spotlight import spotlight

from Toolbox.Tools.ClassificationPreTrain import classification_pretrain
from Toolbox.Tools.Classification import classification
from Toolbox.Tools.Segmentation import segmentation
from Toolbox.Tools.ClassificationInference import classification_inference
from Toolbox.Tools.SegmentationInference import segmentation_inference
from Toolbox.Tools.ViscoreInference import viscore_inference

from Toolbox.Tools.SAM import sam
from Toolbox.Tools.SfM import sfm
from Toolbox.Tools.Segmentation3D import segmentation3d

# For Gooey dropdown
from Toolbox.Tools.Download import get_updated_labelset_list
from Toolbox.Tools.Classification import get_classifier_losses
from Toolbox.Tools.Classification import get_classifier_metrics
from Toolbox.Tools.Classification import get_classifier_encoders
from Toolbox.Tools.Classification import get_classifier_optimizers
from Toolbox.Tools.Segmentation import get_segmentation_losses
from Toolbox.Tools.Segmentation import get_segmentation_metrics
from Toolbox.Tools.Segmentation import get_segmentation_encoders
from Toolbox.Tools.Segmentation import get_segmentation_decoders
from Toolbox.Tools.Segmentation import get_segmentation_optimizers

from Toolbox.Tools.Common import console_user
from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import PATCH_EXTRACTOR
from Toolbox.Tools.Common import FUNC_GROUPS_LIST

from Toolbox.Tools.Patch_Extractor.Patch_Extractor import patch_extractor


# ----------------------------------------------------------------------------------------------------------------------
# Gooey GUI
# ----------------------------------------------------------------------------------------------------------------------

@Gooey(dump_build_config=True,
       program_name="CoralNet Toolbox",
       default_size=(900, 600),  # width, height
       console=True,
       richtext_controls=True,
       shutdown_signal=signal.CTRL_C_EVENT,
       progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
       progress_expr="current / total * 100",
       hide_progress_msg=True,
       timing_options={
           'show_time_remaining': True,
           'hide_time_remaining_on_complete': True,
       })
def main():
    desc = 'Interact with CoralNet, unofficially.'
    parser = GooeyParser(description=desc)

    parser.add_argument('--verbose', help='be verbose', dest='verbose',
                        action='store_true', default=False)

    subs = parser.add_subparsers(help='commands', dest='command')

    # ------------------------------------------------------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------------------------------------------------------
    api_parser = subs.add_parser('API')

    # Panel 1 - CoralNet model API
    api_parser_panel_1 = api_parser.add_argument_group('Access CoralNet Model API',
                                                       'Specify the Source by the ID, and provide one or '
                                                       'more CSV file(s) containing the names of images to '
                                                       'perform predictions on. Images must already exist in '
                                                       'the Source, and CSV file(s) must contain the following '
                                                       'fields: Name, Row, Column.',
                                                       gooey_options={'show_border': True})

    api_parser_panel_1.add_argument('--username', type=str,
                                    metavar="Username",
                                    default=os.getenv('CORALNET_USERNAME'),
                                    help='Username for CoralNet account')

    api_parser_panel_1.add_argument('--password', type=str,
                                    metavar="Password",
                                    default=os.getenv('CORALNET_PASSWORD'),
                                    help='Password for CoralNet account',
                                    widget="PasswordField")

    api_parser_panel_1.add_argument('--remember_username', action="store_false",
                                    metavar="Remember Username",
                                    help='Store Username as an Environmental Variable',
                                    widget="BlockCheckbox")

    api_parser_panel_1.add_argument('--remember_password', action="store_false",
                                    metavar="Remember Password",
                                    help='Store Password as an Environmental Variable',
                                    widget="BlockCheckbox")

    api_parser_panel_1.add_argument('--source_id_1', type=str, required=True,
                                    metavar="Source ID (for images)",
                                    help='The ID of the Source containing images.')

    api_parser_panel_1.add_argument('--source_id_2', type=str, required=False,
                                    metavar="Source ID (for model)",
                                    default=None,
                                    help='The ID of the Source containing model.')

    api_parser_panel_1.add_argument('--points', required=True, type=str,
                                    metavar="Points File",
                                    help='A path to a csv file containing the following: Name, Row, Column',
                                    widget="FileChooser")

    api_parser_panel_1.add_argument('--prefix', required=False, default="",
                                    metavar='Image Name Prefix',
                                    help='A prefix that all images of interest have in common to '
                                         'narrow the search space, else leave blank')

    api_parser_panel_1.add_argument('--output_dir', required=True,
                                    metavar='Output Directory',
                                    default=DATA_DIR,
                                    help='A root directory where all predictions will be saved to.',
                                    widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------------------------------------------------------
    download_parser = subs.add_parser('Download')

    # Panel 1 - Download CoralNet Data given a source
    download_parser_panel_1 = download_parser.add_argument_group('Download By Source ID',
                                                                 'Specify each Source to download by providing the '
                                                                 'associated ID; for multiple sources, add a space '
                                                                 'between each.',
                                                                 gooey_options={'show_border': True})

    download_parser_panel_1.add_argument('--username', type=str,
                                         metavar="Username",
                                         default=os.getenv('CORALNET_USERNAME'),
                                         help='Username for CoralNet account')

    download_parser_panel_1.add_argument('--password', type=str,
                                         metavar="Password",
                                         default=os.getenv('CORALNET_PASSWORD'),
                                         help='Password for CoralNet account',
                                         widget="PasswordField")

    download_parser_panel_1.add_argument('--remember_username', action="store_false",
                                         metavar="Remember Username",
                                         help='Store Username as an Environmental Variable',
                                         widget="BlockCheckbox")

    download_parser_panel_1.add_argument('--remember_password', action="store_false",
                                         metavar="Remember Password",
                                         help='Store Password as an Environmental Variable',
                                         widget="BlockCheckbox")

    download_parser_panel_1.add_argument('--source_ids', type=str, nargs='+', required=False,
                                         metavar="Source IDs", default=[],
                                         help='A list of source IDs to download (provide spaces between each ID).')

    download_parser_panel_1.add_argument('--output_dir', required=True,
                                         metavar='Output Directory',
                                         default=DATA_DIR,
                                         help='A root directory where all downloads will be saved to.',
                                         widget="DirChooser")

    download_parser_panel_1.add_argument('--headless', action="store_true",
                                         default=False,
                                         metavar="Run in Background",
                                         help='Run browser in headless mode',
                                         widget='BlockCheckbox')

    # Panel 2 - Download CoralNet Source ID and Labelset Lists
    download_parser_panel_2 = download_parser.add_argument_group('Download CoralNet Dataframes',
                                                                 'In addition to downloading Source data, dataframes '
                                                                 'containing information on all public Sources and '
                                                                 'Labelsets can also be downloaded.',
                                                                 gooey_options={'show_border': True})

    download_parser_panel_2.add_argument('--source_df', action="store_true",
                                         metavar="Download Source Dataframe",
                                         help='Information on all public Sources.',
                                         widget='BlockCheckbox')

    download_parser_panel_2.add_argument('--labelset_df', action="store_true",
                                         metavar="Download Labelset Dataframe",
                                         help='Information on all Labelsets.',
                                         widget='BlockCheckbox')

    download_parser_panel_2.add_argument('--sources_with', type=str, nargs='+', required=False,
                                         metavar="Download Sources 'With' Dataframe",
                                         help='Download Dataframe of Sources with specific labelsets',
                                         widget='Listbox', choices=get_updated_labelset_list())

    # ------------------------------------------------------------------------------------------------------------------
    # Labelset
    # ------------------------------------------------------------------------------------------------------------------
    labelset_parser = subs.add_parser('Labelset')

    # Panel 1 - Upload CoralNet Data given a source
    labelset_parser_panel_1 = labelset_parser.add_argument_group('Create a new CoralNet Labelset',
                                                                 'Provide a name, short code, functional group, '
                                                                 'description, and a representative image of your new '
                                                                 'labelset. The labelset will be created for the '
                                                                 'source provided the ID, but will be available '
                                                                 'globally.',
                                                                 gooey_options={'show_border': True})

    labelset_parser_panel_1.add_argument('--username', type=str,
                                         metavar="Username",
                                         default=os.getenv('CORALNET_USERNAME'),
                                         help='Username for CoralNet account')

    labelset_parser_panel_1.add_argument('--password', type=str,
                                         metavar="Password",
                                         default=os.getenv('CORALNET_PASSWORD'),
                                         help='Password for CoralNet account',
                                         widget="PasswordField")

    labelset_parser_panel_1.add_argument('--remember_username', action="store_false",
                                         metavar="Remember Username",
                                         help='Store Username as an Environmental Variable',
                                         widget="BlockCheckbox")

    labelset_parser_panel_1.add_argument('--remember_password', action="store_false",
                                         metavar="Remember Password",
                                         help='Store Password as an Environmental Variable',
                                         widget="BlockCheckbox")

    labelset_parser_panel_1.add_argument('--source_id', type=str, required=True,
                                         metavar="Source ID",
                                         help='The ID of the source to upload data to')

    labelset_parser_panel_1.add_argument('--labelset_name', type=str, required=True,
                                         metavar="Labelset Name",
                                         help='The name of the labelset to create')

    labelset_parser_panel_1.add_argument('--short_code', type=str, required=True,
                                         metavar="Short Code",
                                         help='The short code of the labelset to create')

    labelset_parser_panel_1.add_argument('--func_group', type=str, required=True,
                                         metavar="Functional Group",
                                         help='The functional group of the labelset to create',
                                         widget='Dropdown', choices=FUNC_GROUPS_LIST)

    labelset_parser_panel_1.add_argument('--desc', type=str, required=True,
                                         metavar="Description",
                                         help='The Description of the labelset to create')

    labelset_parser_panel_1.add_argument('--image_path', type=str, required=True,
                                         metavar="Thumbnail Path",
                                         help='The path to the image of the labelset to create',
                                         widget="FileChooser")

    labelset_parser_panel_1.add_argument('--headless', action="store_true",
                                         default=False,
                                         metavar="Run in Background",
                                         help='Run browser in headless mode',
                                         widget='BlockCheckbox')

    # ------------------------------------------------------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------------------------------------------------------
    upload_parser = subs.add_parser('Upload')

    # Panel 1 - Upload CoralNet Data given a source
    upload_parser_panel_1 = upload_parser.add_argument_group('Upload By Source ID',
                                                             'Upload data to a Source by providing the ID. Data that '
                                                             'can be uploaded includes images, labelsets, and '
                                                             'annotations.',
                                                             gooey_options={'show_border': True})

    upload_parser_panel_1.add_argument('--username', type=str,
                                       metavar="Username",
                                       default=os.getenv('CORALNET_USERNAME'),
                                       help='Username for CoralNet account')

    upload_parser_panel_1.add_argument('--password', type=str,
                                       metavar="Password",
                                       default=os.getenv('CORALNET_PASSWORD'),
                                       help='Password for CoralNet account',
                                       widget="PasswordField")

    upload_parser_panel_1.add_argument('--remember_username', action="store_false",
                                       metavar="Remember Username",
                                       help='Store Username as an Environmental Variable',
                                       widget="BlockCheckbox")

    upload_parser_panel_1.add_argument('--remember_password', action="store_false",
                                       metavar="Remember Password",
                                       help='Store Password as an Environmental Variable',
                                       widget="BlockCheckbox")

    upload_parser_panel_1.add_argument('--source_id', type=str, required=True,
                                       metavar="Source ID",
                                       help='The ID of the source to upload data to.')

    upload_parser_panel_1.add_argument('--images', required=False, default="",
                                       metavar='Image Directory',
                                       help='Directory containing images to upload.',
                                       widget="DirChooser")

    upload_parser_panel_1.add_argument('--prefix', required=False, default="",
                                       metavar='Image Name Prefix',
                                       help='A prefix to add to each image basename.')

    upload_parser_panel_1.add_argument('--labelset', required=False, type=str, default="",
                                       metavar="Labelset File",
                                       help='A path to the source labelset csv file. '
                                            'The file should contain the following: Label ID, Short Code',
                                       widget="FileChooser")

    upload_parser_panel_1.add_argument('--annotations', required=False, type=str, default="",
                                       metavar="Annotation File",
                                       help='A path the annotation csv file. '
                                            'The file should contain the following: Name, Row, Column, Label',
                                       widget="FileChooser")

    upload_parser_panel_1.add_argument('--headless', action="store_true",
                                       default=False,
                                       metavar="Run in Background",
                                       help='Run browser in headless mode',
                                       widget='BlockCheckbox')

    # ------------------------------------------------------------------------------------------------------------------
    # Annotate
    # ------------------------------------------------------------------------------------------------------------------
    annotate_parser = subs.add_parser('Annotate')

    # Panel 1
    annotate_parser.add_argument_group('Annotate',
                                       'Extract patches manually, which can be used as '
                                       'annotations in CoralNet, or used to train a model '
                                       'locally.',
                                       gooey_options={'show_border': True})

    # ------------------------------------------------------------------------------------------------------------------
    # Patches
    # ------------------------------------------------------------------------------------------------------------------
    patches_parser = subs.add_parser('Patches')

    # Panel 1
    patches_parser_panel_1 = patches_parser.add_argument_group('Crop Patches',
                                                               'Use the following to convert CoralNet formatted '
                                                               'annotation files into patches for training.',
                                                               gooey_options={'show_border': True})

    patches_parser_panel_1.add_argument('--image_dir', required=False,
                                        metavar='Image Directory',
                                        help='Directory containing images associated with annotation file',
                                        widget="DirChooser")

    patches_parser_panel_1.add_argument('--annotation_file', required=False, type=str,
                                        metavar="Annotation File",
                                        help='CoralNet annotation file, or one created using the Annotation tool',
                                        widget="FileChooser")

    patches_parser_panel_1.add_argument("--image_column", type=str, default="Name",
                                        metavar="Image Column",
                                        help="The column specifying the image basename")

    patches_parser_panel_1.add_argument("--label_column", type=str, default="Label",
                                        metavar="Label Column",
                                        help="The column specifying the Label column",
                                        widget='Dropdown',
                                        choices=['Label'] + [f'Machine suggestion {n + 1}' for n in range(5)])

    patches_parser_panel_1.add_argument("--patch_size", type=int, default=112,
                                        metavar="Patch Extent",
                                        help="The extent of each patch extracted")

    # Panel 2
    patches_parser_panel_2 = patches_parser.add_argument_group('Output',
                                                               'Specify the output directory',
                                                               gooey_options={'show_border': True})

    patches_parser_panel_2.add_argument('--output_name', required=False, type=str,
                                        metavar='Output Name',
                                        default="",
                                        help='Optional name for output directory; else timestamped.')

    patches_parser_panel_2.add_argument('--output_dir', required=True,
                                        metavar='Output Directory',
                                        default=DATA_DIR,
                                        help='Root directory where output will be saved',
                                        widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------------------------------------------------------
    visualize_parser = subs.add_parser('Visualize')

    # Panel 1
    visualize_parser_panel_1 = visualize_parser.add_argument_group('Visualize',
                                                                   'View annotations superimposed on each image; toggle'
                                                                   'annotations to be seen as points or squares.',
                                                                   gooey_options={'show_border': True})

    visualize_parser_panel_1.add_argument('--image_dir', required=True,
                                          metavar='Image Directory',
                                          help='A directory where all images are located.',
                                          widget="DirChooser")

    visualize_parser_panel_1.add_argument('--annotations', required=False, type=str,
                                          metavar="Annotations",
                                          help='The path to the annotations dataframe',
                                          widget="FileChooser")

    visualize_parser_panel_1.add_argument("--label_column", required=False, type=str, default='Label',
                                          metavar="Label Column",
                                          help='Label column in Annotations dataframe',
                                          widget='Dropdown',
                                          choices=['Label'] + [f'Machine suggestion {n + 1}' for n in range(5)])

    visualize_parser_panel_1.add_argument('--output_dir',
                                          metavar='Output Directory',
                                          required=True,
                                          help='Root directory where output will be saved',
                                          widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------------------------------------------------------------
    projector_parser = subs.add_parser('Projector')

    # Panel 1
    projector_parser_panel_1 = projector_parser.add_argument_group('Projector',
                                                                   'Display patch data in feature space using '
                                                                   'Tensorboard. Requires a trained model, and a patch '
                                                                   'file.',
                                                                   gooey_options={'show_border': True})

    projector_parser_panel_1.add_argument("--model", type=str,
                                          metavar="Model Path",
                                          help="Path to Best Model and Weights File (.pth)",
                                          widget="FileChooser")

    projector_parser_panel_1.add_argument('--patches', type=str,
                                          metavar="Patch Data",
                                          help='Patches dataframe file',
                                          widget="FileChooser")

    projector_parser_panel_1.add_argument('--output_dir', required=True,
                                          metavar='Output Directory',
                                          default=DATA_DIR,
                                          help='Root directory where output will be saved',
                                          widget="DirChooser")

    projector_parser_panel_2 = projector_parser.add_argument_group('Existing Project',
                                                                   gooey_options={'show_border': True})

    projector_parser_panel_2.add_argument('--project_folder', type=str, default="",
                                          metavar="Project Folder",
                                          help='Path to existing projector project folder.',
                                          widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Spotlight
    # ------------------------------------------------------------------------------------------------------------------
    spotlight_parser = subs.add_parser('Spotlight')

    # Panel 1
    spotlight_parser_panel_1 = spotlight_parser.add_argument_group('Model',
                                                                   'Choose either a pre-trained model (.pth) OR an '
                                                                   'encoder, not both.',
                                                                   gooey_options={'show_border': True})

    spotlight_parser_panel_1.add_argument('--encoder_name', type=str,
                                          metavar="Pretrained Encoder",
                                          help='Encoder, pre-trained on ImageNet dataset.',
                                          widget='Dropdown', choices=get_classifier_encoders())

    spotlight_parser_panel_1.add_argument('--pre_trained_path',
                                          metavar="Pretrained Path",
                                          help='Path to pre-trained model.',
                                          widget="FileChooser")

    spotlight_parser_panel_2 = spotlight_parser.add_argument_group('Patches',
                                                                   'Display patch data in feature space using '
                                                                   'Spotlight. Requires a patch file.',
                                                                   gooey_options={'show_border': True})

    spotlight_parser_panel_2.add_argument('--patches', type=str,
                                          metavar="Patch Data",
                                          help='Patches dataframe file',
                                          widget="FileChooser")

    spotlight_parser_panel_2.add_argument('--batch_size', type=int, default=128,
                                          metavar="Batch Size",
                                          help='The number of samples per batch; GPU dependent.')

    spotlight_parser_panel_2.add_argument('--frac', type=int,
                                          metavar='Percentage',
                                          default=50,
                                          help='Percentage of samples to use.',
                                          widget='Slider',
                                          gooey_options={'min': 0, 'max': 100, 'increment': 1})

    # ------------------------------------------------------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------------------------------------------------------
    points_parser = subs.add_parser('Points')

    # Panel 1
    points_parser_panel_1 = points_parser.add_argument_group('Points',
                                                             'Sample points for each image; points can be used '
                                                             'with the API or Inference tools. Methods include '
                                                             'uniform, random, and stratified-random. Samples are '
                                                             'saved in the Output Directory as a dataframe.',
                                                             gooey_options={'show_border': True})

    points_parser_panel_1.add_argument('--images', required=True, type=str,
                                       metavar="Image Directory",
                                       help='Directory of images to create sampled points for',
                                       widget="DirChooser")

    points_parser_panel_1.add_argument('--output_dir',
                                       metavar='Output Directory',
                                       default=DATA_DIR,
                                       help='Root directory where output will be saved',
                                       widget="DirChooser")

    points_parser_panel_1.add_argument('--sample_method', required=True, type=str,
                                       metavar="Sample Method",
                                       help='Method used to sample points from each image',
                                       widget="Dropdown", choices=['Uniform', 'Random', 'Stratified'])

    points_parser_panel_1.add_argument('--num_points', required=True, type=int, default=2048,
                                       metavar="Number of Points",
                                       help='The number of points to sample from each image')

    # ------------------------------------------------------------------------------------------------------------------
    # Classification Pretrain
    # ------------------------------------------------------------------------------------------------------------------
    classification_pre_parser = subs.add_parser('ClassificationPretrain')

    # Panel 1
    classification_pre_parser_panel_1 = classification_pre_parser.add_argument_group('Classification Pretrain',
                                                                                     'Use the following to pretrain '
                                                                                     'your own CNN encoder; useful for '
                                                                                     'classification and semantic '
                                                                                     'segmentation models.',
                                                                                     gooey_options={
                                                                                         'show_border': True})

    classification_pre_parser_panel_1.add_argument('--patches', required=True, nargs="+",
                                                   metavar="Patch Data",
                                                   help='Patches dataframe file(s)',
                                                   widget="MultiFileChooser")

    classification_pre_parser_panel_1.add_argument('--output_dir', required=True,
                                                   metavar='Output Directory',
                                                   default=DATA_DIR,
                                                   help='Root directory where output will be saved',
                                                   widget="DirChooser")

    # Panel 2
    classification_pre_parser_panel_2 = classification_pre_parser.add_argument_group('Parameters',
                                                                                     'Choose the parameters for model '
                                                                                     'training',
                                                                                     gooey_options={
                                                                                         'show_border': True})

    classification_pre_parser_panel_2.add_argument('--encoder_name', type=str, required=True,
                                                   metavar="Pretrained Encoder",
                                                   help='Encoder, pre-trained on ImageNet dataset',
                                                   widget='Dropdown', choices=get_classifier_encoders())

    classification_pre_parser_panel_2.add_argument('--freeze_encoder', type=int,
                                                   metavar="Freeze Encoder",
                                                   help='Freeze encoder by N%',
                                                   widget='Slider',
                                                   gooey_options={'min': 0, 'max': 100, 'increment': 1})

    classification_pre_parser_panel_2.add_argument('--projection_dim', type=int, default=64,
                                                   metavar="Projection Dimensions",
                                                   help='Projection head dimensionality into latent space')

    classification_pre_parser_panel_2.add_argument('--optimizer', type=str, required=True,
                                                   metavar="Optimizer",
                                                   help='Optimizer used for training model',
                                                   widget='Dropdown', choices=get_classifier_optimizers())

    classification_pre_parser_panel_2.add_argument('--learning_rate', type=float, default=0.0005,
                                                   metavar="Learning Rate",
                                                   help='The value used to incrementally adjust model weights')

    classification_pre_parser_panel_2.add_argument('--num_epochs', type=int, default=25,
                                                   metavar="Number of Epochs",
                                                   help='The number of iterations the model trains for')

    classification_pre_parser_panel_2.add_argument('--batch_size', type=int, default=128,
                                                   metavar="Batch Size",
                                                   help='The number of samples per batch; GPU dependent')

    classification_pre_parser_panel_2.add_argument('--tensorboard', action='store_true',
                                                   metavar="Tensorboard",
                                                   help='Open Tensorboard for viewing model training in real-time',
                                                   widget='BlockCheckbox')

    # ------------------------------------------------------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------------------------------------------------------
    classification_parser = subs.add_parser('Classification')

    # Panel 1
    classification_parser_panel_1 = classification_parser.add_argument_group('Image Classification',
                                                                             'Use the following to train your own '
                                                                             'patch-based image classifier.',
                                                                             gooey_options={'show_border': True})

    classification_parser_panel_1.add_argument('--patches', required=True, nargs="+",
                                               metavar="Patch Data",
                                               help='Patches dataframe file(s)',
                                               widget="MultiFileChooser")

    classification_parser_panel_1.add_argument('--pre_trained_path', required=False,
                                               metavar="Pretrained Path",
                                               help='Path to pre-trained model',
                                               widget="FileChooser")

    classification_parser_panel_1.add_argument('--output_dir', required=True,
                                               metavar='Output Directory',
                                               default=DATA_DIR,
                                               help='Root directory where output will be saved',
                                               widget="DirChooser")

    # Panel 2
    classification_parser_panel_2 = classification_parser.add_argument_group('Parameters',
                                                                             'Choose the parameters for model training',
                                                                             gooey_options={'show_border': True})

    classification_parser_panel_2.add_argument('--encoder_name', type=str, required=True,
                                               metavar="Pretrained Encoder",
                                               help='Encoder, pre-trained on ImageNet dataset',
                                               widget='Dropdown', choices=get_classifier_encoders())

    classification_parser_panel_2.add_argument('--freeze_encoder', type=int,
                                               metavar="Freeze Encoder",
                                               help='Freeze encoder by N%',
                                               widget='Slider', gooey_options={'min': 0, 'max': 100, 'increment': 1})

    classification_parser_panel_2.add_argument('--optimizer', type=str, required=True,
                                               metavar="Optimizer",
                                               help='Optimizer used for training model',
                                               widget='Dropdown', choices=get_classifier_optimizers())

    classification_parser_panel_2.add_argument('--learning_rate', type=float, default=0.0005,
                                               metavar="Learning Rate",
                                               help='The value used to incrementally adjust model weights')

    classification_parser_panel_2.add_argument('--metrics', type=str, required=True, nargs="+",
                                               metavar="Metrics",
                                               help='Metrics used to evaluate the model',
                                               widget='Listbox', choices=get_classifier_metrics())

    classification_parser_panel_2.add_argument('--loss_function', type=str, required=True,
                                               metavar="Loss Function",
                                               help='Loss function for training model',
                                               widget='Dropdown', choices=get_classifier_losses())

    classification_parser_panel_2.add_argument('--even_dist',
                                               metavar="Even Distribution",
                                               help='Downsample majority classes to be +/- N% of minority',
                                               action="store_true",
                                               widget='BlockCheckbox')

    classification_parser_panel_2.add_argument('--about', type=float, default=0.25,
                                               metavar="About",
                                               help='Downsample majority classes by "about" +/- N% of minority class')

    classification_parser_panel_2.add_argument('--weighted_loss', default=True,
                                               metavar="Weighted Loss Function",
                                               help='Recommended; useful if class categories are imbalanced',
                                               action="store_true",
                                               widget='BlockCheckbox')

    classification_parser_panel_2.add_argument('--augment_data',
                                               metavar="Augment Data",
                                               help='Recommended; useful if class categories are imbalanced',
                                               action="store_true",
                                               widget='BlockCheckbox')

    classification_parser_panel_2.add_argument('--dropout_rate', type=int,
                                               metavar="Dropout Rate",
                                               help="Random dropout rate as a form of data augmentation",
                                               widget='Slider', gooey_options={'min': 0, 'max': 100, 'increment': 1})

    classification_parser_panel_2.add_argument('--num_epochs', type=int, default=25,
                                               metavar="Number of Epochs",
                                               help='The number of iterations the model trains for')

    classification_parser_panel_2.add_argument('--batch_size', type=int, default=128,
                                               metavar="Batch Size",
                                               help='The number of samples per batch; GPU dependent')

    classification_parser_panel_2.add_argument('--tensorboard', action='store_true',
                                               metavar="Tensorboard",
                                               help='Open Tensorboard for viewing model training in real-time',
                                               widget='BlockCheckbox')

    # ------------------------------------------------------------------------------------------------------------------
    # Classification Inference
    # ------------------------------------------------------------------------------------------------------------------
    classification_inf_parser = subs.add_parser('ClassificationInference')

    # Panel 1
    classification_inf_parser_panel_1 = classification_inf_parser.add_argument_group('Classification Inference',
                                                                                     'Use a locally trained image '
                                                                                     'classification model'
                                                                                     'to make predictions on sampled '
                                                                                     'points.',
                                                                                     gooey_options={
                                                                                         'show_border': True})

    classification_inf_parser_panel_1.add_argument('--images', required=True,
                                                   metavar="Image Directory",
                                                   help='Directory containing images to make predictions on',
                                                   widget="DirChooser")

    classification_inf_parser_panel_1.add_argument('--points', required=True,
                                                   metavar="Points File",
                                                   help='A file containing sampled points',
                                                   widget="FileChooser")

    classification_inf_parser_panel_1.add_argument('--model', required=True,
                                                   metavar="Model Path",
                                                   help='The path to locally trained model (.pth)',
                                                   widget="FileChooser")

    classification_inf_parser_panel_1.add_argument('--class_map', required=True,
                                                   metavar="Class Map File",
                                                   help='The class mapping JSON file',
                                                   widget="FileChooser")

    classification_inf_parser_panel_1.add_argument("--patch_size", type=int, default=112,
                                                   metavar="Patch Extent",
                                                   help="The extent of each patch extracted")

    classification_inf_parser_panel_1.add_argument('--output_dir', required=True,
                                                   metavar='Output Directory',
                                                   default=DATA_DIR,
                                                   help='Root directory where output will be saved',
                                                   widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Viscore Inference
    # ------------------------------------------------------------------------------------------------------------------
    viscore_inf_parser = subs.add_parser('ViscoreInference')

    # Panel 1
    viscore_inf_parser_panel_1 = viscore_inf_parser.add_argument_group('Viscore Inference',
                                                                       'Use a locally trained image '
                                                                       'classification model to make '
                                                                       'predictions points from Viscore. '
                                                                       'Results are the mode of all views for a '
                                                                       'point.',
                                                                       gooey_options={
                                                                           'show_border': True})

    viscore_inf_parser_panel_1.add_argument('--images', required=True,
                                            metavar="Image Directory",
                                            help='Directory containing images to make predictions on',
                                            widget="DirChooser")

    viscore_inf_parser_panel_1.add_argument('--points', required=True,
                                            metavar="Points File",
                                            help='CoralNet-formatted points file for the plot',
                                            widget="FileChooser")

    viscore_inf_parser_panel_1.add_argument('--user_json', required=True,
                                            metavar="User JSON File",
                                            help='An empty User JSON file for the plot',
                                            widget="FileChooser")

    viscore_inf_parser_panel_1.add_argument('--qclasses_json', required=True,
                                            metavar="QClasses JSON File",
                                            help='A QClasses JSON file for the plot',
                                            widget="FileChooser")

    viscore_inf_parser_panel_1.add_argument('--output_dir', required=True,
                                            metavar='Output Directory',
                                            default=DATA_DIR,
                                            help='Root directory where output will be saved',
                                            widget="DirChooser")

    # Panel 2
    viscore_inf_parser_panel_2 = viscore_inf_parser.add_argument_group('Inference parameters',
                                                                       gooey_options={
                                                                           'show_border': True})

    viscore_inf_parser_panel_2.add_argument('--model', required=True,
                                            metavar="Model Path",
                                            help='The path to locally trained model (.pth)',
                                            widget="FileChooser")

    viscore_inf_parser_panel_2.add_argument('--class_map', required=True,
                                            metavar="Class Map File",
                                            help='The class mapping JSON file',
                                            widget="FileChooser")

    viscore_inf_parser_panel_2.add_argument("--conf", type=int, default=50,
                                            metavar="Confidence",
                                            help="Confidence threshold value (filter)",
                                            widget='Slider', gooey_options={'min': 0, 'max': 100, 'increment': 1})

    viscore_inf_parser_panel_2.add_argument("--patch_size", type=int, default=112,
                                            metavar="Patch Extent",
                                            help="The extent of each patch extracted")

    # ------------------------------------------------------------------------------------------------------------------
    # Semantic Segmentation w/ SAM
    # ------------------------------------------------------------------------------------------------------------------
    sam_parser = subs.add_parser('SAM')

    # Panel 1
    sam_parser_panel_1 = sam_parser.add_argument_group('Segment Anything Model (SAM)',
                                                       'Use the Segment Anything Model (SAM) to create '
                                                       'segmentation masks given labeled points.',
                                                       gooey_options={'show_border': True})

    sam_parser_panel_1.add_argument('--images', required=True,
                                    metavar="Image Directory",
                                    help='Directory containing images to make predictions on',
                                    widget="DirChooser")

    sam_parser_panel_1.add_argument('--annotations', required=True,
                                    metavar="Annotations File",
                                    help='A file containing points with labels',
                                    widget="FileChooser")

    sam_parser_panel_1.add_argument("--label_col", type=str, default='Label',
                                    metavar='Label Column',
                                    help="The column in annotations with labels to use "
                                         "('Label', 'Machine suggestion N, etc.)",
                                    widget='Dropdown',
                                    choices=['Label'] + [f'Machine suggestion {n + 1}' for n in range(5)])

    sam_parser_panel_1.add_argument("--confidence", type=int, default=75,
                                    metavar="Confidence Threshold",
                                    help="Confidence threshold value to filter (percentage)",
                                    widget='Slider', gooey_options={'min': 0, 'max': 100, 'increment': 1})

    sam_parser_panel_1.add_argument('--model_type', type=str, required=True,
                                    metavar="Model Version",
                                    help='Version of SAM model to use',
                                    widget='Dropdown', choices=['vit_b', 'vit_l', 'vit_h'])

    sam_parser_panel_1.add_argument("--points_per_side", type=int, default=64,
                                    metavar='Points Per Side',
                                    help="The number of points to sample from image (power of two)")

    sam_parser_panel_1.add_argument("--points_per_batch", type=int, default=64,
                                    metavar='Points Per Batch',
                                    help="The number of points per batch (power of two)")

    sam_parser_panel_1.add_argument('--output_dir', required=True,
                                    metavar='Output Directory',
                                    default=DATA_DIR,
                                    help='Root directory where output will be saved',
                                    widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------------------------------------------------------
    segmentation_parser = subs.add_parser('Segmentation')

    # Panel 1
    segmentation_parser_panel_1 = segmentation_parser.add_argument_group('Semantic Segmentation',
                                                                         'Use Pytorch to train a deep learning '
                                                                         'semantic segmentation algorithm; provide '
                                                                         'custom masks, or those made by SAM',
                                                                         gooey_options={'show_border': True})

    segmentation_parser_panel_1.add_argument('--masks', type=str, required=True,
                                             metavar="Masks File",
                                             help='The path to the masks csv file',
                                             widget='FileChooser')

    segmentation_parser_panel_1.add_argument('--color_map', type=str, required=True,
                                             metavar='Color Map File',
                                             help='Path to Color Map JSON file',
                                             widget='FileChooser')

    # Panel 2
    segmentation_parser_panel_2 = segmentation_parser.add_argument_group('Parameters',
                                                                         'Choose the parameters for training the model',
                                                                         gooey_options={'show_border': True})

    segmentation_parser_panel_2.add_argument('--encoder_name', type=str, default='mit_b0',
                                             metavar="Encoder",
                                             help='The convolutional encoder, pre-trained on ImageNet dataset',
                                             widget="Dropdown", choices=get_segmentation_encoders())

    segmentation_parser_panel_2.add_argument('--decoder_name', type=str, default='Unet',
                                             metavar='Decoder',
                                             help='The convolutional decoder',
                                             widget="Dropdown", choices=get_segmentation_decoders())

    segmentation_parser_panel_2.add_argument('--metrics', type=str, nargs='+', default=get_segmentation_metrics(),
                                             metavar='Metrics',
                                             help='The metric(s) to evaluate the model',
                                             widget='Listbox', choices=get_segmentation_metrics())

    segmentation_parser_panel_2.add_argument('--loss_function', type=str, default='JaccardLoss',
                                             metavar='Loss Function',
                                             help='The loss function to use to train the model',
                                             widget="Dropdown", choices=get_segmentation_losses())

    segmentation_parser_panel_2.add_argument('--freeze_encoder', type=int,
                                             metavar="Freeze Encoder",
                                             help='Freeze encoder by N%',
                                             widget='Slider', gooey_options={'min': 0, 'max': 100, 'increment': 1})

    segmentation_parser_panel_2.add_argument('--optimizer', type=str, default='Adam',
                                             metavar='Optimizer',
                                             help='The optimizer to use to train the model',
                                             widget="Dropdown", choices=get_segmentation_optimizers())

    segmentation_parser_panel_2.add_argument('--learning_rate', type=float, default=0.0001,
                                             metavar='Learning Rate',
                                             help='Starting learning rate')

    segmentation_parser_panel_2.add_argument('--augment_data', action='store_true',
                                             metavar='Augment Data',
                                             help='Apply affine augmentations to training data',
                                             widget='BlockCheckbox')

    segmentation_parser_panel_2.add_argument('--num_epochs', type=int, default=15,
                                             help='The number of iterations the model trains for')

    segmentation_parser_panel_2.add_argument('--batch_size', type=int, default=2,
                                             metavar='Batch Size',
                                             help='Number of samples per batch during training')

    segmentation_parser_panel_2.add_argument('--tensorboard', action='store_true',
                                             metavar='Tensorboard',
                                             help='Display training on Tensorboard',
                                             widget='BlockCheckbox')

    segmentation_parser_panel_2.add_argument('--output_dir', type=str, required=True,
                                             metavar='Output Directory',
                                             help='Directory to store results',
                                             widget='DirChooser')

    # ------------------------------------------------------------------------------------------------------------------
    # Segmentation Inference
    # ------------------------------------------------------------------------------------------------------------------
    segmentation_inf_parser = subs.add_parser('SegmentationInference')

    # Panel 1
    segmentation_inf_parser_panel_1 = segmentation_inf_parser.add_argument_group('Segmentation Inference',
                                                                                 'Use a trained semantic segmentation '
                                                                                 'model to perform inference on new '
                                                                                 'images',
                                                                                 gooey_options={'show_border': True})

    segmentation_inf_parser_panel_1.add_argument('--images', required=True,
                                                 metavar="Image Directory",
                                                 help='Directory containing images to make predictions on',
                                                 widget="DirChooser")

    segmentation_inf_parser_panel_1.add_argument('--model', required=True,
                                                 metavar="Model Path",
                                                 help='The path to locally trained model (.pth)',
                                                 widget="FileChooser")

    segmentation_inf_parser_panel_1.add_argument('--color_map', required=True,
                                                 metavar="Color Map File",
                                                 help='The Color Mapping JSON file',
                                                 widget="FileChooser")

    segmentation_inf_parser_panel_1.add_argument('--output_dir', required=True,
                                                 metavar='Output Directory',
                                                 default=DATA_DIR,
                                                 help='Root directory where output will be saved',
                                                 widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # SfM
    # ------------------------------------------------------------------------------------------------------------------
    sfm_parser = subs.add_parser('SfM')

    # Panel 1
    sfm_parser_panel_1 = sfm_parser.add_argument_group('Structure from Motion',
                                                       'Use Metashape (2.0.X) API to perform Structure from Motion on '
                                                       'images of a scene.',
                                                       gooey_options={'show_border': True})

    sfm_parser_panel_1.add_argument('--metashape_license', type=str,
                                    metavar="Metashape License (Pro)",
                                    default=os.getenv('METASHAPE_LICENSE'),
                                    help='The license for Professional version of Metashape',
                                    widget="PasswordField")

    sfm_parser_panel_1.add_argument('--remember_license', action="store_false",
                                    metavar="Remember License",
                                    help='Store License as an Environmental Variable',
                                    widget="BlockCheckbox")

    sfm_parser_panel_1.add_argument('--input_dir',
                                    metavar="Image Directory",
                                    help='Directory containing images of scene',
                                    widget="DirChooser")

    sfm_parser_panel_1.add_argument('--output_dir',
                                    metavar='Output Directory',
                                    default=DATA_DIR,
                                    help='Root directory where output will be saved',
                                    widget="DirChooser")

    sfm_parser_panel_1.add_argument('--quality', type=str, default="Medium",
                                    metavar="Quality",
                                    help='Quality of data products',
                                    widget="Dropdown", choices=['Lowest', 'Low', 'Medium', 'High', 'Highest'])

    sfm_parser_panel_1.add_argument('--target_percentage', type=int, default=75,
                                    metavar="Target Percentage",
                                    help='Percentage of points to target for each gradual selection method',
                                    widget='Slider', gooey_options={'min': 0, 'max': 99, 'increment': 1})

    # Panel 2
    sfm_parser_panel_2 = sfm_parser.add_argument_group('Existing Project',
                                                       'Provide an existing project directory to pick up where the '
                                                       'program left off instead of re-running from scratch.',
                                                       gooey_options={'show_border': True})

    sfm_parser_panel_2.add_argument('--project_file', type=str, required=False,
                                    metavar="Project File",
                                    help='Path to existing Metashape project file (.psx)',
                                    widget='FileChooser')

    # ------------------------------------------------------------------------------------------------------------------
    # Seg3D
    # ------------------------------------------------------------------------------------------------------------------
    seg3d_parser = subs.add_parser('Segmentation3D')

    # Panel 1
    seg3d_parser_panel_1 = seg3d_parser.add_argument_group('Semantic Segmentation 3D',
                                                           'Use segmentation masks with Metashape (2.0.X) to create '
                                                           'classified versions of dense point clouds and meshes',
                                                           gooey_options={'show_border': True})

    seg3d_parser_panel_1.add_argument('--metashape_license', type=str,
                                      metavar="Metashape License (Pro)",
                                      default=os.getenv('METASHAPE_LICENSE'),
                                      help='The license for Professional version of Metashape',
                                      widget="PasswordField")

    seg3d_parser_panel_1.add_argument('--remember_license', action="store_false",
                                      metavar="Remember License",
                                      help='Store License as an Environmental Variable',
                                      widget="BlockCheckbox")

    seg3d_parser_panel_1.add_argument('--project_file', required=True,
                                      metavar="Metashape Project File",
                                      help='Path to existing Metashape project file (.psx)',
                                      widget="FileChooser")

    seg3d_parser_panel_1.add_argument('--masks_file', type=str, required=True,
                                      metavar='Masks File',
                                      help='Masks file containing paths for color masks of images',
                                      widget='FileChooser')

    seg3d_parser_panel_1.add_argument('--color_map', type=str, required=True,
                                      metavar="Color Map File",
                                      help='Path to Color Map JSON file',
                                      widget="FileChooser")

    seg3d_parser_panel_1.add_argument('--mask_column', type=str, default='Color Path',
                                      metavar="Mask Column",
                                      help='Column name of masks to use for classification')

    seg3d_parser_panel_1.add_argument('--include_binary_masks', action='store_true',
                                      metavar='Include Binary Masks',
                                      help='Include the use of binary masks as well as colored masks')

    seg3d_parser_panel_1.add_argument('--chunk_index', type=int, default=0,
                                      metavar="Chunk Index",
                                      help='Chunk index to classify; 0-based indexing')

    # ------------------------------------------------------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------------------------------------------------------
    args = parser.parse_args()

    # If user asks, save credentials as env variables for future
    # Gooey is weird, using double negative, not being nefarious
    if 'username' in vars(args) and 'password' in vars(args):
        if not args.remember_username:
            os.environ['CORALNET_USERNAME'] = args.username
        if not args.remember_password:
            os.environ['CORALNET_PASSWORD'] = args.password

    if 'metashape_license' in vars(args):
        if not args.remember_license:
            os.environ['METASHAPE_LICENSE'] = args.metashape_license

    try:
        # Execute command
        if args.command == 'API':
            api(args)

        if args.command == 'Download':
            download(args)

        if args.command == 'Labelset':
            labelset(args)

        if args.command == 'Upload':
            upload(args)

        if args.command == 'Annotate':
            patch_extractor()

        if args.command == 'Patches':
            patches(args)

        if args.command == 'Visualize':
            visualize(args)

        if args.command == 'Projector':
            projector(args)

        if args.command == 'Spotlight':
            spotlight(args)

        if args.command == 'Points':
            points(args)

        if args.command == 'Classification':
            classification(args)

        if args.command == 'ClassificationPretrain':
            classification_pretrain(args)

        if args.command == 'ClassificationInference':
            classification_inference(args)

        if args.command == 'ViscoreInference':
            viscore_inference(args)

        if args.command == 'SAM':
            sam(args)

        if args.command == 'Segmentation':
            segmentation(args)

        if args.command == 'SegmentationInference':
            segmentation_inference(args)

        if args.command == 'SfM':
            sfm(args)

        if args.command == 'Segmentation3D':
            segmentation3d(args)

        print('Done.')

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
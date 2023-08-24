import os
import signal

from Tools.API import *
from Tools.Upload import *
from Tools.Download import *
from Tools.Labelset import *
from Tools.Viscore import *
from Tools.Classifier import *
from Tools.Points import *
from Tools.Patches import *
from Tools.Annotate import *
from Tools.Visualize import *
from Tools.Inference import *
from Tools.SAM import *
from Tools.SfM import *
from Tools.Seg3D import *

from gooey import Gooey, GooeyParser


@Gooey(dump_build_config=True,
       program_name="CoralNet Toolbox",
       default_size=(900, 600),  # width, height
       console=True,
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
                                                       'fields: Name, Row, Column.')

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
                                    help='The ID of the Source containing the model to use, if different.')

    api_parser_panel_1.add_argument('--csv_path', required=True, type=str,
                                    metavar="Points File",
                                    help='A path to a csv file containing the following: Name, Row, Column',
                                    widget="FileChooser")

    api_parser_panel_1.add_argument('--output_dir', required=True,
                                    metavar='Output Directory',
                                    default=os.path.abspath("..\\Data"),
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
                                                                 'between each.')

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
                                         default=os.path.abspath("..\\Data"),
                                         help='A root directory where all downloads will be saved to.',
                                         widget="DirChooser")

    download_parser_panel_1.add_argument('--headless', action="store_false",
                                         default=True,
                                         metavar="Run in Background",
                                         help='Run browser in headless mode',
                                         widget='BlockCheckbox')

    # Panel 2 - Download CoralNet Source ID and Labelset Lists
    download_parser_panel_2 = download_parser.add_argument_group('Download CoralNet Dataframes',
                                                                 'In addition to downloading Source data, dataframes '
                                                                 'containing information on all public Sources and '
                                                                 'Labelsets can also be downloaded.')

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
                                                                 'globally.')

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
                                         widget='FilterableDropdown', choices=FUNC_GROUPS_LIST)

    labelset_parser_panel_1.add_argument('--desc', type=str, required=True,
                                         metavar="Description",
                                         help='The Description of the labelset to create')

    labelset_parser_panel_1.add_argument('--image_path', type=str, required=True,
                                         metavar="Thumbnail Path",
                                         help='The path to the image of the labelset to create',
                                         widget="FileChooser")

    labelset_parser_panel_1.add_argument('--headless', action="store_false",
                                         default=True,
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
                                                             'annotations.')

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
                                       help='A prefix to add to each image basename')

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

    upload_parser_panel_1.add_argument('--headless', action="store_false",
                                       default=True,
                                       metavar="Run in Background",
                                       help='Run browser in headless mode',
                                       widget='BlockCheckbox')

    # ------------------------------------------------------------------------------------------------------------------
    # Viscore
    # ------------------------------------------------------------------------------------------------------------------
    viscore_parser = subs.add_parser('Viscore')

    # Panel 1 - Upload CoralNet Data given a source
    viscore_parser_panel_1 = viscore_parser.add_argument_group('Upload data from Viscore to CoralNet',
                                                               'Provide the output from the Viscore which can filtered '
                                                               'and converted into a CoralNet format. Also provide the '
                                                               'mapping, labelset files, and image directory of data '
                                                               'to upload to CoralNet.')

    viscore_parser_panel_1.add_argument('--username', type=str,
                                        metavar="Username",
                                        default=os.getenv('CORALNET_USERNAME'),
                                        help='Username for CoralNet account')

    viscore_parser_panel_1.add_argument('--password', type=str,
                                        metavar="Password",
                                        default=os.getenv('CORALNET_PASSWORD'),
                                        help='Password for CoralNet account',
                                        widget="PasswordField")

    viscore_parser_panel_1.add_argument('--remember_username', action="store_false",
                                        metavar="Remember Username",
                                        help='Store Username as an Environmental Variable',
                                        widget="BlockCheckbox")

    viscore_parser_panel_1.add_argument('--remember_password', action="store_false",
                                        metavar="Remember Password",
                                        help='Store Password as an Environmental Variable',
                                        widget="BlockCheckbox")

    viscore_parser_panel_1.add_argument('--source_id', type=str, required=True,
                                        metavar="Source ID",
                                        help='The ID of the source to upload data to.')

    viscore_parser_panel_1.add_argument('--images', required=False, default="",
                                        metavar='Image Directory',
                                        help='Directory containing images to upload.',
                                        widget="DirChooser")

    viscore_parser_panel_1.add_argument('--prefix', required=False, default="",
                                        metavar='Image Name Prefix',
                                        help='A prefix to add to each image basename')

    viscore_parser_panel_1.add_argument('--labelset', required=False, type=str,
                                        metavar="Labelset File",
                                        help='A path to the source labelset csv file. '
                                             'The file should contain the following: Label ID, Short Code',
                                        default=os.path.abspath("../Data/Mission_Iconic_Reefs"
                                                                "/MIR_CoralNet_Labelset.csv"),
                                        widget="FileChooser")

    viscore_parser_panel_1.add_argument('--headless', action="store_false",
                                        default=True,
                                        metavar="Run in Background",
                                        help='Run browser in headless mode',
                                        widget='BlockCheckbox')

    # Panel 2 - Viscore specific files
    viscore_parser_panel_2 = viscore_parser.add_argument_group('Viscore',
                                                               'Provide the labels csv file exported from Viscore, '
                                                               'and the mapping csv file that maps the Viscore '
                                                               'project labels to the CoralNet source labels. Modify '
                                                               'which projects dots are retained through filtering.')

    viscore_parser_panel_2.add_argument('--viscore_labels', required=False, type=str,
                                        metavar="Viscore Annotation File",
                                        help='A path to the annotation csv file output from Viscore',
                                        widget="FileChooser")

    viscore_parser_panel_2.add_argument('--mapping_path', required=False, type=str,
                                        metavar="Mapping File",
                                        help='A path to the mapping csv file. The file should contain the mapping '
                                             'between the labels in your Viscore project, and labels used in CoralNet',
                                        default=os.path.abspath(
                                            '../Data/Mission_Iconic_Reefs/MIR_VPI_CoralNet_Mapping.csv'),
                                        widget="FileChooser")

    viscore_parser_panel_2.add_argument('--rand_sub_ceil', type=float, required=False, default=1.0,
                                        metavar="Random Sample",
                                        help='Value used to randomly sample the number of reprojected dots [0 - 1]')

    viscore_parser_panel_2.add_argument('--reprojection_error', type=float, required=False, default=0.01,
                                        metavar="Reprojection Error Threshold",
                                        help='Value used to filter dots based on their reprojection error; '
                                             'dots with error values larger than the provided threshold are filtered')

    viscore_parser_panel_2.add_argument('--view_index', type=int, required=False, default=9001,
                                        metavar="VPI View Index",
                                        help='Value used to filter views based on their VPI View Index; '
                                             'indices of VPI image views after provided threshold are filtered')

    viscore_parser_panel_2.add_argument('--view_count', type=int, required=False, default=9001,
                                        metavar="VPI View Count",
                                        help='Value used to filter views based on the total number of VPI image views; '
                                             'indices of VPI views of dot after provided threshold are filtered')

    viscore_parser_panel_2.add_argument('--output_dir', required=False,
                                        metavar='Output Directory',
                                        default=None,
                                        help='A root directory where the updated Viscore labels will be saved to; '
                                             'defaults to the same directory as Viscore Annotation File',
                                        widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------------------------------------------------------
    visualize_parser = subs.add_parser('Visualize')

    # Panel 1
    visualize_parser_panel_1 = visualize_parser.add_argument_group('Visualize',
                                                                   'View annotations superimposed on each image; toggle'
                                                                   'annotations to be seen as points or squares.')

    visualize_parser_panel_1.add_argument('--image_dir', required=True,
                                          metavar='Image Directory',
                                          help='A directory where all images are located.',
                                          widget="DirChooser")

    visualize_parser_panel_1.add_argument('--annotations', required=False, type=str,
                                          metavar="Annotations",
                                          help='The path to the annotations dataframe',
                                          widget="FileChooser")

    visualize_parser_panel_1.add_argument('--output_dir',
                                          metavar='Output Directory',
                                          default=os.path.abspath("..\\Data"),
                                          help='Root directory where output will be saved',
                                          widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Annotate
    # ------------------------------------------------------------------------------------------------------------------
    annotate_parser = subs.add_parser('Annotate')

    # Panel 1
    annotate_parser_panel_1 = annotate_parser.add_argument_group('Annotate',
                                                                 'Extract patches manually, which can be used as '
                                                                 'annotations in CoralNet, or used to train a model '
                                                                 'locally.')

    annotate_parser_panel_1.add_argument('--patch_extractor_path', required=True, type=str,
                                         metavar="Patch Extractor Path",
                                         help='The path to the CNNDataExtractor.exe',
                                         default=os.path.abspath('./Tools/Patch_Extractor/CNNDataExtractor.exe'),
                                         widget="FileChooser")

    annotate_parser_panel_1.add_argument('--image_dir', required=True,
                                         metavar='Image Directory',
                                         help='A directory where all images are located.',
                                         widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------------------------------------------------------
    points_parser = subs.add_parser('Points')

    # Panel 1
    points_parser_panel_1 = points_parser.add_argument_group('Points',
                                                             'Sample points for each image; points can be used '
                                                             'with the API or Inference tools. Methods include '
                                                             'uniform, random, and stratified-random. Samples are '
                                                             'saved in the Output Directory as a dataframe.')

    points_parser_panel_1.add_argument('--images', required=True, type=str,
                                       metavar="Image Directory",
                                       help='Directory of images to create sampled points for',
                                       widget="DirChooser")

    points_parser_panel_1.add_argument('--output_dir', required=True,
                                       metavar='Output Directory',
                                       default=None,
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
    # Patches
    # ------------------------------------------------------------------------------------------------------------------
    patches_parser = subs.add_parser('Patches')

    # Panel 1
    patches_parser_panel_1 = patches_parser.add_argument_group('Crop Patches',
                                                               'Use the following to convert CoralNet formatted '
                                                               'annotation files into patches for training.')

    patches_parser_panel_1.add_argument('--image_dir', required=False,
                                        metavar='Image Directory',
                                        help='Directory containing images associated with annotation file',
                                        widget="DirChooser")

    patches_parser_panel_1.add_argument('--annotation_file', required=False, type=str,
                                        metavar="Annotation File",
                                        help='CoralNet annotation file, or one created using the Annotation tool',
                                        widget="FileChooser")

    patches_parser_panel_1.add_argument('--output_dir', required=True,
                                        metavar='Output Directory',
                                        default=os.path.abspath("..\\Data"),
                                        help='Root directory where output will be saved',
                                        widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------------------------------------------------
    classifier_parser = subs.add_parser('Classifier')

    # Panel 1
    classifier_parser_panel_1 = classifier_parser.add_argument_group('Patch-Based Classifier',
                                                                     'Use the following to train your own patch-based '
                                                                     'image classifier.')

    classifier_parser_panel_1.add_argument('--patches', required=True, nargs="+",
                                           metavar="Patch Data",
                                           help='Patches dataframe file(s)',
                                           widget="MultiFileChooser")

    classifier_parser_panel_1.add_argument('--output_dir', required=True,
                                           metavar='Output Directory',
                                           default=os.path.abspath("..\\Data"),
                                           help='Root directory where output will be saved',
                                           widget="DirChooser")

    # Panel 2
    classifier_parser_panel_2 = classifier_parser.add_argument_group('Parameters',
                                                                     'Choose the parameters for training the model')

    classifier_parser_panel_2.add_argument('--model_name', type=str, required=True,
                                           metavar="Pretrained Encoder",
                                           help='Encoder, pre-trained on ImageNet dataset',
                                           widget='FilterableDropdown', choices=get_available_models())

    classifier_parser_panel_2.add_argument('--loss_function', type=str, required=True,
                                           metavar="Loss Function",
                                           help='Loss function for training model',
                                           widget='FilterableDropdown', choices=get_available_losses())

    classifier_parser_panel_2.add_argument('--weighted_loss', default=True,
                                           metavar="Weighted Loss Function",
                                           help='Recommended; useful if class categories are imbalanced',
                                           action="store_true",
                                           widget='BlockCheckbox')

    classifier_parser_panel_2.add_argument('--augment_data', default=True,
                                           metavar="Augment Data",
                                           help='Recommended; useful if class categories are imbalanced',
                                           action="store_true",
                                           widget='BlockCheckbox')

    classifier_parser_panel_2.add_argument('--dropout_rate', type=float, default=0.5,
                                           metavar="Drop Out",
                                           help='Recommended; useful if class categories are imbalanced')

    classifier_parser_panel_2.add_argument('--num_epochs', type=int, default=25,
                                           metavar="Number of Epochs",
                                           help='The number of iterations the model is given the training dataset')

    classifier_parser_panel_2.add_argument('--batch_size', type=int, default=128,
                                           metavar="Batch Size",
                                           help='The number of samples per batch; GPU dependent')

    classifier_parser_panel_2.add_argument('--learning_rate', type=float, default=0.0005,
                                           metavar="Learning Rate",
                                           help='The floating point value used to incrementally adjust model weights')

    classifier_parser_panel_2.add_argument('--tensorboard', default=True,
                                           metavar="Tensorboard",
                                           help='Open Tensorboard for viewing model training in real-time',
                                           action="store_true",
                                           widget='BlockCheckbox')

    # ------------------------------------------------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------------------------------------------------
    inference_parser = subs.add_parser('Inference')

    # Panel 1
    inference_parser_panel_1 = inference_parser.add_argument_group('Inference',
                                                                   'Use a locally trained model to make predictions on'
                                                                   'sampled points.')

    inference_parser_panel_1.add_argument('--images', required=True,
                                          metavar="Image Directory",
                                          help='Directory containing images to make predictions on',
                                          widget="DirChooser")

    inference_parser_panel_1.add_argument('--points', required=True,
                                          metavar="Points File",
                                          help='A file containing sampled points',
                                          widget="FileChooser")

    inference_parser_panel_1.add_argument('--model', required=True,
                                          metavar="Model Path",
                                          help='The path to locally trained model (.h5)',
                                          widget="FileChooser")

    inference_parser_panel_1.add_argument('--class_map', required=True,
                                          metavar="Class Map File",
                                          help='The class mapping JSON file',
                                          widget="FileChooser")

    inference_parser_panel_1.add_argument('--output_dir', required=True,
                                          metavar='Output Directory',
                                          default=os.path.abspath("..\\Data"),
                                          help='Root directory where output will be saved',
                                          widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # MSS w/ SAM
    # ------------------------------------------------------------------------------------------------------------------
    sam_parser = subs.add_parser('SAM')

    # Panel 1
    sam_parser_panel_1 = sam_parser.add_argument_group('SAM',
                                                       'Use the Segment Anything Model (SAM) to create '
                                                       'segmentation masks given predicted points.')

    sam_parser_panel_1.add_argument('--images', required=True,
                                    metavar="Image Directory",
                                    help='Directory containing images to make predictions on',
                                    widget="DirChooser")

    sam_parser_panel_1.add_argument('--annotations', required=True,
                                    metavar="Annotations File",
                                    help='A file containing points with labels',
                                    widget="FileChooser")

    sam_parser_panel_1.add_argument('--class_map', required=True,
                                    metavar="Class Map File",
                                    help='The class mapping JSON file',
                                    widget="FileChooser")

    sam_parser_panel_1.add_argument('--model_type', type=str, required=True,
                                    metavar="Model Version",
                                    help='Version of SAM model to use',
                                    widget='FilterableDropdown', choices=['vit_b', 'vit_l', 'vit_h'])

    sam_parser_panel_1.add_argument("--patch_size", type=int, default=360,
                                    metavar="Patch Size",
                                    help="The approximate size of each superpixel formed by SAM")

    sam_parser_panel_1.add_argument("--batch_size", type=int, default=64,
                                    metavar="Batch Size",
                                    help="The number of samples passed to SAM in a batch (GPU dependent)")

    sam_parser_panel_1.add_argument('--plot', default=False,
                                    metavar="Plot Masks",
                                    help='Saves colorized figures of masks in subdirectory',
                                    action="store_true",
                                    widget='BlockCheckbox')

    sam_parser_panel_1.add_argument('--plot_progress', default=False,
                                    metavar="Plot Mask Progress",
                                    help='Saves colorized figures of masks during creation in subdirectory',
                                    action="store_true",
                                    widget='BlockCheckbox')

    sam_parser_panel_1.add_argument('--output_dir', required=True,
                                    metavar='Output Directory',
                                    default=os.path.abspath("..\\Data"),
                                    help='Root directory where output will be saved',
                                    widget="DirChooser")

    # ------------------------------------------------------------------------------------------------------------------
    # SfM
    # ------------------------------------------------------------------------------------------------------------------
    sfm_parser = subs.add_parser('SfM')

    # Panel 1
    sfm_parser_panel_1 = sfm_parser.add_argument_group('SfM',
                                                       'Use Metashape (2.0.X) API to perform Structure from Motion on'
                                                       'images of a scene.')

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
                                    default=os.path.abspath("..\\Data"),
                                    help='Root directory where output will be saved',
                                    widget="DirChooser")

    # Panel 2
    sfm_parser_panel_2 = sfm_parser.add_argument_group('Existing Project',
                                                       'Provide an existing project directory to pick up where the '
                                                       'program left off instead of re-running from scratch.')

    sfm_parser_panel_2.add_argument('--project_dir', type=str, required=False,
                                    metavar="Project Directory",
                                    help='Existing project directory',
                                    widget='DirChooser')

    # ------------------------------------------------------------------------------------------------------------------
    # Seg3D
    # ------------------------------------------------------------------------------------------------------------------
    seg3d_parser = subs.add_parser('Seg3D')

    # Panel 1
    seg3d_parser_panel_1 = seg3d_parser.add_argument_group('Seg3D',
                                                           'Use masks made by SAM with Metashape (2.0.X) API to create '
                                                           'classified versions of dense point clouds and meshes')

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

    seg3d_parser_panel_1.add_argument('--masks_dir', type=str, required=True,
                                      metavar="Masks Directory",
                                      help='Directory containing colored masks for images',
                                      widget='DirChooser')

    seg3d_parser_panel_1.add_argument('--color_map', type=str, required=True,
                                      metavar="Color Map File",
                                      help='Path to Color Map JSON file',
                                      widget="FileChooser")

    seg3d_parser_panel_1.add_argument('--chunk_index', type=int, default=0,
                                      metavar="Chunk Index",
                                      help='Chunk index to classify; 0-based indexing')

    seg3d_parser_panel_1.add_argument('--classify_mesh', action="store_true",
                                      metavar="Classify Mesh",
                                      help='Classify mesh using dense point cloud',
                                      widget="BlockCheckbox")

    seg3d_parser_panel_1.add_argument('--classify_ortho', action="store_true",
                                      metavar="Classify Orthomosaic",
                                      help='Classify orthomosaic using mesh',
                                      widget="BlockCheckbox")

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

    # Execute command
    if args.command == 'API':
        api(args)

    if args.command == 'Download':
        download(args)

    if args.command == 'Labelset':
        labelset(args)

    if args.command == 'Upload':
        upload(args)

    if args.command == 'Viscore':
        args.annotations = viscore(args)
        upload(args)

    if args.command == 'Visualize':
        visualize(args)

    if args.command == 'Annotate':
        annotate(args)

    if args.command == 'Points':
        points(args)

    if args.command == 'Patches':
        patches(args)

    if args.command == 'Classifier':
        train_classifier(args)

    if args.command == 'Inference':
        inference(args)

    if args.command == 'SAM':
        mss_sam(args)

    if args.command == 'SfM':
        sfm(args)

    if args.command == 'Seg3D':
        seg3d(args)

    print('Done.')


if __name__ == '__main__':
    main()

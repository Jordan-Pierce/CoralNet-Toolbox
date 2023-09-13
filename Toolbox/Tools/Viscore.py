import os
import sys
import argparse
import traceback
from tqdm import tqdm

import re
import pandas as pd

from Common import MIR_MAPPING

from API import api
from Upload import upload


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def make_labelset(args):
    """
    Creates a temporary labelset for CoralNet from the mapping file,
    just so we can reduce the amount of input parameters needed.
    """
    print("\n###############################################", flush=True)
    print("Making Labelset", flush=True)
    print("###############################################\n", flush=True)

    # Get the mapping file
    if os.path.exists(args.mapping_path):
        mapping = pd.read_csv(args.mapping_path, index_col=None, sep=",")
    else:
        print(f"ERROR: Mapping file provided doesn't exist; check input provided", flush=True)
        sys.exit(1)

    # Make the labelset file using the columns 'Short Code' and 'ID'
    labelset = mapping[['Short Code', 'Label ID']]
    # Drop duplicates
    labelset = labelset.drop_duplicates()

    # Labelset path
    labelset_path = f"{args.output_dir}\\labelset.csv"
    # Save the labelset file in the provided output directory
    labelset.to_csv(labelset_path)
    # Set the labelset file in args
    args.labelset = labelset_path

    if os.path.exists(labelset_path):
        print(f"NOTE: Labelset created successfully", flush=True)
    else:
        print("ERROR: Labelset could not be created", flush=True)
        sys.exit(1)

    return args


def convert_labels(args):
    """
    Converts the dots and cams JSON files to csv files for Tools. The csv
    files are saved in the output directory.
    """

    print("\n###############################################", flush=True)
    print("Viscore to CoralNet", flush=True)
    print("###############################################\n", flush=True)

    # Get the arguments
    viscore_labels = args.viscore_labels
    mapping_path = args.mapping_path

    # Check that the paths exist
    assert os.path.exists(viscore_labels), 'ERROR: labels path does not exist'
    assert os.path.exists(mapping_path), 'ERROR: Labelsets path does not exist'

    if args.output_dir is None:
        args.output_dir = os.path.dirname(viscore_labels) + "\\"
    else:
        args.output_dir = args.output_dir + "\\"

    # Make the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Make the output path
    basename = f"{os.path.basename(viscore_labels).split('.')[0]}_"
    basename += f"rand_{str(args.rand_sub_ceil).replace('.', '_')}_"
    basename += f"error_{str(args.reprojection_error).replace('.', '_')}_"
    basename += f"vindex_{str(args.view_index)}_"
    basename += f"vcount_{str(args.view_count)}"
    output_file = f"{args.output_dir}{basename}.csv"

    # If the updated file already exists, return early
    if os.path.exists(output_file):
        print(f"NOTE: {basename} already exists", flush=True)
        args.converted_labels = output_file
        return args

    # Pattern to check against
    pattern = r'^[A-Za-z]+_[A-Za-z\d-]+_\d{4}-\d{2}-\d{2}\.csv$'

    # Check if labels file matches expected format
    if not re.match(pattern, os.path.basename(viscore_labels)):
        print(f"ERROR: Label path does not match expected format; check input provided.", flush=True)
        sys.exit(1)

    try:
        print(f'NOTE: Converting {os.path.basename(viscore_labels)} to CoralNet format', flush=True)
        # Open the viscore labels file
        labels = pd.read_csv(viscore_labels, index_col=None)
        # Open the labelset file
        mapping = pd.read_csv(mapping_path, index_col=None, sep=",")
    except Exception as e:
        raise Exception(f'ERROR: Issue opening provided paths')

    # Apply filtering
    if args.reprojection_error:
        labels = labels[labels['ReprojectionError'] <= args.reprojection_error]
    if args.view_index:
        labels = labels[labels['ViewIndex'] <= args.view_index]
    if args.view_count:
        labels = labels[labels['ViewCount'] <= args.view_count]
    if args.rand_sub_ceil:
        labels = labels[labels['RandSubCeil'] <= args.rand_sub_ceil]

    if len(labels) == 0:
        raise Exception(f"ERROR: All labels were filtered; nothing to convert.")

    # To store the updated labels with Tools version of labelsets
    images = []
    annotations = []
    skipped = []

    for i, r in tqdm(labels.iterrows()):

        # if label is NaN, skip
        if pd.isna(r['Label']):
            skipped.append(r['Label'])
            continue

        try:
            # Get the VPI label
            l = r['Label']
            # Find it within the Tools VPI labelset
            lbst = mapping[(mapping['VPI_label_V3'] == l) | (mapping['VPI_label_V4'] == l)]
            # Check that one was found
            if lbst.empty:
                raise Exception
        except Exception as e:
            skipped.append(str(r['Label']))
            continue

        # Viscore exports the path to the image, but
        # CoralNet expects the basename; prepending
        # the plot name to the image basename.
        path = r['Name']
        basename = os.path.basename(path)
        name = f"{args.prefix}-{basename}"

        # For some reason, CoralNet uses the Short Code as
        # the label from within a source; make note of that.
        row = int(r['Row'])
        column = int(r['Column'])
        label = str(lbst['Short Code'].item())

        # Add to the list; other fields are ignored by CoralNet.
        images.append(basename)
        annotations.append([args.prefix, basename, name, row, column, label])

    print(f"NOTE: Updated {len(annotations)} annotations belonging to {len(set(images))} images", flush=True)
    print(f"NOTE: Skipped {len(skipped)} annotations belonging to {set(skipped)}", flush=True)

    annotations = pd.DataFrame(annotations, columns=['Prefix', 'Image Name', 'Name', 'Row', 'Column', 'Label'])
    annotations.to_csv(output_file)

    # Check that file was saved
    if os.path.exists(output_file):
        print(f'NOTE: Successfully saved {basename}', flush=True)
        args.converted_labels = output_file
    else:
        print(f'ERROR: Failed to save {basename}', flush=True)
        sys.exit(1)

    return args


def viscore(args):
    """

    """
    # First convert the Viscore labels to CoralNet format
    args = convert_labels(args)

    # Then, either upload, or use the api
    if args.action == "Upload":
        # Make a temporary labelset from the mapping file
        # to upload with converted labels, just in case.
        args = make_labelset(args)
        # Then, upload the data to CoralNet
        args.annotations = args.converted_labels
        upload(args)
    elif args.action == "API":
        # Otherwise, we'll use the API to make predictions
        args.points = args.converted_labels
        api(args)
    else:
        print("ERROR: Invalid action; must be Upload or API", flush=True)
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description='Viscore to CoralNet')

    parser.add_argument('--username', type=str, metavar="Username",
                        default=os.getenv('CORALNET_USERNAME'),
                        help='Username for CoralNet account.')

    parser.add_argument('--password', type=str, metavar="Password",
                        default=os.getenv('CORALNET_PASSWORD'),
                        help='Password for CoralNet account.')

    parser.add_argument('--action', type=str, required=True, default='Upload',
                        metavar="Action",
                        help='Upload data, or use the API for inference.',
                        choices=['Upload', 'API'])

    parser.add_argument('--source_id', type=str, default='4346',
                        help='The ID of the CoralNet source.')

    parser.add_argument('--prefix', required=False, default="",
                        help='The name of the Viscore layer.')

    parser.add_argument('--images', required=False, default="",
                        help='Directory containing images to upload.')

    parser.add_argument('--viscore_labels', required=False, type=str,
                        help='A path to the original Annotation file exported from Viscore.')

    parser.add_argument('--mapping_path', required=False, type=str,
                        help='A path to the mapping csv file.',
                        default=MIR_MAPPING)

    parser.add_argument('--rand_sub_ceil', type=float, required=False, default=1.0,
                        help='Value used to randomly sample the number of reprojected dots [0 - 1].')

    parser.add_argument('--reprojection_error', type=float, required=False, default=0.01,
                        help='Value used to filter dots based on their reprojection error.')

    parser.add_argument('--view_index', type=int, required=False, default=9001,
                        help='Value used to filter views based on their VPI View Index.')

    parser.add_argument('--view_count', type=int, required=False, default=9001,
                        help='Value used to filter views based on the total number of VPI image views.')

    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run browser in headless mode')

    parser.add_argument('--output_dir', required=False,
                        default=None,
                        help='A root directory where the converted Annotation file will be saved; '
                             'defaults to the same directory as Viscore Annotation file.')

    args = parser.parse_args()

    try:
        viscore(args)
        print('Done.', flush=True)

    except Exception as e:
        print(f'ERROR: {e}', flush=True)
        print(traceback.format_exc(), flush=True)


if __name__ == '__main__':
    main()

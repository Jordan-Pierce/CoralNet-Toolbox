import os
import sys
import argparse
import traceback
from tqdm import tqdm

import re
import pandas as pd

from API import api

from Upload import upload

from Common import log
from Common import MIR_MAPPING


# TODO post-process labels from CoralNet back to Viscore
# TODO Include the layer name (prefix) as a column in the API output just in case


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def make_labelset(args):
    """
    Creates a temporary labelset for CoralNet from the mapping file,
    just so we can reduce the amount of input parameters needed.
    """
    log("\n###############################################")
    log("Making Labelset")
    log("###############################################\n")

    # Get the mapping file
    if os.path.exists(args.mapping_path):
        mapping = pd.read_csv(args.mapping_path, index_col=None, sep=",")
    else:
        log(f"ERROR: Mapping file provided doesn't exist; check input provided")
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
        log(f"NOTE: Labelset created successfully")
    else:
        log("ERROR: Labelset could not be created")
        sys.exit(1)

    return args


def convert_labels_to_coralnet(args):
    """
    Converts the dots and cams JSON files to csv files for Tools. The csv
    files are saved in the output directory.
    """

    log("\n###############################################")
    log("Viscore to CoralNet")
    log("###############################################\n")

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
        log(f"NOTE: {basename} already exists")
        args.converted_viscore_labels = output_file
        return args

    # Pattern to check against
    pattern = r'^[A-Za-z]+_[A-Za-z\d-]+_[\d]{4}[-_]\d{2}[-_]\d{2}\.csv$'

    # Check if labels file matches expected format
    if not re.match(pattern, os.path.basename(viscore_labels)):
        log(f"ERROR: {os.path.basename(viscore_labels)} does not match expected pattern")
        sys.exit(1)

    try:
        log(f'NOTE: Converting {os.path.basename(viscore_labels)} to CoralNet format')
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

        # Everything else in the Viscore dataframe
        annotator = r['Annotator']
        dot = r['Dot']
        x = r['X']
        y = r['Y']
        z = r['Z']
        reprojection_error = r['ReprojectionError']
        view_index = r['ViewIndex']
        view_count = r['ViewCount']
        rand_subceil = r['RandSubCeil']

        # Add to the list; other fields are ignored by CoralNet.
        images.append(basename)
        annotations.append([args.prefix, basename, name, row, column, label,
                            annotator, dot, x, y, z, reprojection_error, view_index, view_count, rand_subceil])

    log(f"NOTE: Updated {len(annotations)} annotations belonging to {len(set(images))} images")
    log(f"NOTE: Skipped {len(skipped)} annotations belonging to {set(skipped)}")

    columns = ['Prefix', 'Image Name', 'Name', 'Row', 'Column', 'Label',
               'Annotator', 'Dot', 'X', 'Y', 'Z', 'ReprojectionError', 'ViewIndex', 'ViewCount', 'RandSubCeil']

    annotations = pd.DataFrame(annotations, columns=columns)
    annotations.to_csv(output_file)

    # Check that file was saved
    if os.path.exists(output_file):
        log(f'NOTE: Successfully saved {basename}')
        args.converted_viscore_labels = output_file
    else:
        log(f'ERROR: Failed to save {basename}')
        sys.exit(1)

    return args


def convert_labels_to_viscore(args):
    """

    """
    log("\n###############################################")
    log("CoralNet to Viscore")
    log("###############################################\n")

    # Get the arguments
    coralnet_predictions = args.predictions
    viscore_labels = args.viscore_labels
    mapping_path = args.mapping_path

    # Check that the paths exist
    assert os.path.exists(coralnet_predictions), 'ERROR: labels path does not exist'
    assert os.path.exists(mapping_path), 'ERROR: Labelsets path does not exist'

    # Make the output path
    basename = f"{os.path.basename(viscore_labels).split('.')[0]}_predictions"
    output_file = f"{args.output_dir}{basename}.csv"

    try:
        log(f'NOTE: Converting {os.path.basename(args.predictions)} back to Viscore format')
        # Open the viscore labels file
        predictions = pd.read_csv(coralnet_predictions, index_col=0)
        # Open the labelset file
        mapping = pd.read_csv(mapping_path, index_col=None, sep=",")
    except Exception as e:
        raise Exception(f'ERROR: Issue opening provided paths')

    # Holds updated label names
    updated_predictions = []

    # Looping through each row
    for i, r in tqdm(predictions.iterrows()):
        # In each row, get the machine suggestions [1 - 5]
        for m in [f'Machine suggestion {i}' for i in range(1, 6)]:
            # Get the machine suggestion for the row
            l = r[m]
            # Find the corresponding Viscore label using short code
            lbst = mapping[mapping['Short Code'] == l]
            # Pass in the VPI V4 label associated with short code
            r[m] = str(lbst['VPI_label_V4'].item())

        updated_predictions.append(r)

    # Output the updated predictions csv
    log(f"NOTE: Updated {len(updated_predictions)} predictions")

    updated_predictions = pd.DataFrame(updated_predictions)
    updated_predictions.to_csv(output_file)

    # Check that file was saved
    if os.path.exists(output_file):
        log(f'NOTE: Successfully saved {basename}')
        args.converted_coralnet_labels = output_file
    else:
        log(f'ERROR: Failed to save {basename}')
        sys.exit(1)

    return args


def viscore(args):
    """

    """
    # First convert the Viscore labels to CoralNet format
    args = convert_labels_to_coralnet(args)

    try:

        # Then, either upload, or use the api
        if args.action == "Upload":
            # Make a temporary labelset from the mapping file
            # to upload with converted labels, just in case.
            args = make_labelset(args)
            args.annotations = args.converted_viscore_labels
            # Then, upload the data to CoralNet
            upload(args)
        elif args.action == "API":
            # Otherwise, we'll set the points to the converted labels
            args.points = args.converted_viscore_labels
            # Prepare for API by adding arguments
            args.source_id_1 = args.source_id
            args.source_id_2 = args.source_id
            args.image_name_contains = args.prefix
            # And upload those to CoralNet for the Model API
            args = api(args)
            # After getting predictions, convert CoralNet labels back to Viscore
            args = convert_labels_to_viscore(args)
        else:
            log("ERROR: Invalid action; must be Upload or API")
            sys.exit()

    except Exception as e:
        log(f"ERROR: Could not complete action '{args.actions}'.\n{e}")


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
        log("Done.\n")

    except Exception as e:
        log(f'ERROR: {e}')
        log(traceback.format_exc())


if __name__ == '__main__':
    main()

import os
import argparse
import pandas as pd
from tqdm import tqdm

from Toolbox.Tools.Upload import *

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def viscore(args):
    """
    Converts the dots and cams JSON files to csv files for Tools. The csv
    files are saved in the output directory.
    """

    print("\n###############################################")
    print("Viscore")
    print("###############################################\n")

    # Get the arguments
    viscore_labels = args.viscore_labels
    mapping_path = args.mapping_path

    if args.output_dir is None:
        output_dir = os.path.dirname(viscore_labels) + "\\"
    else:
        output_dir = args.output_dir + "\\"

    # Make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check that the paths exist
    assert os.path.exists(viscore_labels), 'ERROR: labels path does not exist'
    assert os.path.exists(mapping_path), 'ERROR: Labelsets path does not exist'
    print(f'NOTE: Converting {viscore_labels} to CoralNet format')

    try:
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
        image_name = os.path.basename(path)
        prefix = os.path.basename(os.path.dirname(path))
        name = f"{prefix}-{image_name}"

        # For some reason, CoralNet uses the Short Code as
        # the label from within a source; make note of that.
        row = int(r['Row'])
        column = int(r['Column'])
        label = str(lbst['Short Code'].item())

        # Add to the list; other fields are ignored by CoralNet.
        images.append(image_name)
        annotations.append([prefix, image_name, name, row, column, label])

    print(f"NOTE: Updated {len(annotations)} annotations belonging to {len(set(images))} images")
    print(f"NOTE: Skipped {len(skipped)} annotations belonging to {set(skipped)}")

    # Save the labels as a csv file
    basename = f"{os.path.basename(viscore_labels).split('.')[0]}_"
    basename += f"rand_{str(args.rand_sub_ceil).replace('.', '_')}_"
    basename += f"error_{str(args.reprojection_error).replace('.', '_')}_"
    basename += f"vindex_{str(args.view_index)}_"
    basename += f"vcount_{str(args.view_count)}"
    output_file = f"{output_dir}{basename}.csv"
    annotations = pd.DataFrame(annotations, columns=['Prefix', 'Image Name', 'Name', 'Row', 'Column', 'Label'])
    annotations.to_csv(output_file, index=False)

    # Check that file was saved
    if os.path.exists(output_file):
        print(f'NOTE: Successfully saved {output_file}')
    else:
        print(f'ERROR: Failed to save {output_file}')

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert Viscore labels to CoralNet labels.')

    parser.add_argument('--viscore_labels', type=str,
                        help='The path to the labels csv file output by Viscore VPI')

    parser.add_argument('--mapping_path', type=str,
                        default=os.path.abspath('../../Data/Mission_Iconic_Reefs/MIR_VPI_CoralNet_Mapping.csv'),
                        help='The path to the csv file that maps VPI labels to Tools labels')

    parser.add_argument('--rand_sub_ceil', type=float, required=False, default=1.0,
                        help='Value used to randomly sample the number of reprojected dots [0 - 1]')

    parser.add_argument('--reprojection_error', type=float, required=False, default=0.01,
                        help='Value used to filter dots based on their reprojection error; '
                             'dots with error values larger than the provided threshold are filtered')

    parser.add_argument('--view_index', type=int, required=False, default=9001,
                        help='Value used to filter views based on their VPI View Index; '
                             'indices of VPI image views after provided threshold are filtered')

    parser.add_argument('--view_count', type=int, required=False, default=9001,
                        help='Value used to filter views based on the total number of VPI image views; '
                             'indices of VPI views of dot after provided threshold are filtered')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save updated label csv file.')

    args = parser.parse_args()

    try:
        annotations_file = viscore(args)
        print('Done.')

    except Exception as e:
        print(f'ERROR: Could not convert data\n{e}')


if __name__ == '__main__':
    main()

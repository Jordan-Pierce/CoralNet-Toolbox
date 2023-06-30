import os
import argparse
from tqdm import tqdm

import pandas as pd

import sys
sys.path.append('../')


def convert_to_csv(labels_path, mapping_path, output_dir):
    """
    Converts the dots and cams JSON files to CSV files for CoralNet. The CSV
    files are saved in the output directory.
    """

    print(f'NOTE: Converting {labels_path} to CoralNet format')

    try:
        labels = pd.read_csv(labels_path)

        # Open the labelset file
        mapping = pd.read_csv(mapping_path)

    except Exception as e:
        raise Exception(f'ERROR: Issue opening provided paths')

    # To store the updated labels with CoralNet version of labelsets
    annotations = []

    for i, r in tqdm(labels.iterrows()):

        # if label is NaN, skip
        if pd.isna(r['Label']):
            print(f"NOTE: Skipping null data {r['Label']}")
            continue

        try:
            # Get the VPI label
            l = r['Label']
            # Find it within the CoralNet VPI labelset
            lbst = mapping[(mapping['VPI_label_V3'] == l) | (mapping['VPI_label_V4'] == l)]
            # Check that one was found
            if lbst.empty:
                raise Exception
        except Exception as e:
            print(f'ERROR: Could not locate {r["Label"]} in {mapping_path}; skipping.')
            continue

        # Get the values for the updated label csv file
        # For some reason, CoralNet uses the Short Code as
        # the label from within a source
        name = os.path.basename(r['Name'].item())
        row = int(r['Row'].item())
        column = int(r['Column'].item())
        label = lbst['Short Code'].item()
        # Add to the list
        annotations.append([name, row, column, label])

    # Save the labelsets as a csv file
    basename = os.path.basename(labels_path).split('.')[0] + '_updated.csv'
    output_file = f'{output_dir}{basename}'
    annotations = pd.DataFrame(annotations, columns=['Name', 'Row', 'Column', 'Label'])
    annotations.to_csv(output_file, index=False)

    # Check that file was saved
    if os.path.exists(output_file):
        print(f'NOTE: Successfully saved {output_file}')
    else:
        print(f'ERROR: Failed to save {output_file}')

    return annotations


def main():
    parser = argparse.ArgumentParser(
        description='Convert VISCORE labels to CoralNet labels.')

    parser.add_argument('--labels_path', type=str,
                        help='The path to the labels CSV file output by VISCORE VPI')

    parser.add_argument('--mapping_path', type=str,
                        default='./MIR_VPI_CoralNet_Mapping.csv',
                        help='The path to the CSV that maps VPI labels to CoralNet labels')

    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Directory to save .csv files.')

    args = parser.parse_args()

    # Get the arguments
    labels_path = args.labels_path
    mapping_path = args.mapping_path

    if args.output_dir is None:
        output_dir = os.path.dirname(labels_path) + "\\"

    # Make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check that the paths exist
    assert os.path.exists(labels_path), 'ERROR: labels path does not exist'
    assert os.path.exists(mapping_path), 'ERROR: Labelsets path does not exist'

    try:
        annotations = convert_to_csv(labels_path, mapping_path, output_dir)
        print('Done.')

    except Exception as e:
        print(f'ERROR: Could not convert data\n{e}')


if __name__ == '__main__':
    main()


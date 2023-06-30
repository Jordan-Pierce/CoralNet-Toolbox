import os
import argparse
from tqdm import tqdm

import pandas as pd

import sys
sys.path.append('../')


def convert_to_coralnet(labels_path, labelset_path, output_dir):
    """
    Converts the dots and cams JSON files to CSV files for CoralNet. The CSV
    files are saved in the output directory.
    """

    print(f'NOTE: Converting {labels_path} to CoralNet format')

    try:
        labels = pd.read_csv(labels_path)

        # Open the labelset file
        labelset = pd.read_csv(labelset_path)

    except Exception as e:
        raise Exception(f'ERROR: Issue opening provided paths')

    # To store the updated labels with CoralNet version of labelsets
    annotations = []

    for i, r in tqdm(labels.iterrows()):

        # if label is NaN, skip
        if pd.isna(r['Label']) or r['Label'] in ['Unkn_macro']:
            continue

        try:
            # Get the name of the image
            name = os.path.basename(r['Name'])
            # Get the row of the annotation
            row = int(r['Row'])
            # Get the column of the annotation
            column = int(r['Column'])
            # Get the VPI label
            label = r['Label']
            # Find it within the CoralNet VPI labelset
            lbst = labelset[(labelset['VPI_label_V3'] == label) | (labelset['VPI_label_V4'] == label)]

            # Check that one was found
            if lbst.empty:
                raise Exception
            else:
                short_code = lbst['Short Code'].item()

        except Exception as e:
            print(f'ERROR: Could not locate {r["Label"]} in labelset {labelset_path}; skipping.')
            continue

        # Get the values for the updated label csv file
        # For some reason, CoralNet uses the Short Code as
        # the label from within a source
        a = [name, row, column, short_code]
        annotations.append(a)

    # Save the labelsets as a csv file
    basename = os.path.basename(labels_path).split('.')[0] + '_updated.dots.csv'
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

    parser.add_argument('--labelsets_path', type=str,
                        default='./CoralNet_VPI_Labelset_With_Exact_Match.csv',
                        help='The path to the CoralNet_VPI_Labelsets.csv')

    parser.add_argument('--output_dir', type=str,
                        default='./Data/',
                        help='Directory to save .csv files.')

    args = parser.parse_args()

    # Get the arguments
    labels_path = args.labels_path
    labelsets_path = args.labelsets_path
    output_dir = args.output_dir

    # Make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check that the paths exist
    assert os.path.exists(labels_path), 'ERROR: labels path does not exist'
    assert os.path.exists(labelsets_path), 'ERROR: Labelsets path does not exist'

    try:
        annotations = convert_to_coralnet(labels_path, labelsets_path, output_dir)
        print('Done.')

    except Exception as e:
        print(f'ERROR: Could not convert data\n{e}')


if __name__ == '__main__':
    main()


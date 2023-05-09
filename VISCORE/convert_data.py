import os
import json
import argparse
import pandas as pd


def convert_to_csv(dots_path, cams_path, output_dir):
    """
    Converts the dots and cams JSON files to CSV files for CoralNet. The CSV
    files are saved in the output directory.
    """

    # Open the json files
    with open(dots_path, "r") as f:
        dots_json = json.load(f)

    with open(cams_path, "r") as f:
        cams_json = json.load(f)

    # Create a dictionary, where each dot has its label
    dots = {k: "" for k in list(dots_json['dots'].keys())}
    dots = {k: dots_json['dots'][k]['lbl'] for k in dots.keys()}

    # Create a dictionary, where each cam has its name, and points,
    # containing row, column and label
    cams_dict = {"image_name": "",
                 "width": None,
                 "height": None,
                 "points": None
                 }
    # Create a dictionary for each cam
    cams = {k: cams_dict.copy() for k in list(cams_json['cams'].keys())}

    # Dataframe to hold all the data
    df = pd.DataFrame()

    # Loop through all the cams
    for cam_id in cams.keys():

        # The current cam based on the ID
        print("Processing cam: {}".format(cam_id))

        # Access the current cam
        cam = cams_json['cams'][cam_id]

        # Get the file name
        cams[cam_id]['image_path'] = cam['fn']
        cams[cam_id]['image_name'] = os.path.basename(cam['fn'])

        # Get the width and height
        cams[cam_id]['width'] = cam['wh'][0]
        cams[cam_id]['height'] = cam['wh'][1]

        # Get all the dots for the current cam
        points = []
        for d in cam['dots'].keys():
            # Access the current dot
            dot = cam['dots'][d]
            # Fill in the information
            point = {
                "column": int(dot['px'][0]),
                "row": int(dot['px'][1]),
                "label": dots[d]
            }

            points.append(point)

        # Pass the points for the cam to the dictionary
        cams[cam_id]['points'] = points

        # Create a dataframe from the current cam
        columns = ['column', 'row', 'label']
        data = pd.DataFrame(cams[cam_id]['points'], columns=columns)

        # Fill in the information
        data['image_name'] = cams[cam_id]['image_name']
        data['width'] = cams[cam_id]['width']
        data['height'] = cams[cam_id]['height']
        data['image_path'] = cams[cam_id]['image_path']
        # Concatenate the dataframes
        df = pd.concat([df, data], ignore_index=True)

    # Save the dataframe as a csv file
    basename = os.path.basename(dots_path).split(".")[0] + ".csv"
    output_file = f"{output_dir}{basename}"
    df.to_csv(output_file, index=False)

    # Check that file was saved
    if os.path.exists(output_file):
        print(f"NOTE: Successfully saved {output_file}")
    else:
        print(f"ERROR: Failed to save {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description='Convert Dots and Cams data to CSV format for CoralNet.')

    parser.add_argument('--dots_path', type=str,
                        help='The path to the dots JSON file')

    parser.add_argument('--cams_path', type=str,
                        help='The path to the cams JSON file')

    parser.add_argument('--output_dir', type=str,
                        default="./Data/",
                        help='Directory to save .csv files.')

    args = parser.parse_args()

    # Get the arguments
    dots_path = args.dots_path
    cams_path = args.cams_path
    output_dir = args.output_dir

    # Make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check that the paths exist
    assert os.path.exists(dots_path), "Dots path does not exist."
    assert os.path.exists(cams_path), "Cams path does not exist."

    convert_to_csv(dots_path, cams_path, output_dir)


if __name__ == '__main__':
    main()
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
    cams_dict = {"Name": "",
                 "Width": None,
                 "Height": None,
                 "Points": None
                 }

    cams = {k: cams_dict.copy() for k in list(cams_json['cams'].keys())}

    # Loop through all the cams
    for cam_id in cams.keys():

        # The current cam based on the ID
        print("Processing cam: {}".format(cam_id))

        # Access the current cam
        cam = cams_json['cams'][cam_id]

        # Get the file name
        cams[cam_id]['File_Path'] = cam['fn']
        cams[cam_id]['Name'] = os.path.basename(cam['fn'])

        # Get the width and height
        cams[cam_id]['Width'] = cam['wh'][0]
        cams[cam_id]['Height'] = cam['wh'][1]

        # Get all the dots for the current cam
        points = []
        for d in cam['dots'].keys():
            # Access the current dot
            dot = cam['dots'][d]
            # Fill in the information
            point = {
                "Column": int(dot['px'][0]),
                "Row": int(dot['px'][1]),
                "Label": dots[d]
            }

            points.append(point)

        # Pass the points for the cam to the dictionary
        cams[cam_id]['Points'] = points

        # Create a dataframe from the current cam
        data = cams[cam_id]
        df = pd.DataFrame(data['Points'], columns=['Column', 'Row', 'Label'])
        df['Name'] = data['Name']
        df['Width'] = data['Width']
        df['Height'] = data['Height']
        df['File_Path'] = data['File_Path']

        output_file = f"{output_dir}{data['Name'].split('.')[0]}.csv"
        df.to_csv(output_file, index=False)

        if os.path.exists(output_file):
            print(f"NOTE: Successfully saved {data['Name'].split('.')[0]}.csv")
        else:
            print(f"ERROR: Failed to save {data['Name'].split('.')[0]}.csv")


def main():

    parser = argparse.ArgumentParser(
        description='Convert Dots and Cams data to CSV format for CoralNet.')

    parser.add_argument('--dots_path', type=str,
                        help='The path to the dots JSON file')

    parser.add_argument('--cams_path', type=str,
                        help='The path to the cams JSON file')

    parser.add_argument('--output_dir', type=str,
                        default="./Data/Converted_Data/",
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
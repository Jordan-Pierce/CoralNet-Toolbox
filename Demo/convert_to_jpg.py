import os
import argparse
import numpy as np
from PIL import Image


def convert_tiff_to_jpeg(input_dir, output_dir):
    """

    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all TIFF files in the input directory
    tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

    # Convert each TIFF file to JPEG
    for tiff_file in tiff_files:
        tiff_path = os.path.join(input_dir, tiff_file)
        jpeg_file = os.path.splitext(tiff_file)[0] + '.jpg'
        jpeg_path = os.path.join(output_dir, jpeg_file)

        try:
            # Open the TIFF file using Pillow
            with Image.open(tiff_path) as img:
                # Convert the image to an RGB NumPy array
                img_array = np.array(img)

                # Create a new PIL image from the RGB array
                rgb_img = Image.fromarray(img_array)

                # Save the RGB image as JPEG with a quality of 95 (adjust as needed)
                rgb_img.save(jpeg_path, 'JPEG', quality=100)
            print(f'NOTE: Converted {tiff_file} to {jpeg_file}')
        except Exception as e:
            print(f'ERROR: Issue with converting {tiff_file}: {e}')

    print('NOTE: Conversion complete.')


def main():
    """

    """
    parser = argparse.ArgumentParser(description='Convert TIFF files to JPEG files using Pillow')

    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing TIFF files')

    parser.add_argument('--output_dir', type=str,
                        help='Output directory for saving PNG files')

    args = parser.parse_args()

    convert_tiff_to_jpeg(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

import os
import argparse


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def convert_folder_to_svg(directory):
    """
    Scans a directory for .png files and converts them to .svg format.
    The new SVG files are saved in the same directory as the original images.

    Args:
        directory (str): The path to the folder containing PNG images.
    """
    # Verify the path exists and is a directory
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Process files
    files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    
    if not files:
        print("No .png files found in the specified directory.")
        return

    print(f"Found {len(files)} PNG file(s). Starting conversion...")

    for filename in files:
        input_path = os.path.join(directory, filename)
        
        # Create output path by changing the extension
        output_filename = os.path.splitext(filename)[0] + ".svg"
        output_path = os.path.join(directory, output_filename)

        print(f"Converting: {filename} -> {output_filename}")
        
        try:
            # vtracer conversion with default settings
            vtracer.convert_image_to_svg_py(input_path, output_path)
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")


if __name__ == "__main__":
    
    import vtracer

    # Setup argparse for command-line execution
    parser = argparse.ArgumentParser(description="Convert all PNG files in a directory to SVG.")
    parser.add_argument(
        "path", 
        type=str, 
        help="The folder path containing the .png images."
    )
    
    args = parser.parse_args()
    convert_folder_to_svg(args.path)

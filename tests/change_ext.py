import cv2
import argparse
import os
from pathlib import Path
import multiprocessing as mp


def change_extension(image_path: str, output_format: str) -> str:
    """
    Convert the image format using OpenCV.

    Args:
        image_path (str): Path to the input image
        output_format (str): Desired output format (e.g., 'png', 'jpg')

    Returns:
        str: Path to the converted image
    """
    try:
        # Load the image using OpenCV's imread function
        image = cv2.imread(str(image_path))
        
        # Check if image was loaded successfully
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        
        # Determine the output file path
        output_file = str(Path(image_path).with_suffix(f'.{output_format}'))
        
        # Save the image in the desired format
        cv2.imwrite(output_file, image)
        
        return output_file
        
    except Exception as e:
        # Print error message and return None if any exception occurs
        print(f"Error processing image: {str(e)}")
        return None


def process_single_image(args):
    """
    Process and convert a single image to the specified format.

    Args:
        args (tuple): A tuple containing:
            - img_path (Path): Path to the source image file
            - output_format (str): Desired output format (e.g., 'png', 'jpg')
            - output_path (Path, optional): Path to output directory. If None, creates 'converted' subdirectory

    Returns:
        None: Writes converted image to disk and prints confirmation message
    """
    # Unpack the input arguments
    img_path, output_format, output_path = args
    
    # Convert the image using the change_extension function
    converted = change_extension(img_path, output_format)
    
    if converted is not None:
        # Determine output directory - use provided path or create 'converted' subdirectory
        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = img_path.parent / 'converted'
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename and save the converted image
        output_file = output_dir / f'converted_{img_path.stem}.{output_format}'
        os.rename(converted, output_file)
        print(f"Saved converted image to {output_file}")


def process_path(input_path: str, output_format: str, output_path: str, extension: str) -> None:
    """
    Process a single image or all images in a directory to convert their format.

    Args:
        input_path (str): Path to the input image file or directory
        output_format (str): Desired output format (e.g., 'png', 'jpg')
        output_path (str): Path to the output directory. If None, creates 'converted' subdirectory
        extension (str): Image file extension to process (e.g., 'jpg', 'png')

    Returns:
        None: Processes images and saves converted versions to disk
    """
    # Convert input path to Path object and ensure extension is lowercase
    input_path = Path(input_path)
    extension = extension.lower()
    
    # Check if input path is a file or directory
    if input_path.is_file():
        # Process single image if it matches the specified extension
        if input_path.suffix.lower() == f'.{extension}':
            # Process single image using the process_single_image function
            process_single_image((input_path, output_format, output_path))
    
    # Process all images in a directory
    elif input_path.is_dir():
        # Gather all image paths
        img_paths = list(input_path.glob(f'*.{extension}'))
        
        # Prepare arguments for parallel processing
        process_args = [(p, output_format, output_path) for p in img_paths]
        
        # Create a process pool and process images in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(process_single_image, process_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image(s) to specified format")
    parser.add_argument("input_path", type=str, help="Path to input image or directory")
    parser.add_argument("--format", type=str, required=True, help="Desired output format (e.g., 'png', 'jpg')")
    parser.add_argument("--output", type=str, help="Output path for converted image(s)")
    parser.add_argument("--extension", type=str, default="jpg", help="Image file extension to process")
    
    args = parser.parse_args()
    
    process_path(args.input_path, args.format, args.output, args.extension)

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import multiprocessing as mp
from functools import partial


def resize_image(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    """
    Read and resize an image to specified dimensions using OpenCV.

    Args:
        image_path (str): Path to the input image
        target_width (int): Desired width in pixels
        target_height (int): Desired height in pixels

    Returns:
        np.ndarray: Resized image as numpy array, or None if error occurs
    """
    try:
        # Load the image using OpenCV's imread function
        image = cv2.imread(str(image_path))
        
        # Check if image was loaded successfully
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
            
        # Resize the image to target dimensions using area interpolation
        # INTER_AREA is preferred for downsampling as it gives better quality
        resized_image = cv2.resize(image, 
                                   (target_width, target_height), 
                                   interpolation=cv2.INTER_AREA)
        
        return resized_image
        
    except Exception as e:
        # Print error message and return None if any exception occurs
        print(f"Error processing image: {str(e)}")
        return None

def process_single_image(args):
    """Process and resize a single image to specified dimensions.

    This function takes an image file, resizes it to target dimensions, and saves
    the result to a specified output directory. If no output directory is provided,
    it creates a 'resized' subdirectory in the source image's location.

    Args:
        args (tuple): A tuple containing:
            - img_path (Path): Path to the source image file
            - target_width (int): Desired width in pixels for the resized image
            - target_height (int): Desired height in pixels for the resized image
            - output_path (Path, optional): Path to output directory. If None, creates 'resized' subdirectory

    Returns:
        None: Writes resized image to disk and prints confirmation message

    Example:
        args = (Path('image.jpg'), 800, 600, Path('output'))
        process_single_image(args)
    """
    # Unpack the input arguments
    img_path, target_width, target_height, output_path = args
    
    # Resize the image using the resize_image function
    resized = resize_image(img_path, target_width, target_height)
    
    if resized is not None:
        # Determine output directory - use provided path or create 'resized' subdirectory
        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = img_path.parent / 'resized'
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename and save the resized image
        output_file = output_dir / f'resized_{img_path.name}'
        cv2.imwrite(str(output_file), resized)
        print(f"Saved resized image to {output_file}")

def process_path(input_path: str, target_width: int, 
                 target_height: int, 
                 output_path: str, extension: str) -> None:
    """
    Process a single image or all images in a directory to resize them.
    
    Args:
        input_path (str): Path to the input image file or directory
        target_width (int): Desired width in pixels for the resized image(s)
        target_height (int): Desired height in pixels for the resized image(s)
        output_path (str): Path to the output directory. If None, creates 'resized' subdirectory
        extension (str): Image file extension to process (e.g., 'jpg', 'png')
        
    Returns:
        None: Processes images and saves resized versions to disk
    """
    # Convert input path to Path object and ensure extension is lowercase
    input_path = Path(input_path)
    extension = extension.lower()
    
    # Check if input path is a file or directory
    if input_path.is_file():
        # Process single image if it matches the specified extension
        if input_path.suffix.lower() == f'.{extension}':
            # Process single image using the process_single_image function
            process_single_image((input_path, target_width, target_height, output_path))
    
    # Process all images in a directory
    elif input_path.is_dir():
        # Gather all image paths
        img_paths = list(input_path.glob(f'*.{extension}'))
        
        # Prepare arguments for parallel processing
        process_args = [(p, target_width, target_height, output_path) for p in img_paths]
        
        # Create a process pool and process images in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(process_single_image, process_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize image(s) to specified dimensions')
    parser.add_argument('input_path', type=str, help='Path to input image or directory')
    parser.add_argument('--width', type=int, default=640, help='Target width in pixels')
    parser.add_argument('--height', type=int, default=480, help='Target height in pixels')
    parser.add_argument('--output', type=str, help='Output path for resized image(s)')
    parser.add_argument('--extension', type=str, default='jpg', help='Image file extension to process')
    
    args = parser.parse_args()
    
    process_path(args.input_path, args.width, args.height, args.output, args.extension)
    
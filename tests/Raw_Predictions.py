import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_now():
    """Get current date and time"""
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


def get_image_paths(test_dir):
    """Get all image paths and their true labels from subdirectories"""
    image_paths = []
    labels = []
    test_path = Path(test_dir)
    
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(str(img_path))
                    labels.append(class_dir.name)
    
    return image_paths, labels


def main():
    
    parser = argparse.ArgumentParser(description='Run YOLO predictions on test folder')
    
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to .pt model file')
    
    parser.add_argument('--test_dir', type=str, required=True, 
                        help='Path to test directory containing class subdirectories')
    
    parser.add_argument('--output_dir', type=str, default="./",
                        help='Path to output directory to save output CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.model)
    
    # Get image paths and true labels
    image_paths, true_labels = get_image_paths(args.test_dir)
    
    # Prepare data storage
    results_data = []
    
    # Process images in batches
    results = model(image_paths, stream=True)
    
    # Process results
    for img_path, true_label, result in tqdm(zip(image_paths, true_labels, results), total=len(image_paths)):
        if result.probs is None:
            continue
            
        # Get top 5 predictions and probabilities
        top5_indices = result.probs.top5
        top5_probs = result.probs.top5conf
        top5_classes = [result.names[i] for i in top5_indices]
        
        # Store results
        row_data = {
            'Path': img_path,
            'Label': true_label
        }
        
        # Add top 5 predictions and their probabilities
        for i, (cls, prob) in enumerate(zip(top5_classes, top5_probs), 1):
            row_data[f'Machine suggestion {i}'] = cls
            row_data[f'Machine confidence {i}'] = float(prob)
            
        results_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Save results
    output_path = f"{args.output_dir}/{get_now()}_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
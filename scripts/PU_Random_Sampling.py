import os
import yaml
import shutil
import random
import argparse
from pathlib import Path

def create_noisy_yolo_dataset(input_yaml, output_folder, splits_to_drop, drop_ratio):
    try:
        with open(input_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return

    # Handle absolute vs relative pathing for the base directory
    yaml_path_val = config.get('path', '')
    if os.path.isabs(yaml_path_val):
        original_base_path = Path(yaml_path_val)
    else:
        original_base_path = Path(input_yaml).parent / yaml_path_val

    output_base_path = Path(output_folder).absolute()

    # 1. Copy the entire dataset
    print(f"[*] Copying dataset to: {output_base_path}")
    if output_base_path.exists():
        shutil.rmtree(output_base_path)
    shutil.copytree(original_base_path, output_base_path, ignore=shutil.ignore_patterns('*.py', '.git', '__pycache__'))

    total_boxes_dropped = 0
    total_files_modified = 0

    # 2. Iterate through each requested split
    for split_key in splits_to_drop:
        # Check if the split exists in YAML (handling 'val'/'valid' mapping)
        actual_key = split_key
        if split_key not in config:
            if split_key == 'val' and 'valid' in config:
                actual_key = 'valid'
            elif split_key == 'valid' and 'val' in config:
                actual_key = 'val'
            else:
                print(f"[!] Warning: Split '{split_key}' not found in YAML. Skipping.")
                continue

        img_split_path = Path(config.get(actual_key))
        split_folder_name = img_split_path.name 
        
        # Target: output_folder/train/labels
        target_label_dir = output_base_path / split_folder_name / "labels"

        if not target_label_dir.exists():
            print(f"[!] Warning: Label directory not found at {target_label_dir}. Skipping '{split_key}'.")
            continue

        # 3. Drop the boxes for this split
        print(f"[*] Processing '{split_key}' labels in: {target_label_dir}")
        
        split_boxes_dropped = 0
        split_files_modified = 0

        for label_file in os.listdir(target_label_dir):
            if label_file.endswith('.txt') and label_file != "classes.txt":
                file_path = target_label_dir / label_file
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                new_lines = [line for line in lines if random.random() > drop_ratio]
                dropped_count = len(lines) - len(new_lines)
                
                split_boxes_dropped += dropped_count
                total_boxes_dropped += dropped_count
                
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                split_files_modified += 1
                total_files_modified += 1
        
        print(f"    -> Dropped {split_boxes_dropped} boxes across {split_files_modified} files.")

    # 4. Update YAML paths to be absolute and relative to new folder
    config['path'] = str(output_base_path).replace("\\", "/")
    for s in ['train', 'val', 'test', 'valid']:
        if s in config:
            old_path = Path(config[s])
            config[s] = str(output_base_path / old_path.name).replace("\\", "/")

    new_yaml_name = f"data_dropped_{int(drop_ratio*100)}.yaml"
    new_yaml_path = output_base_path / new_yaml_name
    
    with open(new_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n--- Final Summary ---")
    print(f"Total Files Modified: {total_files_modified}")
    print(f"Total Boxes Dropped:  {total_boxes_dropped}")
    print(f"New Config Created:   {new_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop YOLO boxes from multiple splits.")
    parser.add_argument("--yaml", type=str, required=True, help="Input data.yaml")
    parser.add_argument("--out", type=str, required=True, help="New dataset directory")
    # nargs='+' allows multiple values: --split train val test
    parser.add_argument("--split", type=str, nargs='+', default=["train"], help="Space-separated splits (e.g., train val)")
    parser.add_argument("--ratio", type=float, default=0.5, help="0 to 1 ratio of boxes to drop")

    args = parser.parse_args()
    create_noisy_yolo_dataset(args.yaml, args.out, args.split, args.ratio)
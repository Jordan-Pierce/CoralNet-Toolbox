import argparse
import yaml
from pathlib import Path

def update_yolo_paths(root_folder, output_path=None):
    # Resolve the absolute path of the root directory
    root_path = Path(root_folder).resolve()

    master_train = []
    master_val = []
    names = None
    nc = None

    # Search recursively for all data.yaml files
    print(f"Scanning for data.yaml files in: {root_path}...\n")
    
    # Use rglob to find all data.yaml files in the directory and subdirectories
    for yaml_path in root_path.rglob('data.yaml'):
        # Skip the master_data.yaml if we run this script multiple times
        if yaml_path.name == 'master_data.yaml':
            continue

        with open(yaml_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error reading {yaml_path}: {e}")
                continue

        # Validate that this is a YOLO dataset yaml
        if not data or 'train' not in data or 'val' not in data:
            print(f"Skipping {yaml_path}: missing 'train' or 'val' keys.")
            continue

        # Capture the class names and nc from the first valid dataset we find
        if names is None and 'names' in data:
            names = data['names']
        if nc is None and 'nc' in data:
            nc = data['nc']

        # Get the absolute path of the directory containing THIS data.yaml
        current_dir = yaml_path.parent.resolve()

        # Dynamically extract the folder structure (e.g., 'train/images') from the old path.
        # Normalizing slashes first to handle Windows/Linux mixing as seen in the example.
        old_train_parts = data['train'].replace('\\', '/').split('/')
        old_val_parts = data['val'].replace('\\', '/').split('/')

        # We assume the last two parts of the original path are the relative folders (e.g., 'train', 'images')
        new_train = current_dir / old_train_parts[-2] / old_train_parts[-1]
        new_val = current_dir / old_val_parts[-2] / old_val_parts[-1]

        # Update the dictionary using forward slashes (YOLO handles forward slashes universally)
        data['train'] = new_train.as_posix()
        data['val'] = new_val.as_posix()

        # Save the updated data.yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Updated: {yaml_path}")

        # Append the new absolute paths to our master lists
        master_train.append(data['train'])
        master_val.append(data['val'])

    # ---------------------------------------------------------
    # Create the master_data.yaml file
    # ---------------------------------------------------------
    if master_train and master_val:
        master_data = {
            'names': names,
            'nc': nc,
            'train': master_train,
            'val': master_val
        }

        master_path = Path(output_path).resolve() if output_path else root_path / 'master_data.yaml'
        
        with open(master_path, 'w') as f:
            yaml.dump(master_data, f, default_flow_style=False, sort_keys=False)

        print(f"\nSuccess! Created master configuration at:")
        print(f"{master_path}")
        print(f"Total datasets aggregated: {len(master_train)}")
    else:
        print("\nNo valid data.yaml files found to create a master dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine YOLO data.yaml files into a master dataset configuration.")
    parser.add_argument("root_folder", help="Absolute path to the root folder of your datasets.")
    parser.add_argument("--output", "-o", default=None, help="File path to save master_data.yaml (default: root_folder/master_data.yaml).")
    args = parser.parse_args()
    update_yolo_paths(args.root_folder, args.output)
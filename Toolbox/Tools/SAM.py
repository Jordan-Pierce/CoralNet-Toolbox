import os
import sys
import requests
from tqdm import tqdm

import numpy as np
from skimage.io import imread
from skimage.io import imsave
from scipy.stats import mode as mode2d

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from segment_anything import SamPredictor
from segment_anything import sam_model_registry

from Toolbox.Tools import *
from Toolbox.Tools.Inference import get_class_map

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, bboxes):
        self.data = bboxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def download_checkpoint(url, path):
    """

    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                # Write the content to the file
                file.write(response.content)
            print(f"NOTE: Downloaded file successfully")
            print(f"NOTE: Saved file to {path}")
        else:
            print(f"ERROR: Failed to download file. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred: {e}")


def get_sam_predictor(model_type="vit_l", device='cpu'):
    """

    """
    # URL to download pre-trained weights
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/"

    # The path containing the weights
    sam_root = f"{CACHE_DIR}\\SAM_Weights"
    os.makedirs(sam_root, exist_ok=True)

    # Mapping between the model type, and the checkpoint file name
    sam_dict = {"vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"}

    if model_type not in list(sam_dict.keys()):
        print(f"ERROR: Invalid model type provided; choices are:\n{list(sam_dict.keys())}")
        sys.exit(1)

    # Checkpoint path to model
    path = f"{sam_root}\\{sam_dict[model_type]}"

    # Check to see if the weights of the model type were already downloaded
    if not os.path.exists(path):
        print("NOTE: Model checkpoint does not exist; downloading")
        url = f"{sam_url}{sam_dict[model_type]}"
        # Download the file
        download_checkpoint(url, path)

    # Loading the mode, returning the predictor
    sam_model = sam_model_registry[model_type](checkpoint=path)
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)

    return sam_predictor


def get_bbox(image, y, x, patch_size=224):
    """
    Given an image, and a Y, X location, this function will return a bounding box.
    """

    height, width, _ = image.shape

    # N x N
    size = patch_size // 2

    # Top of the patch, else edge of image
    top = y - size
    if top < 0:
        top = 0

    # Bottom of patch, else edge of image
    bottom = y + size
    if bottom > height:
        bottom = height

    # Left of patch, else edge of image
    left = x - size
    if left < 0:
        left = 0

    # Right of patch, else edge of image
    right = x + size
    if right > width:
        right = width

    # Bounding Box
    bbox = [left, top, right, bottom]

    return bbox


def find_most_common_label_in_area(points, binary_mask, bounding_box):
    """

    """

    # Get the coordinates of the bounding box
    min_x, min_y, max_x, max_y = bounding_box

    # Filter points within the bounding box
    points_in_area = points[(points['Column'] >= min_x) & (points['Column'] <= max_x) &
                            (points['Row'] >= min_y) & (points['Row'] <= max_y)]

    # Filter points that correspond to 1-valued regions in the binary mask
    mask_indices = points_in_area.apply(lambda row: binary_mask[row['Row'], row['Column']].item(), axis=1)
    points_in_mask = points_in_area[mask_indices == 1]

    # Find the most common label
    most_common_label = mode2d(points_in_mask['Label'])[0][0]

    return most_common_label


def get_color_map(N):
    """

    """
    # Calculate angle intervals given number of classes
    angle_step = 360.0 / N
    angles = [angle_step * i for i in range(N)]

    # For each angle interval, calculate a color in RGB space
    # that maximizes distance from one class to another
    color_coordinates = []

    for angle in angles:
        r = int(255 * (1 + np.cos(np.radians(angle))) / 2)
        g = int(255 * (1 + np.cos(np.radians(angle + 120))) / 2)
        b = int(255 * (1 + np.cos(np.radians(angle + 240))) / 2)
        color_coordinates.append([r, g, b])

    return np.array(color_coordinates)


def colorize_mask(mask, class_map, label_colors):
    """

    """
    # Initialize the RGB mask with zeros
    height, width = mask.shape
    rgb_mask = np.full((height, width, 3), fill_value=255, dtype=np.uint8)

    # dict with index as key, rgb as value
    cmap = {v: label_colors[k][0:3] for k, v in class_map.items()}

    # Loop through all index values
    # Set rgb color in colored mask
    for val in np.unique(mask):
        if val in class_map.values():
            color = np.array(cmap[val]) * 255
            rgb_mask[mask == val, :] = color.astype(np.uint8)

    return rgb_mask.astype(np.uint8)


def plot_mask(image, mask_color, points, point_colors, plot_path):
    """

    """

    # Plot title
    fname = os.path.basename(plot_path)

    # Plot masks
    plt.figure(figsize=(10, 10))
    plt.title(fname)
    plt.imshow(image)
    plt.imshow(mask_color, alpha=.75)
    plt.scatter(points['Column'].values, points['Row'].values, c=point_colors, s=1)

    # Save the plot
    plt.savefig(plot_path)
    plt.close()


def mss_sam(args):
    """

    """
    print("\n###############################################")
    print("Multilevel Superpixel Segmentation w/ SAM")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Predictions Dataframe
    if os.path.exists(args.annotations):
        points = pd.read_csv(args.annotations, index_col=0)
        image_names = np.unique(points['Image Name'].to_numpy())
        print(f"NOTE: Found a total of {len(points)} sampled points for {len(image_names)} images")
    else:
        print("ERROR: Points file provided doesn't exist.")
        sys.exit(1)

    # Image files
    if os.path.exists(args.images):
        # Images that are the correct image format
        images = [i for i in glob.glob(f"{args.images}/*.*") if i.split(".")[-1].lower() in IMG_FORMATS]
        # Subset the images list to only contain those with points
        images = [i for i in images if os.path.basename(i) in image_names]
        if not images:
            raise Exception(f"ERROR: No images were found in the directory provided; please check input.")
        else:
            print(f"NOTE: Found {len(images)} images in directory provided")
    else:
        print("ERROR: Image directory provided doesn't exist.")
        sys.exit(1)

    # Class map, Color map
    if os.path.exists(args.class_map):
        # Get the class map, adjust for this tool
        class_map = get_class_map(args.class_map)
        class_map = {v: int(k) for k, v in class_map.items()}
        # Create a color map give the amount of classes
        unique_labels = list(class_map.keys())
        color_map = get_color_map(len(unique_labels))
        # Get the colors per class
        label_colors = {l: color_map[i]/255.0 for i, l in enumerate(unique_labels)}
    else:
        print(f"ERROR: Class Map file provided doesn't exist.")
        sys.exit(1)

    # Model Weights
    try:
        # Load the model with custom metrics
        sam_predictor = get_sam_predictor(args.model_type, device)
        print(f"NOTE: Loaded model {args.model_type}")

    except Exception as e:
        print(f"ERROR: There was an issue loading the model\n{e}")
        sys.exit(1)

    # Setting output directories
    output_dir = f"{args.output_dir}\\masks\\{get_now()}\\"
    plot_dir = f"{output_dir}\\plots\\"
    seg_dir = f"{output_dir}\\segs\\"
    color_dir = f"{output_dir}\\color\\"

    output_mask_csv = f"{output_dir}masks.csv"
    output_color_json = f"{output_dir}Color_Map.json"

    # Create the output directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    # Output for mask dataframe
    mask_df = []

    # Batch size
    batch_size = args.batch_size

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    print("\n###############################################")
    print("Making Masks")
    print("###############################################\n")

    # Loop through each image, extract the corresponding patches
    for i_idx, image_path in enumerate(images):

        # Get the points associated with current image
        name = os.path.basename(image_path)
        current_points = points[points['Image Name'] == name]

        # Skip if there are no points for some reason
        if current_points.empty:
            continue

        # Read the image, get the points, create bounding boxes
        image = imread(image_path)

        print(f"NOTE: Making predictions for {name}")
        # Set the image in sam predictor
        sam_predictor.set_image(image)

        # To hold the updated mask, will be added onto each iteration
        # updated_mask = np.full(shape=image.shape[:2], fill_value=255)
        updated_mask = torch.full(image.shape[:2], fill_value=255, dtype=torch.uint8).to(device)

        # Get all the bounding boxes for the current image
        bboxes = []

        for i, r in current_points.iterrows():
            bboxes.append(get_bbox(image, r['Row'], r['Column'], args.patch_size))

        # Create into a tensor
        bboxes = np.array(bboxes)
        transformed_boxes = torch.tensor(bboxes, device=sam_predictor.device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(transformed_boxes, image.shape[:2])

        # Create a data loader containing the transformed boxes
        custom_dataset = CustomDataset(transformed_boxes)
        data_loader = DataLoader(custom_dataset, batch_size=64, shuffle=False)

        # Loop through batches of boxes, faster
        for batch_idx, batch in enumerate(data_loader):

            try:
                # After setting the current image, get masks for each point / bbox
                masks, _, _ = sam_predictor.predict_torch(point_coords=None,
                                                          point_labels=None,
                                                          boxes=batch,
                                                          multimask_output=False)
            except Exception as e:
                print(f"ERROR: Model could not make predictions\n{e}")
                sys.exit(1)

            # Loop through all the individual masks in the batch
            for m_idx, mask in enumerate(masks):

                try:
                    # CPU Mask
                    mask = mask.squeeze()
                    # Get the current box
                    box = bboxes[batch_idx * batch_size: (batch_idx + 1) * batch_size][m_idx]
                    # Find the most common label within the binary mask (1)
                    label = find_most_common_label_in_area(current_points, mask, box)
                    # convert binary values to correspond to label values
                    updated_mask[mask == 1] = int(class_map[label])
                except:
                    pass

            # Create a screenshot every N% of the number of points
            if batch_idx % int(len(data_loader) * .2) == 0 and args.plot_progress:
                # Colorize the updated mask
                mask_color = colorize_mask(updated_mask.cpu().detach().numpy(), class_map, label_colors)
                point_colors = current_points['Label'].map(label_colors).values
                # Plot and save the mask
                plot_path = f"{plot_dir}{name.split('.')[0]}_{str(batch_idx)}.jpg"
                plot_mask(image, mask_color, current_points, point_colors, plot_path)

        # ------------------------------------------------
        # Save the final masks
        # ------------------------------------------------

        # Convert to numpy for plotting, saving
        final_mask = updated_mask.cpu().detach().numpy()

        # Get the final colored mask, change no data to black
        final_color = colorize_mask(final_mask, class_map, label_colors)
        final_color[final_mask == 255, :] = [0, 0, 0]
        point_colors = current_points['Label'].map(label_colors).values

        # Final figure
        if args.plot:
            # Plot the final mask
            fname = f"{name.split('.')[0]}.jpg"
            plot_path = f"{plot_dir}{fname}"
            plot_mask(image, final_color, current_points, point_colors, plot_path)
            print(f"NOTE: Saved plot to {plot_path}")

        else:
            plot_path = ""

        # Save the seg mask
        mask_path = f"{seg_dir}{name}"
        imsave(fname=mask_path, arr=final_mask.astype(np.uint8))
        print(f"NOTE: Saved seg mask to {mask_path}")

        # Save the color mask
        color_path = f"{color_dir}{name}"
        imsave(fname=color_path, arr=final_color.astype(np.uint8))
        print(f"NOTE: Saved color mask to {color_path}")

        # Add to output list
        mask_df.append([image_path, mask_path, color_path, plot_path])

        # Gooey
        print_progress(i_idx, len(image_names))

    # Save dataframe to root directory
    mask_df = pd.DataFrame(mask_df, columns=['Image Path', 'Seg Path', 'Color Path', 'Plot Path'])
    mask_df.to_csv(output_mask_csv)

    if os.path.exists(output_mask_csv):
        print(f"NOTE: Mask dataframe saved to {output_dir}")
    else:
        print(f"ERROR: Could not save mask dataframe")

    # Create a final class mapping for the seg masks
    seg_map = {k: {} for k in class_map.keys()}

    for l in class_map.keys():
        seg_map[l]['id'] = class_map[l]
        seg_map[l]['color'] = (np.array(label_colors[l][0:3]) * 255).astype(np.uint8).tolist()

    # Save the color mapping json file
    with open(output_color_json, 'w') as output_file:
        json.dump(seg_map, output_file, indent=4)

    if os.path.exists(output_color_json):
        print(f"NOTE: Color Mapping JSON file saved to {output_dir}")
    else:
        print(f"ERROR: Could not save Color Mapping JSON file")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Multilevel Superpixel Segmentation w/ SAM")

    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images to perform inference on")

    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to the points file containing 'Name', 'Row', 'Column', and 'Label' information.")

    parser.add_argument("--patch_size", type=int, default=360,
                        help="The approximate size of each superpixel formed by SAM")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of samples passed to SAM in a batch (GPU dependent)")

    parser.add_argument("--model_type", type=str, default='vit_l',
                        help="Model to use; one of ['vit_b', 'vit_l', 'vit_h']")

    parser.add_argument("--class_map", type=str, required=True,
                        help="Path to the model's Class Map JSON file")

    parser.add_argument("--plot", action='store_true',
                        help="Saves figures of final masks")

    parser.add_argument("--plot_progress", action='store_true',
                        help="Saves figures of masks throughout the process of creation")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        mss_sam(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

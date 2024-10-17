import os
import glob
import shutil

from sklearn.model_selection import train_test_split

# Folder containing with and without folders of images
root = "C:/Users/jordan.pierce/Documents/GitHub/Marine-Debris/Data"
os.makedirs(root, exist_ok=True)

map_folders = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]

if True:
    for map_folder in map_folders:
        with_folder = os.path.join(map_folder, "with")
        without_folder = os.path.join(map_folder, "without")
        assert os.path.exists(with_folder), f"ERROR: {with_folder} does not exist."
        assert os.path.exists(without_folder), f"ERROR: {without_folder} does not exist."

        with_images = glob.glob(os.path.join(with_folder, "*.jpeg"))
        without_images = glob.glob(os.path.join(without_folder, "*.jpeg"))

        # Split the images into train, valid, and test sets
        with_train, with_test = train_test_split(with_images, test_size=0.2)
        with_train, with_valid = train_test_split(with_train, test_size=0.2)

        without_train, without_test = train_test_split(without_images, test_size=0.2)
        without_train, without_valid = train_test_split(without_train, test_size=0.2)

        # Save the images to the appropriate folders (train, valid, and test)
        for image in with_train:
            os.makedirs(os.path.join(root, "train", "with"), exist_ok=True)
            shutil.copy(image, os.path.join(root, "train", "with"))
        for image in with_valid:
            os.makedirs(os.path.join(root, "valid", "with"), exist_ok=True)
            shutil.copy(image, os.path.join(root, "valid", "with"))
        for image in with_test:
            os.makedirs(os.path.join(root, "test", "with"), exist_ok=True)
            shutil.copy(image, os.path.join(root, "test", "with"))

        for image in without_train:
            os.makedirs(os.path.join(root, "train", "without"), exist_ok=True)
            shutil.copy(image, os.path.join(root, "train", "without"))
        for image in without_valid:
            os.makedirs(os.path.join(root, "valid", "without"), exist_ok=True)
            shutil.copy(image, os.path.join(root, "valid", "without"))
        for image in without_test:
            os.makedirs(os.path.join(root, "test", "without"), exist_ok=True)
            shutil.copy(image, os.path.join(root, "test", "without"))

# Print the number of images in each folder
print(f"Train with: {len(glob.glob(os.path.join(root, 'train', 'with', '*.jpeg')))}")
print(f"Valid with: {len(glob.glob(os.path.join(root, 'valid', 'with', '*.jpeg')))}")
print(f"Test with: {len(glob.glob(os.path.join(root, 'test', 'with', '*.jpeg')))}")

print(f"Train without: {len(glob.glob(os.path.join(root, 'train', 'without', '*.jpeg')))}")
print(f"Valid without: {len(glob.glob(os.path.join(root, 'valid', 'without', '*.jpeg')))}")
print(f"Test without: {len(glob.glob(os.path.join(root, 'test', 'without', '*.jpeg')))}")
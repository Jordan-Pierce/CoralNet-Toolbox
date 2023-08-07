import os
import glob
import argparse

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.patches import Rectangle

matplotlib.use('TkAgg')

from Toolbox.Tools import IMG_FORMATS

# Hide the default interactive toolbar
plt.rcParams['toolbar'] = 'None'

# TODO create a savefigure button
class ImageViewer:
    def __init__(self, image_files, annotations):
        self.image_files = image_files
        self.annotations = annotations
        self.current_index = 0
        self.show_annotations = 'points'  # Initial state: annotations are visible as points
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2, right=0.75)  # Adjust right side for buttons and legend

        # Adjust the 'Label' column in annotations to remove leading underscores
        self.annotations['Label'] = self.annotations['Label'].str.lstrip('_')

        # Get all unique class categories from the entire DataFrame
        self.all_class_categories = self.annotations['Label'].unique()

        # Generate a colormap with a fixed number of colors for each class category
        self.color_map = plt.cm.tab20(np.linspace(0, 1, len(self.all_class_categories)))

        self.show_image()
        self.create_buttons()

        # Maximize the figure window (works with 'TkAgg' backend)
        self.fig.canvas.manager.window.state('zoomed')

    def show_image(self):
        self.ax.clear()
        image_path = self.image_files[self.current_index]
        image = plt.imread(image_path)
        self.ax.imshow(image)
        self.ax.axis('off')
        filename = os.path.basename(image_path)
        self.ax.set_title(f"{filename} {' ' * 150} {self.current_index + 1} / {len(self.image_files)}")

        # Get annotations for the current image
        current_image_name = os.path.basename(filename)
        current_annotations = self.annotations[self.annotations['Name'] == current_image_name]

        # Create legend with all class categories and corresponding colors
        legend_elements = []
        for class_category, color in zip(self.all_class_categories, self.color_map):
            legend_elements.append(plt.Line2D([0], [0],
                                              marker='o',
                                              color='w',
                                              label=class_category,
                                              markerfacecolor=color,
                                              markersize=12))

        self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Display annotations as points on the image (if annotations are visible)
        if self.show_annotations == 'points':
            for i, r in current_annotations.iterrows():
                row = int(r['Row'])
                col = int(r['Column'])
                class_category = r['Label']
                color_index = np.where(self.all_class_categories == class_category)[0][0]
                color = self.color_map[color_index]
                self.ax.plot(col, row, marker='o', markersize=8, color=color, linestyle='', markeredgecolor='black')

        elif self.show_annotations == 'squares':
            square_size = 224  # Size of the square patch (can be adjusted as needed)
            for _, row in current_annotations.iterrows():
                row_val = int(row['Row'])
                col_val = int(row['Column'])
                class_category = row['Label']
                color_index = np.where(self.all_class_categories == class_category)[0][0]
                color = self.color_map[color_index]

                # Calculate coordinates for the square
                x = col_val - square_size // 2
                y = row_val - square_size // 2

                # Draw the square with the same color as the corresponding point
                square = Rectangle((x, y), square_size, square_size, linewidth=2, edgecolor=color, facecolor='none')
                self.ax.add_patch(square)
        else:
            pass  # No annotations (do nothing)

        plt.draw()

    def on_backward(self, event):
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.show_image()

    def on_forward(self, event):
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.show_image()

    def on_home(self, event):
        self.current_index = 0
        self.show_image()

    def on_toggle_annotations(self, event):
        # Cycle through different annotation display modes (points, squares, off)
        if self.show_annotations == 'points':
            self.show_annotations = 'squares'
        elif self.show_annotations == 'squares':
            self.show_annotations = 'off'
        else:
            self.show_annotations = 'points'

        self.show_image()

    def on_go_to_image(self, event):
        try:
            index = int(event)
            if 1 <= index <= len(self.image_files):
                self.current_index = index - 1
                self.show_image()
            else:
                print("Invalid image index. Please enter a valid image index.")
        except ValueError:
            print("Invalid input. Please enter a valid image index (numeric value).")

    def create_buttons(self):
        ax_home = plt.axes([0.1, 0.05, 0.1, 0.075])
        self.btn_home = Button(ax_home, 'Home')
        self.btn_home.on_clicked(self.on_home)

        ax_backward = plt.axes([0.25, 0.05, 0.1, 0.075])
        self.btn_backward = Button(ax_backward, '←')
        self.btn_backward.on_clicked(self.on_backward)

        ax_forward = plt.axes([0.4, 0.05, 0.1, 0.075])
        self.btn_forward = Button(ax_forward, '→')
        self.btn_forward.on_clicked(self.on_forward)

        ax_toggle_annotations = plt.axes([0.55, 0.05, 0.1, 0.075])
        self.btn_toggle_annotations = Button(ax_toggle_annotations, 'Toggle Annotations')
        self.btn_toggle_annotations.on_clicked(self.on_toggle_annotations)

        ax_go_to_image = plt.axes([0.70, 0.05, 0.1, 0.075])
        self.txt_go_to_image = TextBox(ax_go_to_image, 'Go to:', initial='1')
        self.txt_go_to_image.on_submit(self.on_go_to_image)


def visualize(args):
    """

    """
    # Set the backend to 'TkAgg' for an external viewer

    print("\n###############################################")
    print("Visualize")
    print("###############################################\n")

    # Pass the variables
    image_dir = args.image_dir
    annotations = args.annotations

    if not os.path.exists(image_dir):
        raise Exception("ERROR: Image directory does not exists; please check input.")
    else:
        # Create a list of image file paths in the specified directory
        image_files = [f for f in glob.glob(f"{image_dir}\*.*") if f.split(".")[-1].lower() in IMG_FORMATS]

        if image_files is []:
            print("NOTE: No images found; exiting")
            return

    if not os.path.exists(annotations):
        print("WARNING: Annotations provided, but they doe not exists; please check input.")
    else:
        annotations = pd.read_csv(annotations, index_col=0)

    # Create the ImageViewer object with the list of images
    image_viewer = ImageViewer(image_files, annotations)

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="View images and annotations")

    parser.add_argument("--image_dir", required=True, type=str,
                        help="Path to the directory containing the images.")

    parser.add_argument("--annotations", required=False, type=str,
                        help='Path to Annotations dataframe.')

    args = parser.parse_args()

    try:
        visualize(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
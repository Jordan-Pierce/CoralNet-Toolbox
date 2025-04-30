A **big** thanks to the researchers at the research arm of the Seattle Aquarium for providing this guide.

```bibtex
@misc{williams2025,
  author = {Williams, Megan},
  title = {SOP to train a classification model in Toolbox},
  institution = {Seattle Aquarium},
  date = {2025-04-04}
}
```

# SOP to train a classification model in Toolbox

The following steps are required to create a training dataset, train a classification model using Ultralytics YOLO in [Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox), and apply the model to make predictions.

## 1. Toolbox installation and setup

- Instructions for installing and running Toolbox can be found in the documentation linked [here](https://www.dropbox.com/scl/fi/ut3k3invqvyhyyj168332/Toolbox_installation.docx?rlkey=qob1tuoqmnaljc87blidohpx3&dl=0).

## 2. Prepare training and testing images

- Ensure images are color corrected and a quality desired for analysis.
- Split images into training and testing folder. Our technique is:
  - Training folder: Move 2 out of every 3 images here.
  - Testing folder: Move 1 out of every 3 images here.

## 3. Load labelset

- Open Toolbox and import your classification labelset
  - Go to Labelset → Import
  - Our JSON labelset can be found [here](https://www.dropbox.com/scl/fi/7rbkh40zzj7xbjx4nydoc/labelset_31.json?rlkey=7curccvmqin4ia1xqazum4h3m&dl=0). You can also preview the labelset in Excel [here](https://www.dropbox.com/scl/fi/o2oxc0fen94m5o8x5a5el/percent_cover_labelset.xlsx?rlkey=kh8dlx9fpo9pz5wxnn8eaq5e4&dl=0).

## 4. Import and annotate training images

- Import training images into Toolbox
  - File → Import → Rasters → Images
- Create image patches (classification annotations)
  1. Select a label in the lower label window.
  2. Choose the image patch tool (rectangle icon) from the toolbar on the left.
  3. In the annotation window (center window), left click the appropriate location in the image to add a patch for that label.
  4. Repeat for each label across your training images.

## 5. Export classification dataset

- After annotating, export the dataset:
  - File → Export → Dataset → Classify
- Toolbox will generate a dataset directory containing train, validation, and test folders with labeled image patches.

## 6. Train classification model

- Start training a YOLO classification model
  - Ultralytics → Train Model → Classify
- In the training window:
  - Dataset: Click Browse and select the exported dataset folder.
  - Model Selection: Choose the Ultralytics YOLO model that fits your needs (e.g., YOLOv8 or YOLOv11). A guide comparing model options is available [here](https://docs.ultralytics.com/compare/yolov8-vs-yolo11/).
  - Parameters:
    - Set the location where you want your trained model to be saved.
    - You can use default training parameters or customize them. More information about the parameters can be found [here](https://docs.ultralytics.com/modes/train/#train-settings).
- Click OK to begin training. You can monitor training progress in the terminal.

## 7. Load and deploy model

- After training completes:
  - Go to Ultralytics → Deploy Model → Classify
  - Under Actions, click Browse Model and select your trained weights file (best.pt).
  - Click load model

## 8. Test the model on new images

- Remove training images:
  - In the image window, click "Select All", right-click and select delete all images to remove.
- Import test images:
  - File → Import → Rasters → Images and choose the testing folder.
- Create random image patches:
  - Click Sample at the top
  - Set your desired sampling configuration (e.g., number of patches).
  - Set Select Label to Review
  - Check "Apply to all images" and click Accept

## 9. Run predictions

- To predict label for the new image patches:
  - For a single image: press Ctrl + 1
  - For all images:
    - Ultralytics → Batch Inference → Classify
    - Check "Apply to all images" and "Predict review annotation"

## 10. Review and correct predictions

- Predicted labels appear for each review image patch.
- Confidence levels are shown in the Confidence window.
- To fix incorrect predictions:
  - Select the image patch so that it is shown in the Confidence window
  - Select the correct label in the label window

## 11. Export and improve dataset

- To analyze results:
  - Export annotation file as a .csv file: File → Export → CSV
- To improve your model:
  - Create a new dataset with the corrected predictions
  - Merge this with the original dataset:
    - Ultralytics → Merge Datasets → Classify
    - Set a name and location for the merged dataset
    - Click → Add Dataset and select the datasets you want to combine

## 12. Improve existing model

- The merged dataset can now be used to train a new model following Step 6.
- Instead of selecting a new YOLO model, you can use your existing model:
  - Under Model Selection switch to Use Existing Model
  - Browse to the model weights (best.pt)
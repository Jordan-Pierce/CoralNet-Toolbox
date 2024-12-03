import time

import numpy as np
import matplotlib.pyplot as plt

import torch

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from mobile_sam import sam_model_registry as mobile_sam_model_registry
from mobile_sam import SamPredictor as MobileSamPredictor

from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.models.sam.amg import batched_mask_to_box


def custom_postprocess(mask, score, logit, img_tensor, orig_img):
    """
    Post-processes SAM's inference outputs to generate object detection masks and bounding boxes.

    Args:
        mask (torch.Tensor): Predicted masks with shape (1, 1, H, W).
        score (torch.Tensor): Confidence scores for each mask with shape (1, 1).
        logit (torch.Tensor): Logits for each mask with shape (1, 1, H, W).
        img_tensor (torch.Tensor): The processed input image tensor with shape (1, C, H, W).
        orig_img (np.ndarray): The original, unprocessed image.

    Returns:
        (Results): Results object containing detection masks, bounding boxes, and other metadata.
    """
    # Ensure the original image is in the correct format
    if not isinstance(orig_img, np.ndarray):
        orig_img = orig_img.cpu().numpy()

    # Ensure mask has the correct shape (1, 1, H, W)
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask to have shape (1, 1, H, W), but got {mask.shape}")

    # Scale masks to the original image size
    scaled_masks = ops.scale_masks(mask.float(), orig_img.shape[:2], padding=False)[0]
    scaled_masks = scaled_masks > 0.5  # Apply threshold to masks

    # Generate bounding boxes from masks using batched_mask_to_box
    pred_bboxes = batched_mask_to_box(scaled_masks)

    # Ensure score and cls have the correct shape
    score = score.squeeze(1)  # Remove the extra dimension
    cls = torch.arange(len(mask), dtype=torch.int32, device=mask.device)

    # Combine bounding boxes, scores, and class labels
    pred_bboxes = torch.cat([pred_bboxes, score[:, None], cls[:, None]], dim=-1)

    # Create names dictionary (placeholder for consistency)
    names = dict(enumerate(str(i) for i in range(len(mask))))

    # Create Results object
    result = Results(orig_img, path=None, names=names, masks=scaled_masks, boxes=pred_bboxes)

    return result


def mobile_sam_time_test(image, num_points=10):
    """
    A time test of mobile same on an image, randomly sampling points and predicting the mask.
    :return:
    """
    # Create SAMPredictor
    t0 = time.perf_counter()
    sam_model = mobile_sam_model_registry['vit_t'](checkpoint="../mobile_sam.pt")
    sam_model.to(device='cpu')
    sam_model.eval()

    predictor = MobileSamPredictor(sam_model)
    print(f"Time to load model: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    # Set image as numpy
    predictor.set_image(image)
    # Record the image as tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    print(f"Time to set: {time.perf_counter() - t1:.2f}s")

    for i in range(num_points):
        t2 = time.perf_counter()
        random_x = np.random.randint(0, image.shape[1])
        random_y = np.random.randint(0, image.shape[0])

        points = np.array([[random_x, random_y], [random_x + 5, random_y + 5], [random_x - 5, random_y - 5]])
        labels = np.array([1, 1, 1])  # Assuming all points are foreground for this example

        input_labels = torch.tensor(labels).to('cpu').unsqueeze(0)
        input_points = torch.as_tensor(points.astype(int), dtype=torch.int64).to('cpu').unsqueeze(0)
        transformed_points = predictor.transform.apply_coords_torch(input_points, image.shape[:2])

        mask, score, logit = predictor.predict_torch(point_coords=transformed_points,
                                                     point_labels=input_labels,
                                                     multimask_output=False)

        # Run the prompt inference method
        results = custom_postprocess(mask, score, logit, image_tensor, image)

        if results:
            for mask in results.masks:
                mask = mask.data.cpu().numpy().squeeze().astype(np.uint8)
                plt.imshow(mask);
                plt.show()

        print(f"Time to predict: {time.perf_counter() - t2:.2f}s")

    # Reset image
    predictor.reset_image()


def sam_time_test(image, num_points=10):
    """
    A time test of mobile same on an image, randomly sampling points and predicting the mask.
    :return:
    """
    # Create SAMPredictor
    t0 = time.perf_counter()
    sam_model = sam_model_registry['vit_b'](checkpoint="../sam_b.pt")
    sam_model.to(device='cpu')
    sam_model.eval()

    predictor = SamPredictor(sam_model)
    print(f"Time to load model: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    # Set image as numpy
    predictor.set_image(image)
    # Record the image as tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    print(f"Time to set: {time.perf_counter() - t1:.2f}s")

    for i in range(num_points):
        t2 = time.perf_counter()
        random_x = np.random.randint(0, image.shape[1])
        random_y = np.random.randint(0, image.shape[0])

        points = np.array([[random_x, random_y], [random_x + 5, random_y + 5], [random_x - 5, random_y - 5]])
        labels = np.array([1, 1, 1])  # Assuming all points are foreground for this example

        input_labels = torch.tensor(labels).to('cpu').unsqueeze(0)
        input_points = torch.as_tensor(points.astype(int), dtype=torch.int64).to('cpu').unsqueeze(0)
        transformed_points = predictor.transform.apply_coords_torch(input_points, image.shape[:2])

        mask, score, logit = predictor.predict_torch(point_coords=transformed_points,
                                                     point_labels=input_labels,
                                                     multimask_output=False)

        # Run the prompt inference method
        results = custom_postprocess(mask, score, logit, image_tensor, image)

        if results:
            for mask in results.masks:
                mask = mask.data.cpu().numpy().squeeze().astype(np.uint8)
                plt.imshow(mask);
                plt.show()

        print(f"Time to predict: {time.perf_counter() - t2:.2f}s")

    # Reset image
    predictor.reset_image()


path = rf"C:\Users\jordan.pierce\Documents\GitHub\CoralNet-Toolbox\Data\Sample\D3\T_S04856.jpg"
image = plt.imread(path)

# Run time tests
mobile_sam_time_test(image)
# sam_time_test(image)

print("Done")
import os.path
import argparse
import traceback
from tqdm import tqdm

import cv2
import numpy as np

import torch
import supervision as sv
from ultralytics import SAM
from ultralytics import YOLO
from ultralytics import RTDETR


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def calculate_slice_parameters(width: int, height: int, slices_x: int = 2, slices_y: int = 2, overlap: float = 0.25):
    """
    Calculate slice parameters for video frames; defaults to 2x2, 10% overlap

    :param width:
    :param height:
    :param slices_x:
    :param slices_y:
    :param overlap:
    :return:
    """
    slice_width = width // slices_x
    slice_height = height // slices_y
    overlap_width = int(slice_width * overlap)
    overlap_height = int(slice_height * overlap)
    adjusted_slice_width = slice_width + overlap_width
    adjusted_slice_height = slice_height + overlap_height
    overlap_ratio_w = overlap_width / adjusted_slice_width
    overlap_ratio_h = overlap_height / adjusted_slice_height

    return (adjusted_slice_width, adjusted_slice_height), (overlap_ratio_w, overlap_ratio_h)


def mask_image(image, masks):
    """

    :param image:
    :param masks:
    :return:
    """
    masked_image = image.copy()

    if masks is not None:
        for mask in masks:
            masked_image[mask.astype(bool)] = 0

    return masked_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoInferencer:
    def __init__(self, weights_path: str, model_type: str, video_path: str, output_dir: str,
                 start_at: int, end_at: int, conf: float, iou: float,
                 track: bool, segment: bool, sahi: bool, show: bool):
        """

        :param weights_path:
        :param model_type:
        :param video_path:
        :param output_dir:
        :param start_at:
        :param end_at:
        :param conf:
        :param iou:
        :param track:
        :param sahi:
        :param show:
        """
        self.weights_path = weights_path
        self.model_type = model_type
        self.video_path = video_path
        self.output_dir = output_dir
        self.start_at = start_at
        self.end_at = end_at
        self.conf = conf
        self.iou = iou
        self.track = track
        self.segment = segment
        self.sahi = sahi
        self.show = show

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"NOTE: Device set as {self.device}")

        self.yolo_model = None
        self.sam_model = None
        self.box_annotator = None
        self.mask_annotator = None
        self.labeler = None
        self.slicer = None

    def load_models(self):
        """
        Loads the models

        :return:
        """
        try:

            if "yolo" in self.model_type.lower():
                self.yolo_model = YOLO(self.weights_path)
            elif "rtdetr" in self.model_type.lower():
                self.yolo_model = RTDETR(self.weights_path)
            else:
                raise ValueError(f"Model type {self.model_type} not supported")

            self.mask_annotator = sv.PolygonAnnotator()
            self.box_annotator = sv.BoxAnnotator()
            self.labeler = sv.LabelAnnotator()

            if self.segment:
                try:
                    self.sam_model = SAM('mobile_sam.pt')
                except Exception as e:
                    raise Exception(f"ERROR: Could not load SAM model!\n{e}")

        except Exception as e:
            raise Exception(f"ERROR: Could not load model!\n{e}")

    def setup_slicer(self, frame):
        """
        Creates the SAHI slicer to be used within slicer callback;
        uses the video dimensions to determine slicer parameters.

        :param frame:
        :return:
        """
        slice_wh, overlap_ratio_wh = calculate_slice_parameters(frame.shape[1], frame.shape[0])

        self.slicer = sv.InferenceSlicer(callback=self.slicer_callback,
                                         slice_wh=slice_wh,
                                         iou_threshold=0.90,
                                         overlap_ratio_wh=overlap_ratio_wh,
                                         overlap_filter_strategy=sv.OverlapFilter.NONE)

    @torch.no_grad()
    def slicer_callback(self, image_slice: np.ndarray):
        """
        Prepares a callback to be used with SAHI

        :param image_slice:
        :return:
        """
        results = self.yolo_model(image_slice,
                                  iou=0.50,
                                  conf=0.50)[0]

        results = sv.Detections.from_ultralytics(results)

        return results

    def apply_sahi(self, frame, detections):
        """
        Performs SAHI on a masked frame, where masked regions are areas
        that have already been detected / segmented by initial inference.

        :param frame:
        :param detections:
        :return:
        """
        if self.slicer is None:
            self.setup_slicer(frame)

        if detections:
            # Mask out the frame where previous detections were
            masked_frame = mask_image(frame, detections.mask)
            # Make predictions on the masked frame
            sahi_detections = self.slicer(masked_frame).with_nmm(0.1, class_agnostic=True)

            if self.segment:
                # Get SAM masks for the sahi detections (original frame)
                sahi_detections = self.apply_sam(masked_frame, sahi_detections)

            if sahi_detections:
                # If any sahi detections, merge
                detections = sv.Detections.merge([detections, sahi_detections])

        return detections

    def apply_sam(self, frame, detections):
        """

        :param frame:
        :param detections:
        :return:
        """
        if detections:
            # Pass bboxes to SAM, store masks in detections
            bboxes = detections.xyxy
            masks = self.sam_model(frame, bboxes=bboxes)[0]
            masks = masks.masks.data.cpu().numpy()
            detections.mask = masks.astype(np.uint8)

        return detections

    def inference(self):
        """
        Runs inference on the video, range of frames specified.

        :return:
        """
        os.makedirs(self.output_dir, exist_ok=True)
        target_video_path = f"{self.output_dir}/{os.path.basename(self.video_path)}"
        frame_generator = sv.get_video_frames_generator(source_path=self.video_path)
        video_info = sv.VideoInfo.from_video_path(video_path=self.video_path)

        self.load_models()

        if self.start_at <= 0:
            self.start_at = 0
        if self.end_at <= -1:
            self.end_at = video_info.total_frames

        f_idx = 0
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):

                if self.start_at < f_idx < self.end_at:
                    
                    if self.track:
                        # Use Ultralytics built-in tracking
                        detections = self.yolo_model.track(frame, 
                                                           persist=True, 
                                                           iou=self.iou, 
                                                           conf=self.conf, 
                                                           tracker='botsort.yaml',  # 'bytetrack',
                                                           device=self.device)[0]
                    else:
                        # Use Ultralytics built-in detection
                        detections = self.yolo_model(frame, 
                                                     iou=self.iou, 
                                                     conf=self.conf, 
                                                     device=self.device)[0]
                    
                    detections = sv.Detections.from_ultralytics(detections)

                    # Perform segmentations
                    if self.segment:
                        detections = self.apply_sam(frame, detections)

                    # Perform smaller detections / segmentations
                    if self.sahi:
                        detections = self.apply_sahi(frame, detections)
                        # Do NMM / NMS with all detections (bboxes)
                        detections = detections.with_nms(0.1, class_agnostic=True)

                    # Prepare the labels
                    class_names = detections.data['class_name'].tolist()
                    confidences = detections.confidence.tolist()
                    labels = [f"{name} {conf:0.2f}" for name, conf in list(zip(class_names, confidences))]

                    # Display frame, boxes, masks, labels
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    frame = self.mask_annotator.annotate(scene=frame, detections=detections)
                    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                    sink.write_frame(frame=frame)

                    if self.show:
                        try:
                            # frame = cv2.resize(frame, (640, 360))
                            cv2.imshow('Predictions', frame)
                        except Exception as e:
                            print(f"cv2.imshow failed: {e}")
                        try:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except Exception as e:
                            print(f"cv2.waitKey failed: {e}")

                f_idx += 1
                if f_idx >= self.end_at:
                    try:
                        cv2.destroyAllWindows()
                    except Exception as e:
                        print(f"cv2.destroyAllWindows failed: {e}")
                    break


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Video Inferencing with YOLO, SAM, and ByteTrack")

    parser.add_argument("--weights_path", required=True, type=str,
                        help="Path to the source weights file")

    parser.add_argument("--model_type", required=True, type=str,
                        help="Model architecture; either YOLO or RTDETR")

    parser.add_argument("--video_path", required=True, type=str,
                        help="Path to the source video file")

    parser.add_argument("--output_dir", required=True, type=str,
                        help="Path to the target video directory (output)")

    parser.add_argument("--start_at", default=0, type=int,
                        help="Frame to start inference")

    parser.add_argument("--end_at", default=-1, type=int,
                        help="Frame to end inference")

    parser.add_argument("--conf", default=0.20, type=float,
                        help="Confidence threshold for the model")

    parser.add_argument("--iou", default=0.30, type=float,
                        help="IOU threshold for the model")

    parser.add_argument("--track", action='store_true',
                        help="Uses ByteTrack to track detections")

    parser.add_argument("--segment", action='store_true',
                        help="Uses SAM to create masks on detections (takes longer)")

    parser.add_argument("--sahi", action='store_true',
                        help="Uses SAHI to find smaller objects (takes longer)")

    parser.add_argument("--show", action='store_true',
                        help="Display the inference video")

    args = parser.parse_args()

    try:
        inferencer = VideoInferencer(
            weights_path=args.weights_path,
            model_type=args.model_type,
            video_path=args.video_path,
            output_dir=args.output_dir,
            start_at=args.start_at,
            end_at=args.end_at,
            conf=args.conf,
            iou=args.iou,
            track=args.track,
            segment=args.segment,
            sahi=args.sahi,
            show=args.show
        )
        inferencer.inference()
        print("Done.")
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

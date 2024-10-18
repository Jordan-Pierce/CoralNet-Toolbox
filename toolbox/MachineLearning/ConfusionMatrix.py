import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConfusionMatrixMetrics:
    """
    A class for calculating TP, FP, TN, FN, precision, recall, accuracy,
    and per-class accuracy from a confusion matrix.

    Attributes:
        matrix (np.ndarray): The confusion matrix.
        num_classes (int): The number of classes.
    """

    def __init__(self, matrix, class_mapping):
        """
        Initialize the ConfusionMatrixMetrics with a given confusion matrix.

        Args:
            matrix (np.ndarray): The confusion matrix.
        """
        self.matrix = matrix
        self.num_classes = matrix.shape[0]
        self.class_mapping = class_mapping

    def calculate_tp(self):
        """
        Calculate true positives for each class.

        Returns:
            np.ndarray: An array of true positives for each class.
        """
        return np.diagonal(self.matrix)

    def calculate_fp(self):
        """
        Calculate false positives for each class.

        Returns:
            np.ndarray: An array of false positives for each class.
        """
        return self.matrix.sum(axis=0) - np.diagonal(self.matrix)

    def calculate_fn(self):
        """
        Calculate false negatives for each class.

        Returns:
            np.ndarray: An array of false negatives for each class.
        """
        return self.matrix.sum(axis=1) - np.diagonal(self.matrix)

    def calculate_tn(self):
        """
        Calculate true negatives for each class.

        Returns:
            np.ndarray: An array of true negatives for each class.
        """
        total = self.matrix.sum()
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        fn = self.calculate_fn()
        return total - (tp + fp + fn)

    def calculate_precision(self):
        """
        Calculate precision for each class.

        Returns:
            np.ndarray: An array of precision values for each class.
        """
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        return tp / (tp + fp + 1e-16)  # avoid division by zero

    def calculate_recall(self):
        """
        Calculate recall for each class.

        Returns:
            np.ndarray: An array of recall values for each class.
        """
        tp = self.calculate_tp()
        fn = self.calculate_fn()
        return tp / (tp + fn + 1e-16)  # avoid division by zero

    def calculate_accuracy(self):
        """
        Calculate accuracy for all classes combined.

        Returns:
            float: The accuracy value.
        """
        tp = self.calculate_tp().sum()
        total = self.matrix.sum()
        return tp / total

    def calculate_per_class_accuracy(self):
        """
        Calculate per-class accuracy.

        Returns:
            np.ndarray: An array of accuracy values for each class.
        """
        tp = self.calculate_tp()
        total_per_class = self.matrix.sum(axis=1)
        return tp / (total_per_class + 1e-16)  # avoid division by zero

    def get_metrics_all(self):
        """
        Get all metrics (TP, FP, TN, FN, precision, recall, accuracy) for all classes combined.

        Returns:
            dict: A dictionary containing all calculated metrics for all classes combined.
        """
        tp = self.calculate_tp().sum()
        fp = self.calculate_fp().sum()
        tn = self.calculate_tn().sum()
        fn = self.calculate_fn().sum()
        precision = tp / (tp + fp + 1e-16)  # avoid division by zero
        recall = tp / (tp + fn + 1e-16)  # avoid division by zero
        accuracy = self.calculate_accuracy()

        return {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy
        }

    def get_metrics_per_class(self):
        """
        Get all metrics (TP, FP, TN, FN, precision, recall, accuracy)
        per class in a dictionary.

        Returns:
            dict: A dictionary containing all calculated metrics per class.
        """
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        tn = self.calculate_tn()
        fn = self.calculate_fn()
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        accuracy = self.calculate_per_class_accuracy()

        metrics_per_class = {}
        for i in range(self.num_classes):
            try:
                class_name = self.class_mapping[i]
                metrics_per_class[class_name] = {
                    'TP': tp[i],
                    'FP': fp[i],
                    'TN': tn[i],
                    'FN': fn[i],
                    'Precision': precision[i],
                    'Recall': recall[i],
                    'Accuracy': accuracy[i]
                }
            except KeyError:
                print("Warning: Class mapping not found for class index", i)

        return metrics_per_class

    def save_metrics_to_json(self, directory, filename="metrics.json"):
        """
        Save the metrics to a JSON file.

        Args:
            directory (str): The directory where the JSON file will be saved.
            filename (str): The name of the JSON file. Default is "metrics.json".
        """
        os.makedirs(directory, exist_ok=True)

        metrics_all = self.get_metrics_all()
        metrics_per_class = self.get_metrics_per_class()

        results = {
            'All Classes': metrics_all,
            'Per Class': metrics_per_class
        }

        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os
import ujson as json
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConfusionMatrixMetrics:
    """
    A comprehensive class for calculating and visualizing confusion matrix metrics.

    Attributes:
        top1 (float): Top-1 accuracy.
        top5 (float): Top-5 accuracy.
        matrix (np.ndarray): The confusion matrix.
        num_classes (int): The number of classes.
        class_mapping (dict): Mapping of class indices to class names.
        total_predictions (int): Total number of predictions.
        class_distributions (np.ndarray): Distribution of samples across classes.
    """

    def __init__(self, results, class_mapping):
        """
        Initialize the ConfusionMatrixMetrics class.

        Args:
            results: An object containing confusion matrix and accuracy metrics.
            class_mapping (dict): Mapping of class indices to class names.
        """
        self.matrix = results.confusion_matrix.matrix
        self.num_classes = self.matrix.shape[0]
        self.class_mapping = class_mapping

        # Get the top-1 and top-5 accuracies, if available
        self.top1 = results.top1 if hasattr(results, 'top1') else None
        self.top5 = results.top5 if hasattr(results, 'top5') else None

        # Add background class if needed
        if len(self.class_mapping) + 1 == self.num_classes:
            self.class_mapping[self.num_classes] = 'background'

        # Calculate total predictions and class distributions
        self.total_predictions = np.sum(self.matrix)
        self.class_distributions = np.sum(self.matrix, axis=1) / self.total_predictions

    def calculate_tp(self):
        """Calculate true positives for each class."""
        return np.diagonal(self.matrix)

    def calculate_fp(self):
        """Calculate false positives for each class."""
        return self.matrix.sum(axis=0) - np.diagonal(self.matrix)

    def calculate_fn(self):
        """Calculate false negatives for each class."""
        return self.matrix.sum(axis=1) - np.diagonal(self.matrix)

    def calculate_tn(self):
        """Calculate true negatives for each class."""
        total = self.matrix.sum()
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        fn = self.calculate_fn()
        return total - (tp + fp + fn)

    def calculate_precision(self):
        """Calculate precision for each class."""
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        return tp / (tp + fp + 1e-16)

    def calculate_recall(self):
        """Calculate recall for each class."""
        tp = self.calculate_tp()
        fn = self.calculate_fn()
        return tp / (tp + fn + 1e-16)

    def calculate_f1_score(self):
        """Calculate F1 score for each class."""
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        return 2 * (precision * recall) / (precision + recall + 1e-16)

    def calculate_specificity(self):
        """Calculate specificity (true negative rate) for each class."""
        tn = self.calculate_tn()
        fp = self.calculate_fp()
        return tn / (tn + fp + 1e-16)

    def calculate_accuracy(self):
        """Calculate overall accuracy."""
        tp = self.calculate_tp().sum()
        total = self.matrix.sum()
        return tp / total

    def calculate_per_class_accuracy(self):
        """Calculate accuracy for each class."""
        tp = self.calculate_tp()
        total_per_class = self.matrix.sum(axis=1)
        return tp / (total_per_class + 1e-16)

    def calculate_balanced_accuracy(self):
        """Calculate balanced accuracy for each class."""
        return (self.calculate_recall() + self.calculate_specificity()) / 2

    def calculate_metrics_summary(self):
        """
        Calculate a comprehensive summary of metrics.

        Returns:
            dict: Dictionary containing summary metrics.
        """
        metrics = {
            'Accuracy': self.calculate_accuracy(),
            'Balanced Accuracy': np.mean(self.calculate_balanced_accuracy()),
            'Top1 Accuracy': self.top1,
            'Top5 Accuracy': self.top5,
            'Macro Precision': np.mean(self.calculate_precision()),
            'Macro Recall': np.mean(self.calculate_recall()),
            'Macro F1': np.mean(self.calculate_f1_score()),
            'Weighted F1': np.average(self.calculate_f1_score(), weights=self.class_distributions),
        }
        return metrics

    def get_metrics_all(self):
        """
        Get all metrics for all classes combined.

        Returns:
            dict: Dictionary containing combined metrics.
        """
        tp = self.calculate_tp().sum()
        fp = self.calculate_fp().sum()
        tn = self.calculate_tn().sum()
        fn = self.calculate_fn().sum()

        metrics = {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Precision': tp / (tp + fp + 1e-16),
            'Recall': tp / (tp + fn + 1e-16),
            'Accuracy': self.calculate_accuracy(),
            'F1 Score': np.mean(self.calculate_f1_score()),
            'Balanced Accuracy': np.mean(self.calculate_balanced_accuracy()),
        }
        return metrics

    def get_metrics_per_class(self):
        """
        Get detailed metrics for each class.

        Returns:
            dict: Dictionary containing per-class metrics.
        """
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        tn = self.calculate_tn()
        fn = self.calculate_fn()
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1_score()
        specificity = self.calculate_specificity()
        balanced_acc = self.calculate_balanced_accuracy()

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
                    'F1 Score': f1[i],
                    'Specificity': specificity[i],
                    'Balanced Accuracy': balanced_acc[i],
                    'Support': np.sum(self.matrix[i]),
                    'Distribution (%)': self.class_distributions[i] * 100
                }
            except KeyError:
                print(f"Warning: Class mapping not found for class index {i}")

        return metrics_per_class

    def save_confusion_matrix_png(self, directory, filename="confusion_matrix_toolbox.png", normalized=False):
        """
        Save the confusion matrix as a PNG image.

        Args:
            directory (str): The directory where the PNG file will be saved.
            filename (str): The name of the PNG file. Default is "confusion_matrix.png".
            normalized (bool): Whether to normalize the confusion matrix. Default is False.
        """
        os.makedirs(directory, exist_ok=True)

        if normalized:
            cm = self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
        else:
            cm = self.matrix.astype(int)  # Ensure the matrix is integer type
            title = "Confusion Matrix"

        # Dynamically adjust figure size based on number of classes but maintain square shape
        figsize = (8 + (self.num_classes // 2), 8 + (self.num_classes // 2))
        plt.figure(figsize=figsize)

        def format_value(x):
            """Format values with K, M suffix for thousands and millions"""
            if not normalized:
                if x >= 1e6:
                    return f'{x / 1e6:.1f}M'
                elif x >= 1000:
                    return f'{x / 1000:.1f}K'
                else:
                    return f'{int(x)}'
            else:
                return f'{x:.2f}'  # 2 decimal places for normalized values

        # Create annotation array
        annot = np.zeros_like(cm, dtype='<U10')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = format_value(cm[i, j])

        # Create the heatmap
        ax = sns.heatmap(cm,
                         annot=annot,
                         fmt='',  # Empty format string as we're using custom annotations
                         cmap='Blues',
                         xticklabels=self.class_mapping.values(),
                         yticklabels=self.class_mapping.values(),
                         square=True,
                         cbar_kws={'format': FuncFormatter(lambda x, p: format_value(x))})

        # Highlight the diagonal squares with green perimeters
        for i in range(self.num_classes):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='lightblue', lw=2))

        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        file_path = os.path.join(directory, filename)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

    def save_normalized_confusion_matrix_png(self, directory, filename="confusion_matrix_normalized_toolbox.png"):
        """
        Save the normalized confusion matrix as a PNG image.

        Args:
            directory (str): The directory where the PNG file will be saved.
            filename (str): The name of the PNG file. Default is
        """
        self.save_confusion_matrix_png(directory, filename, normalized=True)

    def save_real_confusion_matrix_png(self, directory, filename="confusion_matrix_toolbox.png"):
        """
        Save the real-valued confusion matrix as a PNG image.

        Args:
            directory (str): The directory where the PNG file will be saved.
            filename (str): The name of the PNG file.
        """
        self.save_confusion_matrix_png(directory, filename, normalized=False)

    def save_metrics_report(self, directory, filename="metrics_report.csv"):
        """
        Save a detailed metrics report as CSV.

        Args:
            directory (str): Output directory
            filename (str): Output filename
        """
        metrics_dict = {
            'Class': list(self.class_mapping.values()),
            'Total Samples': np.sum(self.matrix, axis=1),
            'True Positives': self.calculate_tp(),
            'False Positives': self.calculate_fp(),
            'False Negatives': self.calculate_fn(),
            'True Negatives': self.calculate_tn(),
            'Precision': self.calculate_precision(),
            'Recall': self.calculate_recall(),
            'F1 Score': self.calculate_f1_score(),
            'Specificity': self.calculate_specificity(),
            'Balanced Accuracy': self.calculate_balanced_accuracy(),
            'Class Distribution (%)': self.class_distributions * 100,
        }

        df = pd.DataFrame(metrics_dict)
        df.to_csv(os.path.join(directory, filename), index=False, float_format='%.4f')

    def save_results(self, directory):
        """
        Save comprehensive results including metrics, visualizations, and optional report.

        Args:
            directory (str): Output directory
        """
        # Create output directory
        os.makedirs(directory, exist_ok=True)

        # Save metrics as JSON
        metrics_all = self.get_metrics_all()
        metrics_per_class = self.get_metrics_per_class()
        summary_metrics = self.calculate_metrics_summary()

        results = {
            'Summary Metrics': summary_metrics,
            'All Classes': metrics_all,
            'Per Class': metrics_per_class
        }

        # Dump results to JSON file
        with open(os.path.join(directory, "metrics.json"), 'w') as f:
            json.dump(results, f, indent=4)

        # Save detailed report if requested
        self.save_metrics_report(directory)

        # Save confusion matrices (both normalized and real versions)
        self.save_normalized_confusion_matrix_png(directory)
        self.save_real_confusion_matrix_png(directory)

from multiprocessing.pool import ThreadPool

import numpy as np

from ultralytics.utils import TQDM
from ultralytics.utils import NUM_THREADS
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.dataset import SemanticDataset
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.dataset import PolygonSemanticDataset


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def cumulative_probabilities(weights):
    """
    Turn per-sample weights into cumulative sampling probabilities.

    A draw is then a binary search, O(log N). np.random.choice re-validates the whole
    probability vector on every call, O(N), which made each epoch O(N^2).

    Args:
        weights (list): A non-negative weight per sample.

    Returns:
        numpy array or None: Cumulative probabilities ending at exactly 1, or None when no
            sample has any weight (sampling then falls back to the dataloader's own order).
    """
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if weights.size == 0 or total <= 0:
        return None

    cumulative = np.cumsum(weights / total)
    # Rounding can leave the last entry just under 1, letting a draw run off the end
    cumulative[-1] = 1.0
    return cumulative


def sample_index(cumulative):
    """
    Draw one sample index from cumulative sampling probabilities.

    side="right" never lands on a zero-weight sample: its cumulative value equals the one
    before it, so the search steps past it.

    Args:
        cumulative (numpy array): Cumulative probabilities from cumulative_probabilities().

    Returns:
        int: The sampled index.
    """
    return int(np.searchsorted(cumulative, np.random.random(), side="right"))


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class WeightedClassificationDataset(ClassificationDataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedClassificationDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """
        super(WeightedClassificationDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.probabilities = None

        # Weights only drive sampling on the training split
        if not self.train_mode:
            return

        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Class indices follow the model's names, not this split's folders, so a class with no
        folder here can still index past len(self.base.classes); bincount sizes to fit.

        Returns:
            dict: A dict containing the counts for each class.
        """
        desc = f"{self.prefix}Weighted sampling: counting classes"
        class_ids = [class_idx for _, class_idx, _, _ in TQDM(self.samples, desc=desc)]
        self.counts = np.bincount(np.array(class_ids, dtype=int), minlength=len(self.base.classes))
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for _, class_idx, _, _ in self.samples:
            weight = self.agg_func(self.class_weights[class_idx])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the cumulative sampling probabilities based on the weights.

        Returns:
            numpy array or None: Cumulative sampling probabilities, one per label.
        """
        return cumulative_probabilities(self.weights)

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        if self.train_mode and self.probabilities is not None:
            index = sample_index(self.probabilities)

        return super(WeightedClassificationDataset, self).__getitem__(index)


class WeightedInstanceDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """
        super(WeightedInstanceDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.probabilities = None

        # Weights only drive sampling on the training split
        if not self.train_mode:
            return

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in TQDM(self.labels, desc=f"{self.prefix}Weighted sampling: counting instances"):
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the cumulative sampling probabilities based on the weights.

        Returns:
            numpy array or None: Cumulative sampling probabilities, one per label.
        """
        return cumulative_probabilities(self.weights)

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if self.train_mode and self.probabilities is not None:
            index = sample_index(self.probabilities)

        return self.transforms(self.get_image_and_label(index))


class WeightedSemanticDataset(SemanticDataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedSemanticDataset.

        A mask has no instances to count, so a class counts once for each training mask it
        appears in, and each mask is weighted by the classes present in it.
        """
        super(WeightedSemanticDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.probabilities = None

        # Masks are only read to weight the training split
        if not self.train_mode:
            return

        # Fraction of a mask a class must cover to count as present; skips slivers at tile edges
        self.min_fraction = 0.001

        self.present_classes = self.find_present_classes()
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def num_mask_classes(self):
        """
        Number of pixel values that are classes. A binary (nc == 1) mask still holds two:
        0 for background and 1 for the foreground.

        Returns:
            int: The number of class values a mask can hold, excluding the 255 ignore label.
        """
        return max(int(self.data.get("nc", 0)), 2)

    def mask_classes(self, index):
        """
        Find the classes present in one training mask, read the same way training reads it.

        Returns:
            numpy array: The class ids covering at least min_fraction of the mask.
        """
        shape = self.labels[index].get("shape")
        mask = self.load_mask(index, image_shape=tuple(shape) if shape is not None else None)
        if self.include_class is not None:
            mask[~np.isin(mask, self.include_class)] = 255

        pixels = np.bincount(mask.ravel(), minlength=256)[:self.num_mask_classes()]
        return np.flatnonzero(pixels >= max(1, self.min_fraction * mask.size))

    def find_present_classes(self):
        """
        Find the classes present in every training mask, reading the masks in parallel.

        Returns:
            list: An array of class ids for each mask.
        """
        desc = f"{self.prefix}Weighted sampling: reading masks"
        with ThreadPool(NUM_THREADS) as pool:
            # imap keeps mask order and yields as masks finish, so the bar can advance
            results = pool.imap(self.mask_classes, range(len(self.labels)))
            return list(TQDM(results, desc=desc, total=len(self.labels)))

    def count_instances(self):
        """
        Count the number of training masks each class appears in

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = np.zeros(self.num_mask_classes(), dtype=int)
        for classes in self.present_classes:
            self.counts[classes] += 1

        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each mask based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each mask.
        """
        weights = []
        for classes in self.present_classes:
            # A mask of nothing but ignore pixels supervises nothing; never draw it
            if classes.size == 0:
                weights.append(0)
                continue

            weight = self.agg_func(self.class_weights[classes])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the cumulative sampling probabilities based on the weights.

        Returns:
            numpy array or None: Cumulative sampling probabilities, one per mask.
        """
        return cumulative_probabilities(self.weights)

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        if self.train_mode and self.probabilities is not None:
            index = sample_index(self.probabilities)

        return super(WeightedSemanticDataset, self).__getitem__(index)


class WeightedPolygonSemanticDataset(WeightedSemanticDataset, PolygonSemanticDataset):
    """
    WeightedSemanticDataset over polygon labels that are rasterized into masks.

    Nothing to add: the method resolution order puts PolygonSemanticDataset's label hooks
    and its rasterizing load_mask ahead of SemanticDataset's, so the presence scan reads
    the same masks training does.
    """

# coralnet_toolbox/Explorer/core/feature_binning.py
"""
Feature bins: group annotations by appearance with a Ward hierarchy.

The hierarchy is built once per feature set and cut at any K for free, which is
what lets the gallery's bin count change without re-clustering. Ward on
L2-normalised, PCA-reduced features matched or beat K-Means on the benchmark
(Coralscapes, 39 reef classes, K=32: purity 0.536 vs 0.535 for the app's old
StandardScaler -> K-Means on DINOv2, 0.468 vs 0.440 on colour), and unlike
K-Means it is deterministic and gives a dendrogram order for the bins, so
neighbouring bins in the gallery are neighbours in feature space.

Exact Ward keeps an n^2/2 distance matrix, so above EXACT_WARD_MAX_ITEMS the
items are first condensed into FAISS K-Means leaves, and Ward runs on the
leaves weighted by their sizes.
"""

import hashlib

import numpy as np

# Exact scipy Ward holds n*(n-1)/2 float64 distances: 3,000 items is ~36 MB.
EXACT_WARD_MAX_ITEMS = 3000
# K-Means leaves that stand in for the items above that size.
LEAF_CLUSTERS = 1000
# Features wider than this are PCA-reduced before clustering, which keeps the
# leaf path fast. 64 cost DINOv2 a point of purity; 128 matched no reduction.
PCA_MAX_DIMS = 128
# Bounds of the automatic bin count. Within them the search runs from
# sqrt(N)/4 to sqrt(N) bins: bins are for labelling, and a cut at the very top
# of the tree (2-6 bins) mixes classes that a human would separate at a glance.
AUTO_K_MIN = 3
AUTO_K_MAX = 40
# A height ratio at least this big picks its K even below that range: real
# annotation sets peaked at 1.4, well-separated groups showed 13 to 27.
DECISIVE_GAP_RATIO = 2.0


def prepare_features(features, metric='cosine', max_dims=PCA_MAX_DIMS, seed=0):
    """Return the float32 matrix the hierarchy clusters on.

    Cosine features (deep embeddings) are L2-normalised so euclidean Ward on
    them ranks like cosine distance. Euclidean features (colour, texture) are
    already scaled per block and are used as they are; a global StandardScaler
    would blow their sparse histogram bins up into noise.
    """
    X = np.asarray(features, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    if max_dims and X.shape[1] > max_dims and X.shape[0] > max_dims:
        from sklearn.decomposition import PCA
        X = PCA(n_components=max_dims, random_state=seed).fit_transform(X)
    return np.ascontiguousarray(X, dtype=np.float32)


def weighted_ward(centroids, weights):
    """Ward linkage over weighted points, as a scipy linkage matrix.

    Each point stands for a cluster of `weights[i]` items. Merging clusters a
    and b raises the within-cluster sum of squares by
    na*nb/(na+nb) * |ca - cb|^2, so two big leaves cost more to merge than two
    singletons the same distance apart. Heights are reported the way scipy's
    'ward' does, sqrt(2 * that increase), so both paths read on one scale.

    Nearest-neighbour chain: Ward is reducible, so chasing nearest neighbours
    until two clusters are each other's nearest finds every merge in
    O(n^2 * d) without a distance matrix.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    n = len(centroids)
    if n < 2:
        return np.empty((0, 4), dtype=np.float64)
    total = 2 * n - 1
    C = np.zeros((total, centroids.shape[1]), dtype=np.float64)
    C[:n] = centroids
    size = np.zeros(total, dtype=np.float64)
    size[:n] = np.asarray(weights, dtype=np.float64)
    active = np.zeros(total, dtype=bool)
    active[:n] = True
    height = np.zeros(total, dtype=np.float64)

    merges = []      # (a, b, height) in internal ids, in the order found
    next_id = n
    chain = []

    def cost(a, idx):
        diff = C[idx] - C[a]
        return size[a] * size[idx] / (size[a] + size[idx]) * np.einsum('ij,ij->i', diff, diff)

    while next_id < total:
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        a = chain[-1]
        idx = np.flatnonzero(active)
        idx = idx[idx != a]
        costs = cost(a, idx)
        b = int(idx[int(np.argmin(costs))])
        best = costs.min()
        # Prefer the previous chain link on a tie, or the chain can cycle.
        if len(chain) > 1:
            prev = chain[-2]
            prev_cost = cost(a, np.array([prev]))[0]
            if prev_cost <= best:
                b, best = prev, prev_cost
        if len(chain) > 1 and b == chain[-2]:
            chain.pop()
            chain.pop()
            m = next_id
            next_id += 1
            na, nb = size[a], size[b]
            C[m] = (na * C[a] + nb * C[b]) / (na + nb)
            size[m] = na + nb
            active[a] = active[b] = False
            active[m] = True
            # Ward is monotone, but rounding can put a parent a hair below its
            # child, and the renumbering below relies on children sorting first.
            height[m] = max(float(np.sqrt(2.0 * max(best, 0.0))), height[a], height[b])
            merges.append((a, b, height[m], m))
        else:
            chain.append(b)

    return _renumber_merges(merges, n)


def _renumber_merges(merges, n):
    """Turn (a, b, height, new_id) merges found in any order into a scipy linkage.

    Chain merges are found out of height order; scipy wants them sorted with
    cluster ids numbered in that order. Replay them through a union-find over
    leaf representatives to renumber.
    """
    rep = np.arange(2 * n - 1)
    for a, _b, _h, m in merges:
        rep[m] = rep[a]
    order = sorted(range(len(merges)), key=lambda i: merges[i][2])
    parent = np.arange(n)
    label = np.arange(n)        # component root -> its current scipy id

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    Z = np.zeros((n - 1, 4), dtype=np.float64)
    for row, i in enumerate(order):
        a, b, h, _m = merges[i]
        ra, rb = find(rep[a]), find(rep[b])
        la, lb = label[ra], label[rb]
        parent[rb] = ra
        label[ra] = n + row
        Z[row] = (min(la, lb), max(la, lb), h, 0)
    # Leaf counts per merge, as scipy's tree validators expect (the weights
    # live only in the heights).
    counts = np.ones(2 * n - 1, dtype=np.float64)
    for row in range(n - 1):
        counts[n + row] = counts[int(Z[row, 0])] + counts[int(Z[row, 1])]
        Z[row, 3] = counts[n + row]
    return Z


def _kmeans_leaves(X, n_leaves, seed=0):
    """Condense X into n_leaves K-Means centroids: (assignment, centroids, counts)."""
    n_leaves = int(min(n_leaves, len(X)))
    try:
        import faiss
        # A few items per leaf is the point here, not a training-set warning.
        km = faiss.Kmeans(X.shape[1], n_leaves, niter=20, seed=seed, verbose=False,
                          min_points_per_centroid=1)
        km.train(np.ascontiguousarray(X, dtype=np.float32))
        _d, assign = km.index.search(np.ascontiguousarray(X, dtype=np.float32), 1)
        assign = assign.ravel().astype(np.int64)
    except Exception:
        from sklearn.cluster import MiniBatchKMeans
        assign = MiniBatchKMeans(n_leaves, random_state=seed, n_init=3).fit_predict(X).astype(np.int64)
    # Drop empty leaves and recompute centroids from the members.
    used, assign = np.unique(assign, return_inverse=True)
    counts = np.bincount(assign, minlength=len(used)).astype(np.float64)
    centroids = np.zeros((len(used), X.shape[1]), dtype=np.float64)
    np.add.at(centroids, assign, X)
    centroids /= counts[:, None]
    return assign, centroids, counts


class FeatureHierarchy:
    """A Ward tree over one feature matrix, cut at any number of bins.

    Bins come out numbered in dendrogram order, so bin i and i+1 are close in
    feature space, and items within a bin can be ordered most-typical first.
    """

    def __init__(self, features, metric='cosine', max_dims=PCA_MAX_DIMS,
                 exact_max_items=EXACT_WARD_MAX_ITEMS, n_leaves=LEAF_CLUSTERS, seed=0):
        self.X = prepare_features(features, metric=metric, max_dims=max_dims, seed=seed)
        self.n_items = len(self.X)
        self._labels_cache = {}
        if self.n_items < 2:
            self.leaf_of_item = np.zeros(self.n_items, dtype=np.int64)
            self.Z = np.empty((0, 4), dtype=np.float64)
            self.n_leaves = self.n_items
            return
        if self.n_items <= exact_max_items:
            from scipy.cluster.hierarchy import linkage
            self.leaf_of_item = np.arange(self.n_items, dtype=np.int64)
            self.Z = linkage(self.X.astype(np.float64), method='ward')
        else:
            # Several items per leaf, or the leaves are just a noisier copy of the items.
            n_leaves = int(min(n_leaves, max(2, self.n_items // 4)))
            assign, centroids, counts = _kmeans_leaves(self.X, n_leaves, seed=seed)
            self.leaf_of_item = assign
            self.Z = weighted_ward(centroids, counts)
        self.n_leaves = int(self.leaf_of_item.max()) + 1 if self.n_items else 0

    @property
    def max_k(self):
        return max(1, self.n_leaves)

    def heights(self):
        """Merge heights, ascending (Ward heights are monotone)."""
        return self.Z[:, 2] if len(self.Z) else np.empty(0)

    def gap_ratios(self, k_min=2, k_max=AUTO_K_MAX):
        """{K: height ratio} of the merge that would join K bins into K-1 over the one before.

        Cutting the tree into K bins undoes its last K-1 merges. A big jump
        between those two merges means the K bins are much further apart than
        anything inside them. Ward heights grow towards the root, so the raw
        gap always favours K=2; the ratio compares like with like.
        """
        h = self.heights()
        n = len(h) + 1
        out = {}
        for k in range(max(2, int(k_min)), int(min(k_max, n - 1)) + 1):
            # Floor the lower merge so exact duplicates (height 0) cannot win.
            out[k] = float(h[n - k] / max(h[n - k - 1], 1e-6 * h[-1], 1e-12))
        return out

    def auto_k(self, k_min=None, k_max=None):
        """Pick the bin count at the largest relative gap in the merge heights.

        A decisive jump (ratio >= DECISIVE_GAP_RATIO) anywhere up to k_max
        wins: well-separated groups show ratios of 10 and more. Otherwise the
        search runs over sqrt(N)/4 .. sqrt(N) bins (clamped to AUTO_K_MIN ..
        AUTO_K_MAX). Real annotation sets peaked at ratios of 1.2-1.4 near the
        root, and taking that peak chose 3 bins at 0.35 purity on Coralscapes,
        against 22 bins at 0.51 within the range.
        """
        root_n = np.sqrt(max(self.n_items, 1))
        if k_min is None:
            k_min = max(AUTO_K_MIN, round(root_n / 4))
        if k_max is None:
            k_max = max(k_min, min(AUTO_K_MAX, round(root_n)))
        if self.n_leaves < 3:
            return max(1, self.n_leaves)
        everywhere = self.gap_ratios(2, k_max)
        if not everywhere:
            return min(int(k_min), self.n_leaves)
        decisive_k = max(everywhere, key=everywhere.get)
        if everywhere[decisive_k] >= DECISIVE_GAP_RATIO:
            return decisive_k
        in_range = {k: r for k, r in everywhere.items() if k >= k_min}
        if not in_range:
            return max(everywhere)
        return max(in_range, key=in_range.get)

    def labels(self, k):
        """Bin index per item, 0..k-1, numbered in dendrogram order."""
        k = int(max(1, min(k, self.max_k)))
        cached = self._labels_cache.get(k)
        if cached is not None:
            return cached
        if self.n_leaves < 2 or k == 1:
            out = np.zeros(self.n_items, dtype=np.int64)
            self._labels_cache[k] = out
            return out
        n = self.n_leaves
        # Union-find over the first n-k merges leaves exactly k components.
        parent = np.arange(2 * n - 1)
        for row in range(n - k):
            a, b = int(self.Z[row, 0]), int(self.Z[row, 1])
            parent[a] = parent[b] = n + row
        root = np.arange(2 * n - 1)
        for node in range(2 * n - 2, -1, -1):
            p = parent[node]
            root[node] = root[p] if p != node else node
        leaf_root = root[:n]
        # Number the bins by first appearance along the dendrogram's leaf order.
        from scipy.cluster.hierarchy import leaves_list
        order = leaves_list(self.Z)
        bin_of_root = {}
        for leaf in order:
            r = int(leaf_root[leaf])
            if r not in bin_of_root:
                bin_of_root[r] = len(bin_of_root)
        leaf_bin = np.array([bin_of_root[int(r)] for r in leaf_root], dtype=np.int64)
        out = leaf_bin[self.leaf_of_item]
        self._labels_cache[k] = out
        return out

    def centroid_distances(self, labels):
        """Each item's euclidean distance to its own bin's centroid (clustering space)."""
        labels = np.asarray(labels, dtype=np.int64)
        if self.n_items == 0:
            return np.empty(0, dtype=np.float32)
        k = int(labels.max()) + 1
        sums = np.zeros((k, self.X.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, self.X)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        centroids = sums / np.maximum(counts, 1)[:, None]
        return np.linalg.norm(self.X - centroids[labels], axis=1).astype(np.float32)

    def ordered_bins(self, k):
        """Item indices per bin, bins in dendrogram order, most typical item first."""
        labels = self.labels(k)
        dists = self.centroid_distances(labels)
        order = np.lexsort((dists, labels))
        sorted_labels = labels[order]
        cuts = np.flatnonzero(np.diff(sorted_labels)) + 1
        return [chunk for chunk in np.split(order, cuts) if chunk.size]


def ids_digest(ids):
    """A short stable fingerprint of an ordered id list (hierarchy cache key)."""
    h = hashlib.sha1(usedforsecurity=False)
    for ann_id in ids:
        h.update(str(ann_id).encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()


class HierarchyCache:
    """Keeps the last few hierarchies so re-cutting or re-sorting is free."""

    def __init__(self, max_entries=4):
        self.max_entries = max_entries
        self._entries = {}

    def get(self, key, build):
        entry = self._entries.pop(key, None)
        if entry is None:
            entry = build()
        self._entries[key] = entry       # most recently used last
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        return entry

    def clear(self):
        self._entries.clear()


def color_tint_from_features(features):
    """Mean RGB (0-255) of a bin from colour features, or None if they are not colour.

    The colour extractors put the H/S/V mean+std moments at dims 64..69, scaled
    by COLOR_MOMENT_WEIGHT; the means are dims 64, 66 and 68.
    """
    from coralnet_toolbox.Explorer.core.feature_vectors import COLOR_FEATURE_DIMS, COLOR_MOMENT_WEIGHT
    F = np.asarray(features, dtype=np.float32)
    if F.ndim != 2 or F.shape[1] < COLOR_FEATURE_DIMS or len(F) == 0:
        return None
    import cv2
    h = float(np.clip(F[:, 64].mean() / COLOR_MOMENT_WEIGHT, 0, 1)) * 179.0
    s = float(np.clip(F[:, 66].mean() / COLOR_MOMENT_WEIGHT, 0, 1)) * 255.0
    v = float(np.clip(F[:, 68].mean() / COLOR_MOMENT_WEIGHT, 0, 1)) * 255.0
    rgb = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])

import numpy as np

from mincut import get_min_mistakes_cut


def compute_partition_medians(data, labels, n_clusters):
    data = np.asarray(data)
    labels = np.asarray(labels, dtype=int)
    centers = []
    for cluster in range(n_clusters):
        members = data[labels == cluster]
        if len(members) == 0:
            raise ValueError(
                f"Algorithm 2 cannot compute a median center: cluster {cluster} is empty"
            )
        center = np.median(members, axis=0)
        if not np.isfinite(center).all():
            raise ValueError(
                f"Algorithm 2 median center for cluster {cluster} contains NaN/Inf"
            )
        centers.append(center)
    return np.asarray(centers)


class Node:
    def __init__(self, x_data, y_data, center_indices, split_feature=None,
                 split_value=None, left=None, right=None, label=None):
        self.x_data = x_data
        self.y_data = y_data
        self.center_indices = np.asarray(center_indices, dtype=int)
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        return self.left is None and self.right is None


class Tree:
    def __init__(self):
        self.root = None

    def build(self, x_data, label_matrix, centers):
        self.root = self._build_node(
            np.asarray(x_data), np.asarray(label_matrix, dtype=int),
            np.asarray(centers), np.arange(len(centers), dtype=int)
        )
        return self

    def _build_node(self, x_data, y_data, centers, center_indices):
        node = Node(x_data=x_data, y_data=y_data, center_indices=center_indices)
        if len(center_indices) == 1:
            node.label = int(center_indices[0])
            return node

        valid_centers = np.zeros(len(centers), dtype=int)
        valid_centers[center_indices] = 1
        best_feature = None
        best_threshold = None
        minimum_mistakes = float("inf")

        for view_index in range(y_data.shape[1]):
            result = get_min_mistakes_cut(
                x_data,
                y_data[:, view_index],
                centers,
                valid_centers,
                np.ones(x_data.shape[1], dtype=int),
            )
            if result is None:
                continue
            cut, mistakes = result
            if mistakes < minimum_mistakes:
                best_feature = int(cut.col)
                best_threshold = float(cut.threshold)
                minimum_mistakes = mistakes

        if best_feature is None:
            raise RuntimeError(
                "Algorithm 2 found no valid axis-aligned split for the surviving centers"
            )

        left_center_indices = center_indices[
            centers[center_indices, best_feature] <= best_threshold
        ]
        right_center_indices = center_indices[
            centers[center_indices, best_feature] > best_threshold
        ]
        if len(left_center_indices) == 0 or len(right_center_indices) == 0:
            raise RuntimeError("Algorithm 2 selected a split with an empty center branch")

        left_mask = x_data[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        node.split_feature = best_feature
        node.split_value = best_threshold
        node.left = self._build_node(
            x_data[left_mask], y_data[left_mask], centers, left_center_indices
        )
        node.right = self._build_node(
            x_data[right_mask], y_data[right_mask], centers, right_center_indices
        )
        return node

    def predict(self, data):
        predictions = []
        for sample in np.asarray(data):
            node = self.root
            while not node.is_leaf():
                node = node.left if sample[node.split_feature] <= node.split_value else node.right
            predictions.append(node.label)
        return np.asarray(predictions, dtype=int)


def tree_fusion(label_matrix, views, n_clusters):
    per_view_centers = [
        compute_partition_medians(view, label_matrix[:, view_index], n_clusters)
        for view_index, view in enumerate(views)
    ]
    centers = np.concatenate(per_view_centers, axis=1)
    data = np.concatenate(views, axis=1)
    tree = Tree().build(data, label_matrix, centers)
    return tree, tree.predict(data), centers

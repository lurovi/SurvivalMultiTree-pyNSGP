from __future__ import annotations
from typing import List
from genepro.node_impl import Feature, OOHRdyFeature


import numpy as np


class MultiTree:
    def __init__(self):
        self.alpha = None
        self.trees = []
        self.coefficients = []
        self.offset = 0.0
        self.actual_trees_indices = []

        self.objectives = []
        self.crowding_distance = 0
        self.rank = 0
        self.fitness = 0

        self.cox = None
        self.scaler = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.get_output(X)

    def __str__(self) -> str:
        strings_representations = [self.trees[tree_index].get_readable_repr() for tree_index in self.actual_trees_indices]
        return ' | '.join(strings_representations)

    def __len__(self) -> int:
        lengths = [len(self.trees[tree_index]) for tree_index in self.actual_trees_indices]
        return max(lengths)

    def get_readable_repr(self) -> str:
        return str(self)

    def number_of_trees(self) -> int:
        return len(self.trees)

    def number_of_actual_trees(self) -> int:
        return len(self.actual_trees_indices)

    def get_output(self, X: np.ndarray) -> np.ndarray:
        outs = []
        for tree in self.trees:
            out = tree(X)
            outs.append(out)
        outs = np.array(outs)
        return outs.reshape((len(X), len(self.trees)))

    def dominates(self, other):
        better_somewhere = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            if self.objectives[i] < other.objectives[i]:
                better_somewhere = True

        return better_somewhere

    @staticmethod
    def extract_feature_ids(tree):
        feature_ids = set()
        for node in tree.get_subtree():
            if isinstance(node, Feature) or isinstance(node, OOHRdyFeature):
                feature_ids.add(node.id)
        return list(feature_ids)

    @staticmethod
    def extract_feature_nodes(tree):
        feature_nodes = list()
        for node in tree.get_subtree():
            if isinstance(node, Feature) or isinstance(node, OOHRdyFeature):
                feature_nodes.append(node)
        return feature_nodes

    @staticmethod
    def extract_usable_leaves(
        idx_tree: int,
        mt: MultiTree,
        leaf_nodes: List,
        partition_features: bool = False,
    ) -> List:
        usable_leaf_nodes = leaf_nodes
        if partition_features:
            features = list()
            constants = list()
            for l in leaf_nodes:
                if isinstance(l, Feature) or isinstance(l, OOHRdyFeature):
                    features.append(l)
                else:
                    constants.append(l)
            # pick features from all other trees
            other_trees = [t for i, t in enumerate(mt.trees) if i != idx_tree and i in mt.actual_trees_indices]
            used_features = set()
            for ot in other_trees:
                used_features.update(MultiTree.extract_feature_ids(ot))
            usable_leaf_nodes = [
                f for f in features if f.id not in used_features
            ] + constants
        return usable_leaf_nodes

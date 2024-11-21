from typing import List
from genepro.evo import Evolution
from genepro.node_impl import Feature
from tqdm import tqdm
import pandas as pd
import sympy as sp

from typing import Callable

from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import shuffle
import time, inspect
from copy import deepcopy
from joblib.parallel import Parallel, delayed

from genepro.node import Node
from genepro.node_impl import Constant, Plus, Minus
from genepro.variation import *
from genepro.variation import (
    __undergo_variation_operator,
    __check_tree_meets_all_constraints,
)

from survival_genepro.more_variation import generate_random_nonlinear_tree


class MultiTree:
    def __init__(self):
        self.trees = []
        self.fitness = 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.get_output(X)

    def __len__(self) -> int:
        lengths = [len(tree) for tree in self.trees]
        return max(lengths)

    def get_output(self, X: np.ndarray) -> np.ndarray:
        outs = []
        for tree in self.trees:
            out = tree(X)
            outs.append(out)
        outs = np.array(outs)
        return outs.reshape((len(X), len(self.trees)))


def _extract_feature_ids(tree):
    feature_ids = set()
    for node in tree.get_subtree():
        if isinstance(node, Feature):
            feature_ids.add(node.id)
    return list(feature_ids)


def _generate_random_multitree(
    internal_nodes: List,
    leaf_nodes: List,
    max_depth: int,
    partition_features: bool = False,
    min_trees_init: int = 2,
    max_trees_init: int = 5,
) -> MultiTree:
    mt = MultiTree()

    features = list()
    constants = list()
    for l in leaf_nodes:
        if isinstance(l, Feature):
            features.append(l)
        else:
            constants.append(l)

    num_trees = np.random.choice(list(range(min_trees_init, max_trees_init + 1)))
    features_used = set()
    for _ in range(num_trees):
        curr_leaves = leaf_nodes
        if partition_features:
            if len(features_used) == len(features):
                continue
            curr_leaves = [f for f in features if f.id not in features_used] + constants

        tree = generate_random_nonlinear_tree(
            internal_nodes=internal_nodes, leaf_nodes=curr_leaves, max_depth=max_depth
        )

        features_used.update(_extract_feature_ids(tree))
        mt.trees.append(tree)

    return mt


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
            if isinstance(l, Feature):
                features.append(l)
            else:
                constants.append(l)
        # pick features from all other trees
        other_trees = [t for i, t in enumerate(mt.trees) if i != idx_tree]
        used_features = set()
        for ot in other_trees:
            used_features.update(_extract_feature_ids(ot))
        usable_leaf_nodes = [
            f for f in features if f.id not in used_features
        ] + constants
    return usable_leaf_nodes


def multitree_level_crossover(
    mt: MultiTree,
    donor_mt: MultiTree,
    idx_picked_tree: "int | None" = None,
) -> MultiTree:

    if idx_picked_tree is None:
        idx_picked_tree = np.random.randint(len(mt.trees))

    donor_tree = deepcopy(np.random.choice(donor_mt.trees))

    # drop the tree to be replaced
    mt.trees.pop(idx_picked_tree)
    mt.trees.insert(idx_picked_tree, donor_tree)
    return mt


def _generate_offspring_multitree(
    parent_mt: MultiTree,
    crossovers: list,
    mutations: list,
    coeff_opts: list,
    donors: list,
    internal_nodes: list,
    leaf_nodes: list,
    max_depth: int,
    constraints: dict = {"max_tree_size": 100},
    partition_features: bool = False,
    prob_delete_tree: float = 0.05,
    prob_init_tree: float = 0.1,
    prob_mt_crossover: float = 0.0,
) -> MultiTree:

    if prob_mt_crossover > 0 and partition_features:
        raise ValueError(
            "Partition features and multitree crossover are not compatible"
        )

    # set the offspring to a copy (to be modified) of the parent
    offspring_mt = deepcopy(parent_mt)
    idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # Case: delete tree
    if np.random.uniform() < prob_delete_tree and len(offspring_mt.trees) > 1:
        offspring_mt.trees.pop(idx_picked_tree)
        # update idx picked tree
        idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # compute what features can be used
    usable_leaf_nodes = extract_usable_leaves(
        idx_picked_tree, offspring_mt, leaf_nodes, partition_features
    )

    # Case: generate a new tree to add
    if np.random.uniform() < prob_init_tree and len(usable_leaf_nodes) > 0:
        # initialize a new tree
        new_tree = generate_random_nonlinear_tree(
            internal_nodes=internal_nodes,
            leaf_nodes=usable_leaf_nodes,
            max_depth=max_depth,
        )
        offspring_mt.trees.append(new_tree)
        # update idx picked tree
        idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # Case: multitree crossover
    if np.random.uniform() < prob_mt_crossover:
        # pick a donor
        donor_mt = np.random.choice(donors)
        offspring_mt = multitree_level_crossover(
            offspring_mt,
            donor_mt,
            idx_picked_tree=idx_picked_tree,
        )
        idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # Next: undergo node-level variation operators

    # pick the tree to modify
    offspring_tree = deepcopy(offspring_mt.trees[idx_picked_tree])
    # create a backup for constraint violation
    backup_tree = deepcopy(offspring_tree)

    # apply variation operators in a random order
    all_var_ops = crossovers + mutations + coeff_opts
    random_order = np.arange(len(all_var_ops))
    shuffle(random_order)
    for i in random_order:
        var_op = all_var_ops[i]
        # randomize donors
        donor_trees = [np.random.choice(donor.trees) for donor in donors]
        offspring_tree = __undergo_variation_operator(
            var_op,
            offspring_tree,
            crossovers,
            mutations,
            coeff_opts,
            np.random.choice(donor_trees),
            internal_nodes,
            usable_leaf_nodes,
        )

        # check offspring_tree meets constraints, else revert to backup
        if not __check_tree_meets_all_constraints(offspring_tree, constraints):
            # revert to backup
            offspring_tree = deepcopy(backup_tree)
        else:
            # update backup
            backup_tree = deepcopy(offspring_tree)

    # print("len of offspring tree", len(offspring_tree.get_subtree()))
    # update with offspring tree
    offspring_mt.trees.pop(idx_picked_tree)
    offspring_mt.trees.insert(idx_picked_tree, offspring_tree)

    # assert len is not violated
    # for off_tree in offspring_mt.trees:
    #    if len(off_tree.get_subtree()) > constraints["max_tree_size"]:
    #        print(f"Starting from {len(backup_tree.get_subtree())} nodes")
    #        print("mt crossover happened:", mt_crossover_happened)
    #        raise ValueError(f"Tree size constraint violated")

    return offspring_mt


class BootstrappedEvolution(Evolution):

    def __init__(
        self,
        survival_score,
        n_bootstrap: int = 1000,
        bootstrap_size: float = 0.5,
        sample_w_replacement: bool = True,
        prob_init_tree: float = 0.1,
        prob_delete_tree: float = 0.05,
        prob_mt_crossover: float = 0.0,
        min_trees_init: int = 2,
        max_trees_init: int = 5,
        drop_features: bool = False,
        only_generated: bool = False,
        early_stopping_rounds: int = 5,
        inner_val_size: int = 0.1,
        top_common_to_extract: int = 5,
        *args,
        **kwargs,
    ):
        kwargs["fitness_function"] = None
        super().__init__(*args, **kwargs)
        self.fitness_function = None
        self.survival_score = survival_score
        self.inner_val_size = inner_val_size

        self.n_bootstrap = n_bootstrap
        self.bootstrap_size = bootstrap_size
        self.sample_w_replacement = sample_w_replacement
        self.top_common_to_extract = top_common_to_extract

        self.population = []
        self.num_evals = 0
        self.num_gens = 0
        self.best_of_gens = []
        self.drop_features = drop_features
        self.only_generated = only_generated
        self.early_stopping_rounds = early_stopping_rounds

        self.prob_init_tree = prob_init_tree
        self.prob_delete_tree = prob_delete_tree
        self.prob_mt_crossover = prob_mt_crossover
        self.min_trees_init = min_trees_init
        self.max_trees_init = max_trees_init

        self.final_surv_model = None
        self.evolved_trees = []  # for convenience to match boosted version

    def reset(self):
        self.final_surv_model = None
        self.best_of_gens = []
        self.evolved_trees = []

    def evolve(
        self,
        X,
        y,
    ):
        self.reset()
        X_val, y_val = None, None
        if self.inner_val_size > 0:
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.inner_val_size,
                random_state=np.random.randint(1000),
                stratify=[y_i[0] for y_i in y],
            )

        def fitness_function(
            idx,
            mt,
            survival_score,
        ) -> float:

            # subsample the data based on idx
            rand_state_idx = int(idx % self.bootstrap_size)

            X_b = X.reset_index(drop=True).sample(
                frac=self.bootstrap_size,
                replace=self.sample_w_replacement,
                random_state=rand_state_idx,
            )
            # get respective y_b
            y_b = y[X_b.index]

            # the tree takes the original X
            X_generated = self._generate_transformed_X(X_b, mt_to_use=mt)
            score, _ = survival_score(X_generated, y_b)
            return score

        self.fitness_function = lambda idx, mt: fitness_function(
            idx, mt, self.survival_score
        )

        self._evolve_inner()

        # get most common trees
        common_trees = self._extract_common_trees(
            top=self.top_common_to_extract,
            pop_quantile=0.5,
        )
        num_common_trees = len(common_trees)
        # filter out redundancies
        self.evolved_trees = self._filter_out_redundancies(common_trees, X)
        num_filtered_out = num_common_trees - len(self.evolved_trees)
        if self.verbose:
            print(
                f"Number of common trees: {len(self.evolved_trees)} ({num_filtered_out} were filtered out)"
            )
            print("Trees are:")
            for tree in self.evolved_trees:
                print("\t", sp.simplify(tree.get_readable_repr()))

        # set final stuff
        X_transf = self._generate_transformed_X(X)
        _, self.final_surv_model = self.survival_score(X_transf, y)

        """
        if self.inner_val_size > 0:
            best_score_val = float("-inf")
            best_generalizing_mt_idx = -1
            for i, mt in enumerate(self.population):
                X_transf_val = self._generate_transformed_X(X_val, mt_to_use=mt)
                score_val, _ = self.survival_score(X_transf_val, y_val)
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_generalizing_mt_idx = i
            # if self.verbose:
            #    print(f"Best generalizing best-of-generations idx: {best_generalizing_mt_idx}")
            # set to the best-generalizing
            self.evolved_trees = self.population[best_generalizing_mt_idx].trees
            # fit on the entire training set
            X_transf = self._generate_transformed_X(X)
            _, self.final_surv_model = self.survival_score(X_transf, y)
        else:
            X_transf = self._generate_transformed_X(X)
            _, self.final_surv_model = self.survival_score(X_transf, y)
        """

    def _evolve_inner(self):
        """
        Performs the evolution process
        """
        self._initialize_population()
        no_improvement_count = 0
        least_fitness_so_far = float("-inf")
        while self.num_gens < self.max_gens:
            self._perform_generation()

            # update the evolved trees anyway
            self.evolved_trees = self.best_of_gens[-1].trees

            # check early stopping
            min_fitness = min([mt.fitness for mt in self.population])
            if min_fitness > least_fitness_so_far:
                least_fitness_so_far = min_fitness
                # self.evolved_trees = self.best_of_gens[-1].trees
                no_improvement_count = 0
            # check early stopping
            elif self.early_stopping_rounds > 0:
                no_improvement_count += 1

            if self.verbose:
                print(
                    "gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {},\tno improv since: {}".format(
                        self.num_gens,
                        self.best_of_gens[-1].fitness,
                        [
                            len(tree.get_subtree())
                            for tree in self.best_of_gens[-1].trees
                        ],
                        no_improvement_count,
                    )
                )
            if no_improvement_count >= self.early_stopping_rounds:
                print("-> early stopping")
                break

    def _initialize_population(self):
        """
        Generates a random initial population and evaluates it
        """
        # initialize the population
        self.population = Parallel(n_jobs=self.n_jobs)(
            delayed(_generate_random_multitree)(
                internal_nodes=self.internal_nodes,
                leaf_nodes=self.leaf_nodes,
                max_depth=self.init_max_depth,
                partition_features=self.drop_features,
                min_trees_init=self.min_trees_init,
                max_trees_init=self.max_trees_init,
            )
            for _ in range(self.pop_size)
        )

        # evaluate the trees and store their fitness
        fitnesses = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fitness_function)(
                idx,
                mt,
            )
            for idx, mt in enumerate(self.population)
        )

        for i in range(self.pop_size):
            self.population[i].fitness = fitnesses[i]
            # store eval cost
            self.num_evals += self.pop_size

        # store best at initialization
        best = self.population[np.argmax([t.fitness for t in self.population])]
        self.best_of_gens.append(deepcopy(best))
        self.evolved_trees = self.best_of_gens[-1].trees

    def _perform_generation(self):
        """
        Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
        """
        # select promising parents
        # sel_fun = self.selection["fun"]
        # parents = sel_fun(self.population, self.pop_size, **self.selection["kwargs"])
        parents = self.population

        # generate offspring
        offspring_population = Parallel(n_jobs=self.n_jobs)(
            delayed(_generate_offspring_multitree)(
                mt,
                self.crossovers,
                self.mutations,
                self.coeff_opts,
                parents,
                self.internal_nodes,
                self.leaf_nodes,
                max_depth=self.init_max_depth,
                constraints={"max_tree_size": self.max_tree_size},
                partition_features=self.drop_features,
                prob_delete_tree=self.prob_delete_tree,
                prob_init_tree=self.prob_init_tree,
                prob_mt_crossover=self.prob_mt_crossover,
            )
            for mt in parents
        )

        # evaluate each offspring and store its fitness
        fitnesses = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fitness_function)(
                idx,
                mt,
            )
            for idx, mt in enumerate(offspring_population)
        )
        for i in range(self.pop_size):
            offspring_population[i].fitness = fitnesses[i]
        # store cost
        self.num_evals += self.pop_size
        # update the population for the next iteration
        for i in range(self.pop_size):
            if offspring_population[i].fitness >= self.population[i].fitness:
                self.population[i] = deepcopy(offspring_population[i])

        # update info
        self.num_gens += 1
        best = self.population[np.argmax([mt.fitness for mt in self.population])]
        self.best_of_gens.append(deepcopy(best))

    def _generate_transformed_X(
        self, X, mt_to_use=None, drop_constant=False, drop_corr_threshold=0.0
    ):

        if mt_to_use is None:
            mt_to_use = MultiTree()
            mt_to_use.trees = self.evolved_trees
            # mt_to_use = self.best_of_gens[-1]

        X_transf = pd.DataFrame(
            mt_to_use.get_output(X.to_numpy().astype(float)),
            columns=[f"gen_feature_{i}" for i in range(len(mt_to_use.trees))],
        )

        # drop all constant columns
        if drop_constant:
            for col in X_transf.columns:
                if len(X_transf[col].unique()) == 1:
                    X_transf.drop(col, axis=1, inplace=True)

        # drop 2nd correlated features
        if drop_corr_threshold > 0:
            to_drop_corr = []
            for i in range(len(X_transf.columns)):
                for j in range(i + 1, len(X_transf.columns)):
                    if X_transf.corr().iloc[i, j] > drop_corr_threshold:
                        to_drop_corr.append(X_transf.columns[j])
            X_transf = X_transf.drop(to_drop_corr, axis=1)

        if not self.only_generated:
            X_to_append = X.copy()
            if self.drop_features:
                features_to_drop = set()
                for t in mt_to_use.trees:
                    features_to_drop.update(_extract_feature_ids(t))
                if len(features_to_drop) > 0:
                    X_to_append = X.drop(X.columns[list(features_to_drop)], axis=1)

            # drop from X_transf columns that are already in X_to_append
            # in terms of correlation
            if drop_corr_threshold > 0:
                to_drop_corr = []
                for i in range(len(X_transf.columns)):
                    for j in range(len(X_to_append.columns)):
                        if (
                            X_transf.corrwith(X_to_append.iloc[:, j]).iloc[i]
                            > drop_corr_threshold
                        ):
                            to_drop_corr.append(X_transf.columns[i])
                X_transf = X_transf.drop(to_drop_corr, axis=1)

            X_transf = pd.concat(
                [X_transf.reset_index(drop=True), X_to_append.reset_index(drop=True)],
                axis=1,
            ).reset_index(drop=True)

        return X_transf

    def _extract_common_trees(
        self,
        top: int = 5,
        min_count: int = 0,
        pop_quantile: float = 0.5,
    ):
        if top > 0 and min_count > 0:
            raise ValueError("Cannot set both top and min_count")
        if top == 0 and min_count == 0:
            raise ValueError("Either top or min_count must be set")

        # get best part of population
        best_population = sorted(
            self.population, key=lambda x: x.fitness, reverse=True
        )[: int(pop_quantile * len(self.population))]

        seen_trees = dict()
        for mt in best_population:
            for tree in mt.trees:
                # "normalize" constants
                cpy = deepcopy(tree)
                for n in cpy.get_subtree():
                    if isinstance(n, Constant):
                        v = n.get_value()
                        n.set_value(np.round(v))
                    
                str_repr = cpy.get_readable_repr()
                str_repr = sp.simplify(str_repr)
                if str_repr not in seen_trees:
                    seen_trees[str_repr] = {
                        "count": 1,
                        "trees": [tree],
                    }
                else:
                    seen_trees[str_repr]["count"] += 1
                    seen_trees[str_repr]["trees"].append(tree)

        # reduce to 1 representative tree per item
        for k, v in seen_trees.items():
            trees = v["trees"]
            # get the smallest
            min_size = min([len(t.get_subtree()) for t in trees])
            min_tree = [t for t in trees if len(t.get_subtree()) == min_size][0]
            seen_trees[k]["tree"] = min_tree

        if top > 0:
            common_trees = sorted(
                seen_trees.values(), key=lambda x: x["count"], reverse=True
            )[:top]
            
            baba = sorted(
                seen_trees.values(), key=lambda x: x["count"], reverse=True
            )
            for tree in baba:
                print("count:", tree["count"])
                print(sp.simplify(tree["tree"].get_readable_repr()))
            
            return [t["tree"] for t in common_trees]
        else:
            common_trees = [
                t["tree"] for t in seen_trees.values() if t["count"] >= min_count
            ]
            return common_trees

    def _filter_out_redundancies(
        self,
        trees: List[Node],
        X: pd.DataFrame,
        corr_threshold: float = 0.98,
    ) -> List[Node]:
        # redundant trees have constant output
        # or correlated output
        to_remove = []
        for i, tree in enumerate(trees):
            out = tree(X.to_numpy().astype(float))
            if len(np.unique(out)) == 1:
                to_remove.append(i)
        # sort descending
        to_remove = sorted(to_remove, reverse=True)
        for idx in to_remove:
            trees.pop(idx)
        
        tree_outputs = [tree(X.to_numpy().astype(float)) for tree in trees]
        to_remove = set()
        # remove any tree that is correlated to the initial columns
        for i in range(len(trees)):
            for j in range(len(X.columns)):
                if np.corrcoef(tree_outputs[i], X.iloc[:, j])[0, 1] > corr_threshold:
                    to_remove.add(i)
        # sort descending
        to_remove = sorted(list(to_remove), reverse=True)
        for idx in to_remove:
            trees.pop(idx)
            tree_outputs.pop(idx)

        # finally, remove trees that are correlated to each other
        to_remove = set()
        for i in range(len(tree_outputs)):
            for j in range(i + 1, len(tree_outputs)):
                out_i = tree_outputs[i]
                out_j = tree_outputs[j]
                if np.corrcoef(out_i, out_j)[0, 1] > corr_threshold:
                    to_remove.add(j)
        # sort descending
        to_remove = sorted(list(to_remove), reverse=True)
        for idx in to_remove:
            trees.pop(idx)
            
        return trees

    def predict(self, X):
        X_transf = self._generate_transformed_X(X)
        preds = self.final_surv_model.predict(X_transf)
        return preds

    def score(self, X, y):
        X_transf = self._generate_transformed_X(X)
        try:
            score = self.final_surv_model.score(X_transf, y)
        except ValueError:
            score = float("nan")
        return score

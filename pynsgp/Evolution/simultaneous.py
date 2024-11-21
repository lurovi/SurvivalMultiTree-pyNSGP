from genepro.evo import Evolution
import pandas as pd

from typing import Callable

from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy
from joblib.parallel import Parallel, delayed

from pynsgp.Utils.sksurv_util import tune_coxnet
from pynsgp.Nodes.MultiTree import MultiTree, _create_dummy_multi_tree, _generate_random_multitree, _extract_feature_ids
from pynsgp.Variation.Variation import _generate_offspring_multitree


class SimultaneousEvolution(Evolution):

    def __init__(
        self,
        survival_score: Callable,
        prob_init_tree: float = 0.1,
        prob_delete_tree: float = 0.05,
        prob_mt_crossover: float = 0.0,
        min_trees_init: int = 2,
        max_trees_init: int = 5,
        drop_features: bool = False,
        drop_numerical_from_X: bool = False,
        only_generated: bool = False,
        early_stopping_rounds: int = 5,
        inner_val_size: int = 0.1,
        perform_final_tuning: bool = False,
        final_tune_n_hyperp_folds: int = 3,
        final_tune_metric: str = "cindex",
        *args,
        **kwargs,
    ):
        kwargs["fitness_function"] = None
        super().__init__(*args, **kwargs)
        self.fitness_function = None
        self.survival_score = survival_score
        self.inner_val_size = inner_val_size
        self.population = []
        self.num_evals = 0
        self.num_gens = 0
        self.best_of_gens = []
        self.best_val_fitnesses = []
        self.drop_features = drop_features
        self.only_generated = only_generated
        self.drop_numerical_from_X = drop_numerical_from_X
        self.early_stopping_rounds = early_stopping_rounds

        self.prob_init_tree = prob_init_tree
        self.prob_delete_tree = prob_delete_tree
        self.prob_mt_crossover = prob_mt_crossover
        self.min_trees_init = min_trees_init
        self.max_trees_init = max_trees_init

        self.final_surv_model = None
        self.evolved_trees = []  # for convenience to match boosted version

        self.perform_final_tuning = perform_final_tuning
        self.final_tune_n_hyperp_folds = final_tune_n_hyperp_folds
        self.final_tune_metric = final_tune_metric

    def reset(self):
        self.final_surv_model = None
        self.best_of_gens = []
        self.evolved_trees = []

    def update_fitness_function(self, X, y, X_val=None, y_val=None):
        def fitness_function(
            mt,
            survival_score,
        ) -> float:
            # the tree takes the original X
            X_generated = self._generate_transformed_X(X, mt_to_use=mt)
            score, surv_model = survival_score(X_generated, y)
            if X_val is not None:
                X_gen_val = self._generate_transformed_X(X_val, mt_to_use=mt)
                try:
                    score = surv_model.score(X_gen_val, y_val)
                except:
                    return 0
            return score

        self.fitness_function = lambda mt: fitness_function(mt, self.survival_score)

    def evolve(
        self,
        X,
        y,
    ):
        self.reset()

        X_val, y_val = None, None
        if self.inner_val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.inner_val_size,
                random_state=np.random.randint(1000),
                stratify=[y_i[0] for y_i in y],
            )
        else:
            X_train, y_train = X, y

        self.update_fitness_function(X_train, y_train)
        self._evolve_inner(X_train, y_train, X_val, y_val)

        best_score = float("-inf")
        if self.inner_val_size > 0:
            # get based on valid
            best_mt_idx = np.argmax(self.best_val_fitnesses)
            mt = self.best_of_gens[best_mt_idx]
            #mt = self.cleanup_mt(X, mt, corr_threshold=0.98)
            self.evolved_trees = mt.trees
        else:
            mt = self.best_of_gens[-1]
            #mt = self.cleanup_mt(X, mt, corr_threshold=0.98)
            self.evolved_trees = mt.trees
            
        X_transf = self._generate_transformed_X(X)
        if self.perform_final_tuning:
            self.final_surv_model = tune_coxnet(
                X_transf,
                y,
                n_hyperp_folds=self.final_tune_n_hyperp_folds,
                downsample_alphas=False,
                metric=self.final_tune_metric,
            )
            if self.final_surv_model is None:
                best_score = float("-inf")
            else:
                best_score = self.final_surv_model.score(X_transf, y)
        else:
            best_score, self.final_surv_model = self.survival_score(X_transf, y)

        # compare with dummy tree (constant 0) and keep best
        dummy_mt = _create_dummy_multi_tree()
        dummy_Xt = self._generate_transformed_X(X, mt_to_use=dummy_mt)
        dummy_surv_model = tune_coxnet(
            dummy_Xt,
            y,
            n_hyperp_folds=self.final_tune_n_hyperp_folds,
            downsample_alphas=False,
            metric=self.final_tune_metric,
        )
        dummy_score = dummy_surv_model.score(dummy_Xt, y)
        if dummy_score > best_score:
            print(
                "Warning: evolution ({:.3f}) performed worse than baseline ({:.3f})".format(
                    best_score, dummy_score
                )
            )
            self.evolved_trees = dummy_mt.trees
            self.final_surv_model = dummy_surv_model

        return None


    def cleanup_mt(
        self, X: pd.DataFrame, mt: MultiTree, corr_threshold: float = 0.98
    ) -> MultiTree:
        X_prime = pd.DataFrame(
            mt.get_output(X.to_numpy().astype(float)),
            columns=[f"gen_feature_{i}" for i in range(len(mt.trees))],
        )
        trees_to_drop = set()
        # drop all constant columns
        for i, col in enumerate(X_prime.columns):
            if len(X_prime[col].unique()) == 1:
                trees_to_drop.add(i)

        if corr_threshold > 0:
            # drop corr features wrt X_prime
            for i in range(len(X_prime.columns)):
                for j in range(i + 1, len(X_prime.columns)):
                    if X_prime.corr().iloc[i, j] > corr_threshold:
                        trees_to_drop.add(j)

            # drop corr features wrt X
            for i in range(len(X_prime.columns)):
                for j in range(len(X.columns)):
                    if X_prime.corrwith(X.iloc[:, j]).iloc[i] > corr_threshold:
                        trees_to_drop.add(i)

        # drop trees
        if self.verbose:
            print(
                f"\tCleanup: {trees_to_drop} ({len(trees_to_drop)}) dropped out of {len(mt.trees)}"
            )
        mt.trees = [t for i, t in enumerate(mt.trees) if i not in trees_to_drop]
        return mt

    def _evolve_inner(self, X, y, X_val=None, y_val=None):
        """
        Performs the evolution process
        """

        self._initialize_population()
        no_improvement_count = 0
        best_fitness_so_far = float("-inf")
        while self.num_gens < self.max_gens:

            self._perform_generation()

            # update the evolved trees to the last best
            self.evolved_trees = self.best_of_gens[-1].trees

            # check early stopping
            best_fit_curr = self.best_of_gens[-1].fitness

            # if val is provided, then check on val
            if X_val is not None:
                X_val_transf = self._generate_transformed_X(X_val)
                val_score = self.survival_score(X_val_transf, y_val)[0]
                best_fit_curr = val_score
                self.best_val_fitnesses.append(val_score)

            if best_fit_curr > best_fitness_so_far:
                best_fitness_so_far = best_fit_curr
                no_improvement_count = 0
            # check early stopping
            elif self.early_stopping_rounds > 0:
                no_improvement_count += 1

            if self.verbose:
                print(
                    "gen: {},\tbest of gen fitness train: {:.3f}, val: {:.3f},\tbest of gen size: {},\tno improv since: {}".format(
                        self.num_gens,
                        self.best_of_gens[-1].fitness,
                        best_fit_curr if X_val is not None else float("nan"),
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
                mt,
            )
            for mt in self.population
        )

        for i in range(self.pop_size):
            self.population[i].fitness = fitnesses[i]
            # store eval cost
            self.num_evals += self.pop_size

        # store best at initialization
        best = self.population[np.argmax([t.fitness for t in self.population])]
        self.best_of_gens.append(deepcopy(best))

    def _perform_generation(self):
        """
        Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
        """
        # select promising parents
        sel_fun = self.selection["fun"]
        parents = sel_fun(self.population, self.pop_size, **self.selection["kwargs"])
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
                mt,
            )
            for mt in offspring_population
        )
        for i in range(self.pop_size):
            offspring_population[i].fitness = fitnesses[i]
        # store cost
        self.num_evals += self.pop_size
        # update the population for the next iteration
        self.population = offspring_population
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
            if self.drop_numerical_from_X:
                # enable only categorical to be appended
                X_to_append = X_to_append[
                    [col for col in X.columns if len(np.unique(X[col])) == 2]
                ]
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

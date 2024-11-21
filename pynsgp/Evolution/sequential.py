from tqdm import tqdm
import pandas as pd
import numpy as np
from genepro.evo import Evolution
from genepro.node_impl import Feature
from survival_genepro.more_node_impl import OOHRdyFeature
from survival_genepro.sksurv_util import tune_coxnet

class SequentialEvolution(Evolution):

    def __init__(
        self,
        survival_score,
        n_sequential_rounds: int = 5,
        drop_features: bool = False,
        only_generated: bool = False,
        early_stopping_rounds: int = 5,
        inner_val_size: float = 0.1,
        final_tune_n_hyperp_folds: int = 3,
        final_tune_metric: str = "cindex",
        *args,
        **kwargs,
    ):
        kwargs["fitness_function"] = None
        super().__init__(*args, **kwargs)
        self.survival_score = survival_score
        self.inner_val_size = inner_val_size
        self.n_sequential_rounds = n_sequential_rounds
        self.inner_val_size = inner_val_size
        self.used_features = set()
        self.evolved_trees = []
        self.drop_features = drop_features
        self.only_generated = only_generated
        self.early_stopping_rounds = early_stopping_rounds
        self.final_tune_n_hyperp_folds = final_tune_n_hyperp_folds
        self.final_tune_metric = final_tune_metric
        self.final_surv_model = None
        
    def gen_fitness_function(self, X, y):
        X_transf_cache = self._generate_transformed_X(X)
        
        def fitness_function(tree):
            X_transf = self._generate_transformed_X(
                X, extra_tree=tree, X_transf_cache=X_transf_cache,
            )
            score, _ = self.survival_score(X_transf, y)
            return score
        
        return fitness_function

    def _extract_feature_ids(self, tree):
        for node in tree.get_subtree():
            if isinstance(node, Feature) or isinstance(node, OOHRdyFeature):
                yield node.id

    def reset(self):
        self.used_features = set()
        self.evolved_trees = []
        self.final_surv_model = None
        self.best_of_gens = []
        self.X, self.y, self.X_transf_cache = None, None, None

    def evolve(
        self,
        X,
        y,
    ):
        self.reset()
        self.X = X
        self.y = y
                
        for i in tqdm(range(self.n_sequential_rounds)):
            self.best_of_gens = []
            self.fitness_function = self.gen_fitness_function(X, y)

            if len(self.used_features) == X.shape[1]:
                if self.verbose:
                    print("All features used")
                break

            # leaf nodes of this round:
            # original not used so far and after the pre-pended new ones
            if self.drop_features:
                leaf_nodes = [
                    x for x in self.leaf_nodes
                    if not (isinstance(x, Feature) or isinstance(x, OOHRdyFeature)) or (x.id not in self.used_features)
                ]
                self.leaf_nodes = leaf_nodes

            # evolve
            self._evolve_inner()

            # extract best tree
            best_tree = self.best_of_gens[-1]
                            
            # extract original feature ids
            feature_ids = list(self._extract_feature_ids(best_tree))
            self.used_features.update(feature_ids)

            # store the best tree
            self.evolved_trees.append(best_tree)
            
        # compute and store best model
        X_transf = self._generate_transformed_X(X)
        
        #_, self.final_surv_model = self.survival_score(X_transf, y)
        # perform a final tuning of the final surv model
        X_transf = self._generate_transformed_X(X)
        self.final_surv_model = tune_coxnet(  # could pass it as "final tuning step"
            X_transf,
            y,
            tune_tol=1e-7,
            n_hyperp_folds=self.final_tune_n_hyperp_folds,
            downsample_alphas=False,
            metric=self.final_tune_metric,
        )
        
    def _evolve_inner(self):
        """
        Performs the evolution process
        """
        self._initialize_population()
        no_improvement_count = 0
        best_fitness_so_far = float("-inf")
        while self.num_gens < self.max_gens:
            self._perform_generation()
            
            # check early stopping
            if self.best_of_gens[-1].fitness > best_fitness_so_far:
                best_fitness_so_far = self.best_of_gens[-1].fitness
                no_improvement_count = 0
            # check early stopping
            elif self.early_stopping_rounds > 0:
                no_improvement_count += 1
                
            if self.verbose:
                print("gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {},\tno improv since: {}".format(
                    self.num_gens, self.best_of_gens[-1].fitness, len(self.best_of_gens[-1]), no_improvement_count
                ))
            if no_improvement_count >= self.early_stopping_rounds:
                print("-> early stopping")
                break

    def _generate_transformed_X(
        self,
        X,
        extra_tree=None,
        X_transf_cache=None
    ):
        trees = []
        X_transf = None
        if X_transf_cache is None:
            trees = self.evolved_trees
        else: 
            X_transf = X_transf_cache.copy()
        
        used_features_extra_tree = set()
        if extra_tree is not None:
            trees = trees + [extra_tree]
            used_features_extra_tree.update(
                self._extract_feature_ids(extra_tree)
            )
        
        for i, tree in enumerate(trees):
            curr_pred = pd.DataFrame(tree(X.to_numpy().astype(float)), columns=[f"gen_feature_{i}"])
            X_transf = pd.concat(
                [
                    X_transf.reset_index(drop=True) if X_transf is not None else None,
                    curr_pred.reset_index(drop=True)
                ],
                axis=1,
            )
        used_features = self.used_features.union(used_features_extra_tree)
        if not self.only_generated:
            X_to_concat = X.copy()
            if self.drop_features and len(used_features) > 0:
                X_to_concat = X.drop(X.columns[list(used_features)], axis=1)
                
            X_transf = pd.concat(
                [
                    X_transf.reset_index(drop=True) if X_transf is not None else None,
                    X_to_concat.reset_index(drop=True)
                ],
                axis=1
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

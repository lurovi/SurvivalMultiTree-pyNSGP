import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import sympy as sp

from genepro.node_impl import (
    Feature,
    Constant,
    Plus,
    Minus,
    Times,
    Div,
    Square,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
)

from pynsgp.Nodes.more_node_impl import (
    ExpPlus, ExpTimes, LogSlack, SqrtSlack, DivSlack,
    UnprotectedDiv, UnprotectedLog, 
    AnalyticQuotient,
    LogSquare,
    Sigmoid,
    OOHRdyFeature,
)

from pynsgp.Evolution.sequential import SequentialEvolution
from pynsgp.Evolution.simultaneous import SimultaneousEvolution
from pynsgp.Evolution.bootstrapped import BootstrappedEvolution


class SurvGeneProEstimator(BaseEstimator):
    def __init__(
        self,
        survival_score=None,
        mode="sequential",
        only_numerical_features=False,
        evo_kwargs=dict(),
    ):
        self.mode = mode
        self.only_numerical_features = only_numerical_features
        self.evo_kwargs = evo_kwargs

        # set up default internal nodes if not provided
        if "internal_nodes" not in self.evo_kwargs:
            self.evo_kwargs["internal_nodes"] = [
                Plus(), Minus(),
                Times(), #UnprotectedDiv(),
                AnalyticQuotient(),
                LogSquare(),
                Sigmoid(),
                # DivSlack(10.0),
                # DivSlack(-10.0),
                # UnprotectedLog(),
                # ExpPlus(), ExpTimes(),
                # LogSlack(10.0),
                # SqrtSlack(10.0),
                # Square(),
                # Sqrt(),
                # Exp(),
                # Log(),
                # Sin(),
                # Cos(),
            ]
        # default leaf nodes can only be set at fitting time (when we know X)
        # hence, initially set to none here, will be set when calling .fit(X,y)
        if "leaf_nodes" not in evo_kwargs:
            self.evo_kwargs["leaf_nodes"] = None

        if self.mode == "sequential":
            self.evo = SequentialEvolution(
                survival_score=survival_score, **self.evo_kwargs
            )
        elif self.mode == "simultaneous":
            self.evo = SimultaneousEvolution(
                survival_score=survival_score, **self.evo_kwargs
            )
        elif self.mode == "bootstrapped":
            self.evo = BootstrappedEvolution(
                survival_score=survival_score, **self.evo_kwargs
            )
        else:
            raise ValueError(f"mode must be 'sequential' or 'simultaneous', got: {self.mode}")

    def fit(self, X, y):
        # check that X and y have correct shape
#        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        # default generation of leaf nodes
        if not self.evo.leaf_nodes:
            #self.evo.leaf_nodes = [Feature(i) for i in range(X.shape[1])]
            self.evo.leaf_nodes = [OOHRdyFeature(i) for i in range(X.shape[1])]
            
            if self.only_numerical_features:
                #self.evo.leaf_nodes = [
                    #Feature(i) for i in range(X.shape[1])
                self.evo.leaf_nodes = [
                    OOHRdyFeature(i) for i in range(X.shape[1])                   
                    if (
                        X[X.columns[i]].dtype in ["float64", "int64"]
                        and
                        len(np.unique(X.iloc[:, i])) > 2
                    )
                ]
            
            self.evo.leaf_nodes += [Constant(1.0), Constant(2.0)]
            
        self.evo.evolve(X, y)
        
        assert hasattr(self.evo, "final_surv_model"), "final_surv_model not set after evolve"
        assert self.evo.final_surv_model is not None, "final_surv_model is None after evolve"

    def predict(self, X):
        # check is fit had been called
        check_is_fitted(self)
        return self.evo.predict(X)

    def score(self, X, y):
        # check is fit had been called
        check_is_fitted(self)
        return self.evo.score(X, y)
    
    def get_final_model(self, X):
        check_is_fitted(self)
        trees = self.evo.evolved_trees
        
        # get readable representations of trees
        readbl_trees = [tree.get_readable_repr() for tree in trees]
        # replace feature ids with feature names
        feature_ids = [f"x_{i}" for i in range(X.shape[1])]
        for i, rtree in enumerate(readbl_trees):
            # gotta parse feature ids in reverse order to avoid replacing substrings
            for j in range(len(feature_ids)-1, -1, -1):
                colname = X.columns[j]
                feature_id = feature_ids[j]
                rtree = rtree.replace(
                    feature_id,
                    colname,
                )
            readbl_trees[i] = rtree
            
        sp_trees = []
        for rtree in readbl_trees:
            sp_tree = sp.simplify(rtree)
            sp_trees.append(sp_tree)
            
        # must also add to the final model the original columns that
        # ended up being used
        if not self.evo.only_generated:
            X_transf = self.evo._generate_transformed_X(X)
            original_col_used = [col for col in X.columns if col in X_transf.columns]
            for col in original_col_used:
                sp_trees += [sp.Symbol(col)]
                
        return {
            "features": sp_trees,
            "surv_model": self.evo.final_surv_model,
        }
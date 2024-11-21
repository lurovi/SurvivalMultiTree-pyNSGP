from collections.abc import Callable
from typing import List, Dict, Any, Tuple
import warnings
import yaml
import pandas as pd

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge
from sksurv.tree import SurvivalTree
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    StratifiedKFold,
)

# from sksurv.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

from pynsgp.scikit import SurvGeneProEstimator
from pynsgp.Utils.sksurv_util import hyperparam_tuning, tune_coxnet
from scoring import compute_metric_scores
from genepro.variation import (
    subtree_crossover,
    node_level_crossover,
    subtree_mutation,
    one_point_mutation,
    coeff_mutation,
)
from genepro.selection import tournament_selection


# CONFIG SETUP
config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
N_JOBS = config["n_jobs"]
SEED = config["seed"]
KEEP_WARNINGS = config["warnings"]
N_HYPERPARAM_FOLDS = config["n_hyperparam_folds"]

VERBOSE_EVO = config["verbose_evo"]
DROP_FEATURES_EVO = config["drop_features_evo"]
ONLY_GENERATED_EVO = config["only_generated_evo"]
N_SEQUENTIAL_ROUNDS_EVO = config["n_sequential_rounds_evo"]
POP_SIZE_EVO = config["pop_size_evo"]
MAX_GEN_EVO = config["max_gen_evo"]
MAX_TREE_SIZE_EVO = config["max_tree_size_evo"]
INIT_MAX_DEPTH_EVO = config["init_max_depth_evo"]
PROB_INIT_TREE_EVO = config["prob_init_tree_evo"]
PROB_DELETE_TREE_EVO = config["prob_delete_tree_evo"]
PROB_MT_CROSSOVER_EVO = config["prob_mt_crossover_evo"]
ONLY_NUMERICAL_FEATURES_EVO = config["only_numerical_features_evo"]
DROP_NUMERICAL_FROM_X_EVO = config["drop_numerical_from_X_evo"]
EARLY_STOPPING_ROUNDS_EVO = config["early_stopping_rounds_evo"]
MIN_TREES_INIT_EVO = config["min_trees_init_evo"]
MAX_TREES_INIT_EVO = config["max_trees_init_evo"]
PRETUNE_COXNET_PARAMS_EVO = config["pretune_coxnet_params_evo"]
INNER_VAL_SIZE_EVO = config["inner_val_size_evo"]
COX_TOL_EVO = config["cox_tol_evo"]
COX_DOWNSAMPLE_ALPHA_EVO = config["cox_downsample_alpha_evo"]
COX_SIMPLE_EVO = config["cox_simple_evo"]
COX_MAXITER_EVO = config["cox_maxiter_evo"]
PRETUNE_COXNET_EASEOFF_EVO = config["pretune_coxnet_easeoff_evo"]
COX_ALPHA_MIN_RATIO_EVO = config["cox_alpha_min_ratio_evo"]
COX_PERFORM_FINAL_TUNING_EVO = config["cox_performing_final_tuning_evo"]
COX_FIXED_NUM_FEATS_EVO = config["cox_fixed_num_features_evo"]


EVO_KWARGS = dict(
    n_jobs=N_JOBS,
    pop_size=POP_SIZE_EVO,
    max_gens=MAX_GEN_EVO,
    max_tree_size=MAX_TREE_SIZE_EVO,
    init_max_depth=INIT_MAX_DEPTH_EVO,
    verbose=VERBOSE_EVO,
    inner_val_size=INNER_VAL_SIZE_EVO,
    drop_features=DROP_FEATURES_EVO,
    drop_numerical_from_X=DROP_NUMERICAL_FROM_X_EVO,
    only_generated=ONLY_GENERATED_EVO,
    n_sequential_rounds=N_SEQUENTIAL_ROUNDS_EVO,
    min_trees_init=MIN_TREES_INIT_EVO,
    max_trees_init=MAX_TREES_INIT_EVO,
    prob_init_tree=PROB_INIT_TREE_EVO,
    prob_delete_tree=PROB_DELETE_TREE_EVO,
    prob_mt_crossover=PROB_MT_CROSSOVER_EVO,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS_EVO,
    final_tune_n_hyperp_folds=N_HYPERPARAM_FOLDS,
    perform_final_tuning=COX_PERFORM_FINAL_TUNING_EVO,
    crossovers=[
        {"fun": node_level_crossover, "rate": 0.25},
        {"fun": subtree_crossover, "rate": 0.0},
    ],
    mutations=[
        {"fun": subtree_mutation, "rate": 0.25, "kwargs": {"max_depth": 1}},
        {"fun": one_point_mutation, "rate": 0.25},
    ],
    coeff_opts=[
        {
            "fun": coeff_mutation,
            "rate": 0.9,
            "kwargs": {
                "prob_coeff_mut": 0.5,
                "temp": 0.1,
            },
        },
    ],
)


if not KEEP_WARNINGS:
    warnings.simplefilter("ignore")


def load_runner(
    runner_name: str,
) -> "Callable[[pd.DataFrame, np.ndarray], Tuple[float, Any]]":
    if runner_name == "coxph":
        return run_coxph
    elif runner_name == "coxnet":
        return run_coxnet
    elif runner_name == "coxnet_fixedfeatnum":
        return run_coxnet_fixed_features
    elif runner_name == "ipc_ridge":
        return run_ipc_ridge
    elif runner_name == "gradboost":
        return run_gradboost
    elif runner_name == "survival_tree":
        return run_survival_tree
    elif runner_name == "randomforest":
        return run_randomforest
    elif runner_name == "sequential_evolution":
        return run_sequential_evolution
    elif runner_name == "simultaneous_evolution":
        return run_simultaneous_evolution
    elif runner_name == "bootstrapped_evolution":
        return run_bootstrapped_evolution
    else:
        raise ValueError(f"Unknown runner {runner_name}")


def run_coxph(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:
    return run_coxnet(
        X,
        y,
        n_hyperp_folds=n_hyperp_folds,
        tune_tol=1e-7,
        downsample_alphas=False,
        pretuned_params=None,
        metric=metric,
        just_coxph=True,
    )
    
def run_coxnet(
    X,
    y,
    n_hyperp_folds: int = N_HYPERPARAM_FOLDS,
    tune_tol=1e-7,
    downsample_alphas=False,
    pretuned_params=None,
    metric="cindex",
    just_coxph=False,
    max_iter=100000,
    alpha_min_ratio=0.01,
    fixed_num_features: int | None = None,
) -> Tuple[float, Any]:

    if just_coxph:
        best_model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=[0.0],
            max_iter=max_iter,
            normalize=True,
        )

        try:
            best_model.fit(X, y)
            score, _ = compute_metric_scores(
                model=best_model, X_train=X, y_train=y, metric=metric
            )
        except (ValueError, ArithmeticError):
            return 0.0, best_model
        return score, best_model

    # estimate initial alphas
    best_model = None
    # fit best model
    try:
        if pretuned_params:
            best_model = CoxnetSurvivalAnalysis(
                l1_ratio=pretuned_params["l1_ratio"],
                alphas=[pretuned_params["alpha"]],
                max_iter=max_iter,
                normalize=True,
                alpha_min_ratio=alpha_min_ratio,
            )
            best_model.fit(X, y)
        else:
            if fixed_num_features is not None:
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=0.9,
                    max_iter=max_iter,
                    alpha_min_ratio=alpha_min_ratio,
                    normalize=True,
                )
                model.fit(X, y)
                coef_t = model.coef_.T
                for i in range(len(model.alphas_)):
                    num_nonzero_coefs = np.sum(coef_t[i] != 0)
                    if num_nonzero_coefs == fixed_num_features:
                        break
                best_model = CoxnetSurvivalAnalysis(
                    l1_ratio=0.9,
                    alphas=[model.alphas_[i]],
                    max_iter=max_iter,
                    normalize=True,
                    alpha_min_ratio=alpha_min_ratio,
                )
                best_model.fit(X, y)
            else:
                best_model = tune_coxnet(
                    X,
                    y,
                    tune_tol=tune_tol,
                    n_hyperp_folds=n_hyperp_folds,
                    downsample_alphas=downsample_alphas,
                    metric=metric,
                    max_iter=max_iter,
                    alpha_min_ratio=alpha_min_ratio,
                )
        score, _ = compute_metric_scores(
            model=best_model, X_train=X, y_train=y, metric=metric
        )
    except (ValueError, ArithmeticError):
        return 0.0, best_model

    return score, best_model


def run_coxnet_fixed_features(
    X,
    y,
    n_hyperp_folds: int = N_HYPERPARAM_FOLDS,
    tune_tol=1e-7,
    downsample_alphas=False,
    pretuned_params=None,
    metric="cindex",
    fixed_num_features: int = 5,
) -> Tuple[float, Any]:
    return run_coxnet(
        X,
        y,
        n_hyperp_folds=n_hyperp_folds,
        tune_tol=tune_tol,
        downsample_alphas=downsample_alphas,
        pretuned_params=pretuned_params,
        metric=metric,
        fixed_num_features=fixed_num_features,
    )


def run_ipc_ridge(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
    tune_tol=1e-7,
    downsample_alphas=False,
    pretuned_params=None,
) -> Tuple[float, Any]:
    param_grid = {
        "alpha": [1e-2, 1e-1, 1.0, 10.0, 100.0],
        "tol": [tune_tol],
    }
    if pretuned_params:
        best_model = IPCRidge(
            alpha=pretuned_params["alpha"],
            tol=tune_tol,
        )
        best_model.fit(X, y)
    else:
        best_model = hyperparam_tuning(
            IPCRidge(),
            X,
            y,
            param_grid=param_grid,
            n_hyperp_folds=n_hyperp_folds,
            metric=metric,
        )
    score, _ = compute_metric_scores(
        model=best_model, X_train=X, y_train=y, metric=metric
    )

    return score, best_model


def run_survival_tree(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:
    param_grid = {
        "max_depth": [3, 6],
        "min_samples_split": [2, 8],
        "min_samples_leaf": [1, 4],
        "max_features": [0.5, 1.0],
        "splitter": ["best", "random"],
    }

    best_model = hyperparam_tuning(
        SurvivalTree(),
        X,
        y,
        param_grid=param_grid,
        n_hyperp_folds=n_hyperp_folds,
        metric=metric,
    )
    score, _ = compute_metric_scores(
        model=best_model, X_train=X, y_train=y, metric=metric
    )

    return score, best_model


def run_gradboost(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:

    param_grid = {
        "loss": ["coxph", "ipcwls"],
        "learning_rate": [0.1, 0.01],
        "n_estimators": [50, 250],
        "max_depth": [2, 4],
        "min_samples_split": [2, 8],
        "min_samples_leaf": [1, 4],
    }

    best_model = hyperparam_tuning(
        GradientBoostingSurvivalAnalysis(
            validation_fraction=INNER_VAL_SIZE_EVO,
            n_iter_no_change=10 if INNER_VAL_SIZE_EVO > 0.0 else None,
        ),
        X,
        y,
        param_grid=param_grid,
        n_hyperp_folds=n_hyperp_folds,
        metric=metric,
    )
    score, _ = compute_metric_scores(
        model=best_model, X_train=X, y_train=y, metric=metric
    )

    return score, best_model


def run_randomforest(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:

    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 6],
        "min_samples_split": [2, 8],
        "min_samples_leaf": [1, 4],
    }

    best_model = hyperparam_tuning(
        RandomSurvivalForest(),
        X,
        y,
        param_grid=param_grid,
        n_hyperp_folds=n_hyperp_folds,
        metric=metric,
    )
    score, _ = compute_metric_scores(
        model=best_model, X_train=X, y_train=y, metric=metric
    )

    return score, best_model


def _run_evolution(
    X,
    y,
    mode="sequential",
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:

    evo_kwargs = EVO_KWARGS.copy()

    if mode == "sequential":
        # delete prob_init_tree and prob_delete_tree
        evo_kwargs.pop("prob_init_tree")
        evo_kwargs.pop("prob_delete_tree")
        evo_kwargs.pop("min_trees_init")
        evo_kwargs.pop("max_trees_init")
        evo_kwargs.pop("prob_mt_crossover")
    elif mode == "simultaneous":
        # delete sequential
        evo_kwargs.pop("n_sequential_rounds")
    elif mode == "bootstrapped":
        evo_kwargs.pop("n_sequential_rounds")
        assert evo_kwargs["inner_val_size"] == 0.0

    evo_kwargs["final_tune_metric"] = metric

    pretuned_params = None
    if PRETUNE_COXNET_PARAMS_EVO:
        # pretuned coxnet alpha
        _, cx = run_coxnet(
            X,
            y,
            n_hyperp_folds=n_hyperp_folds,
            tune_tol=1e-7,
            downsample_alphas=False,
            metric=metric,
            alpha_min_ratio=COX_ALPHA_MIN_RATIO_EVO,
            fixed_num_features=COX_FIXED_NUM_FEATS_EVO,
        )
        # we lower alpha to give leeway to evolution
        # will anyway be tuned back in the end
        if hasattr(cx, "estimator"):
            no_nonzero_coefs = np.sum(cx.estimator.coef_ != 0)
            pretuned_params = {
                "l1_ratio": cx.estimator.l1_ratio,
                "alpha": cx.estimator.alphas_[0] * PRETUNE_COXNET_EASEOFF_EVO,
            }
        else:
            no_nonzero_coefs = np.sum(cx.coef_ != 0)
            pretuned_params = {
                "l1_ratio": cx.l1_ratio,
                "alpha": cx.alphas_[0] * PRETUNE_COXNET_EASEOFF_EVO,
            }
        print(
            "Pretuning no. nonzero coefs",
            no_nonzero_coefs,
            "(no features is",
            X.shape[1],
            ")",
        )
        print("Pretuned coxnet params", pretuned_params)

    def _run_coxnet_w_options(X, y):
        return run_coxnet(
            X,
            y,
            n_hyperp_folds=n_hyperp_folds,
            tune_tol=COX_TOL_EVO if not pretuned_params else 1e-7,
            downsample_alphas=COX_DOWNSAMPLE_ALPHA_EVO,
            pretuned_params=pretuned_params,
            metric=metric,
            just_coxph=COX_SIMPLE_EVO,
            max_iter=COX_MAXITER_EVO,
            alpha_min_ratio=COX_ALPHA_MIN_RATIO_EVO,
            fixed_num_features=COX_FIXED_NUM_FEATS_EVO,
        )

    model = SurvGeneProEstimator(
        survival_score=_run_coxnet_w_options,
        mode=mode,
        only_numerical_features=ONLY_NUMERICAL_FEATURES_EVO,
        evo_kwargs=evo_kwargs,
    )

    # now fit and score
    model.fit(X, y)
    try:
        score, _ = compute_metric_scores(
            model=model, X_train=X, y_train=y, metric=metric
        )
    except ValueError:
        return 0.0, model

    return score, model


def run_sequential_evolution(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:

    return _run_evolution(
        X,
        y,
        mode="sequential",
        metric=metric,
        n_hyperp_folds=n_hyperp_folds,
    )


def run_simultaneous_evolution(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:

    return _run_evolution(
        X,
        y,
        mode="simultaneous",
        metric=metric,
        n_hyperp_folds=n_hyperp_folds,
    )


def run_bootstrapped_evolution(
    X,
    y,
    metric="cindex",
    n_hyperp_folds=N_HYPERPARAM_FOLDS,
) -> Tuple[float, Any]:

    return _run_evolution(
        X,
        y,
        mode="bootstrapped",
        metric=metric,
        n_hyperp_folds=n_hyperp_folds,
    )

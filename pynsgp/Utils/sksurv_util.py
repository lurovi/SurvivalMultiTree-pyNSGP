from typing import List, Dict, Any, Tuple

import warnings
import numpy as np

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer, as_cumulative_dynamic_auc_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    StratifiedKFold,
)


COXNET_L1_RATIO = 0.9
ALPHA_MIN_RATIO = 0.1

def hyperparam_tuning(
    model, X, y, param_grid: Dict[str, List[Any]], n_hyperp_folds=3,
    metric="cindex",
    n_jobs=1,
) -> "Model":
    
     # times follow https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
    # see gbsg_times
    lower, upper = np.percentile([y_i[1] for y_i in y], [10, 90])
    times = np.arange(lower, upper + 1)
    
    folds = StratifiedKFold(
        n_splits=n_hyperp_folds, shuffle=True,
        random_state=np.random.randint(9999),
    ).split(X, [y_i[0] for y_i in y])
    
    metric_aware_model = model
    if metric == "cindex_ipcw":
        metric_aware_model = as_concordance_index_ipcw_scorer(
            model, tau=times[-1],
        )
    elif metric == "mean_auc":
        metric_aware_model = as_cumulative_dynamic_auc_scorer(
            model, times=times,
        )
    # fix param grid
    if metric != "cindex":
        # change all keys by prepending "estimator__"
        param_grid = {
            f"estimator__{k}": v for k, v in param_grid.items()
        }
    
    gcv = GridSearchCV(
        metric_aware_model,
        param_grid=param_grid,
        cv=folds,
        error_score=0.0,
        n_jobs=n_jobs,
        refit=True,
    ).fit(X, y)

    best_model = gcv.best_estimator_
    
    return best_model

def tune_coxnet(
    X, y,
    tune_tol: float = 1e-7,
    n_hyperp_folds: int = 3,
    downsample_alphas: bool = False,
    metric="cindex",
    max_iter=100000,
    alpha_min_ratio=ALPHA_MIN_RATIO,
) -> "CoxnetSurvivalAnalysis":
    
    try:
        if len(X.columns) < 2:
            estimated_alphas = [0.0]
        else:
            coxnet = CoxnetSurvivalAnalysis(
                l1_ratio=COXNET_L1_RATIO,
                normalize=True,
                max_iter=max_iter,
                alpha_min_ratio=alpha_min_ratio,
            )
            coxnet.fit(X.astype(float), y)
            estimated_alphas = coxnet.alphas_
            
        if downsample_alphas and len(estimated_alphas) > 10:
            # pick equally-spaced alphas
            estimated_alphas = estimated_alphas[::len(estimated_alphas) // 10]
        
        best_model = hyperparam_tuning(
            CoxnetSurvivalAnalysis(
                tol=tune_tol,
                normalize=True,
                max_iter=max_iter,
                alpha_min_ratio=alpha_min_ratio,
            ),
            X.astype(float),
            y,
            param_grid={
                "l1_ratio": [0.1, 0.5, 0.9],
                "alphas": [[v] for v in estimated_alphas]
            },
            n_hyperp_folds=n_hyperp_folds,
            n_jobs=1,  # force it to one to keep parallelization only on evolution
            metric=metric,
        )
    except (ValueError, ArithmeticError) as e:
        warnings.warn(f"Error in Coxnet fitting {e}")
        print("Error in Coxnet fitting", e)
        return None
        
    return best_model


def coxnet_at_n_coeffs(
    X, y,
    n_coefs: int,
    tune_tol: float = 1e-7,
    n_hyperp_folds: int = 3,
    max_iter: int = 100000,
) -> "CoxnetSurvivalAnalysis":
    coxnet = CoxnetSurvivalAnalysis(
        l1_ratio=COXNET_L1_RATIO,
        normalize=True,
        max_iter=max_iter,
    )
    coxnet.fit(X.astype(float), y)
    
    
from sksurv.metrics import concordance_index_ipcw

from pynsgp.Utils.data import load_dataset, preproc_dataset
from sklearn.model_selection import train_test_split
from genepro.node_impl import *
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP
import numpy as np
import random


def get_coxnet_at_k_coefs(cox: CoxnetSurvivalAnalysis, k: int) -> tuple:
    coef_t = cox.coef_.T
    alphas_n_coefs_at_k = []
    for i, alpha in enumerate(cox.alphas_):
        num_nonzero_coefs = np.sum(coef_t[i] != 0)
        if num_nonzero_coefs == k:
            alphas_n_coefs_at_k.append((alpha, coef_t[i]))
        if num_nonzero_coefs > k:
            break

    #mask = np.ones(len(coefs))
    #for i, (alpha, coefs) in enumerate(alphas_n_coefs_at_k):
    #    nonzero_positions = coefs / np.where(coefs == 0, 1, coefs)
    #    if mask is None:
    #        mask = nonzero_positions
    #    else:
    #        mask = mask * nonzero_positions
    #if np.sum(mask) != k:
    # take the median coefs
    num_options = len(alphas_n_coefs_at_k)
    alpha = alphas_n_coefs_at_k[num_options // 2][0]
    coefs = alphas_n_coefs_at_k[num_options // 2][1]
    #else:
    #    print("Taking mean")
    #    alpha = np.mean([alpha for alpha, _ in alphas_n_coefs_at_k])
    #    coefs = np.mean([coefs for _, coefs in alphas_n_coefs_at_k], axis=0)

    return alpha, coefs


def set_random_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)


def load_preprocess_data(
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        test_size
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_dataset(dataset_name=dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=[y_i[0] for y_i in y],
        random_state=random_state
    )

    X_train, y_train, col_transformer = preproc_dataset(
        X_train,
        y_train,
        name=dataset_name,
        drop_corr_threhsold=corr_drop_threshold,
        scale_numerical=scale_numerical
    )

    X_test, y_test, _ = preproc_dataset(
        X_test,
        y_test,
        col_transformer=col_transformer,
        name=dataset_name,
        scale_numerical=scale_numerical
    )

    return X_train, X_test, y_train, y_test


def run_gridsearch_cox_net(
        linear_model,
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        test_size
) -> None:
    if linear_model not in ('coxnet', 'coxph'):
        raise AttributeError(f'Unrecognized linear model {linear_model}.')

    possible_linear_models = {
        'coxnet': CoxnetSurvivalAnalysis(),
        'coxph': CoxPHSurvivalAnalysis()
    }

    possible_grids = {
        'coxnet': {
            'n_alphas': [10, 100, 200, 300],
            'alpha_min_ratio': [0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.2, 0.5, 0.8, 0.9]
        },
        'coxph': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'ties': ['breslow', 'efron'],
            'n_iter': [100, 500, 1000]
        }
    }

    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=corr_drop_threshold,
        scale_numerical=scale_numerical,
        random_state=random_state,
        dataset_name=dataset_name,
        test_size=test_size
    )


def run_cox_net(
        linear_model,
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        test_size
) -> None:
    if linear_model not in ('coxnet', 'coxph'):
        raise AttributeError(f'Unrecognized linear model {linear_model}.')

    if linear_model == 'coxnet':
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.9,
            alpha_min_ratio=0.1,
            max_iter=1000000,
        )
    elif linear_model == 'coxph':
        model = CoxPHSurvivalAnalysis(
            alpha=0.1,
            n_iter=100000
        )

    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=corr_drop_threshold,
        scale_numerical=scale_numerical,
        random_state=random_state,
        dataset_name=dataset_name,
        test_size=test_size
    )

    lower, upper = np.percentile([y_i[1] for y_i in y_train], [5, 95])
    times = np.arange(lower, upper)
    tau = times[-1]

    largest_value = 1e+8
    largest_safe_value = 1e+4

    model.fit(X_train, y_train)
    for k in range(1, 100 + 1):
        try:
            result = get_coxnet_at_k_coefs(model, k)
            print('Num Distinct Raw Features ', k)
            print(result)
            risk_scores = model.predict(X_train, alpha=float(result[0]))
            risk_scores.clip(-largest_value, largest_value, out=risk_scores)
            error = -1.0 * concordance_index_ipcw(
                survival_train=y_train, survival_test=y_train, estimate=risk_scores, tau=tau
            )[0]
            print('q1: ', error, ' q2: ', k)
            print()
        except Exception:
            break


def run_evolution(
        crossovers,
        mutations,
        coeff_opts,
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        test_size,
        pop_size,
        num_gen,
) -> None:
    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=corr_drop_threshold,
        scale_numerical=scale_numerical,
        random_state=random_state,
        dataset_name=dataset_name,
        test_size=test_size
    )

    nsgp = NSGP(
        crossovers=crossovers,
        mutations=mutations,
        coeff_opts=coeff_opts,
        pop_size=pop_size,
        max_generations=num_gen,
        max_evaluations=-1,
        max_time=-1,
        functions=[Plus(), Minus(), Times(), Div(), Log(), Exp(), Power(), Sin(), Cos()], # più nodi
        use_erc=True,
        error_metric='cindex_ipcw',
        size_metric='distinct_raw_features',
        prob_delete_tree=0.05,
        prob_init_tree=0.1,
        prob_mt_crossover=0.1, # diminuire
        initialization_max_tree_height=4,
        min_depth=2,
        tournament_size=3,
        max_tree_size=40,
        partition_features=False,
        min_trees_init=3,
        max_trees_init=6, # eliminare gli alberi dal multitree che hanno coefficiente nullo
        penalize_duplicates=True,
        verbose=True,
        alpha=1e-6,
        n_iter=100
    )
    nsgp.fit(X_train, y_train)

    # Obtain the front of non-dominated solutions (according to the training set)
    front = nsgp.get_front()
    print('len front:', len(front))
    for solution in front:
        print(solution.get_readable_repr())

    print(str(nsgp))

import os.path
import time

import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from pymoo.indicators.hv import HV

from pynsgp.Utils.pickle_persist import compress_pickle
from pynsgp.Utils.data import load_dataset, cox_net_path_string, nsgp_path_string, simple_onehot_and_nan_drop
from sklearn.model_selection import train_test_split
from genepro.node_impl import *
from sksurv.linear_model import CoxnetSurvivalAnalysis
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP
import numpy as np
import random


def get_coxnet_at_k_coefs(cox: CoxnetSurvivalAnalysis, k: int) -> tuple | None:
    coef_t = cox.coef_.T
    alphas_n_coefs_at_k = []
    for i, alpha in enumerate(cox.alphas_):
        num_nonzero_coefs = np.sum(coef_t[i] != 0)
        if num_nonzero_coefs == k:
            alphas_n_coefs_at_k.append((alpha, coef_t[i]))
        if num_nonzero_coefs > k:
            break

    num_options = len(alphas_n_coefs_at_k)
    if num_options == 0:
        return None
    alpha = alphas_n_coefs_at_k[num_options // 2][0]
    coefs = alphas_n_coefs_at_k[num_options // 2][1]

    return alpha, coefs


def set_random_seed(random_state: int, square: bool = True):
    if square:
        random.seed(random_state ** 2)
        np.random.seed(random_state ** 2)
    else:
        random.seed(random_state)
        np.random.seed(random_state)


def load_preprocess_data(
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        test_size,
        square = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_dataset(dataset_name=dataset_name)

    if square:
        random_state = random_state ** 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=[y_i[0] for y_i in y],
        random_state=random_state
    )

    #X_train, y_train, col_transformer = preproc_dataset(
    #    X_train,
    #    y_train,
    #    name=dataset_name,
    #    drop_corr_threhsold=corr_drop_threshold,
    #    scale_numerical=scale_numerical
    #)

    #X_test, y_test, _ = preproc_dataset(
    #    X_test,
    #    y_test,
    #    col_transformer=col_transformer,
    #    name=dataset_name,
    #    scale_numerical=scale_numerical
    #)

    X_train, y_train, col_transformer = simple_onehot_and_nan_drop(X_train, y_train)
    X_test, y_test, _ = simple_onehot_and_nan_drop(X_test, y_test, col_transformer)

    return X_train.to_numpy(), X_test.to_numpy(), y_train, y_test


def run_cox_net(
        results_path,
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        normalize,
        test_size,
        n_alphas,
        l1_ratio,
        alpha_min_ratio,
        max_iter,
        verbose
) -> None:
    final_path = cox_net_path_string(
        base_path=results_path,
        method='coxnet',
        dataset_name=dataset_name,
        normalize=normalize,
        test_size=test_size,
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter
    )
    if not os.path.isdir(final_path):
        os.makedirs(final_path, exist_ok=True)

    model_file_name: str = f'model_seed{random_state}'
    output_file_name: str = f'output_seed{random_state}.csv'

    model = CoxnetSurvivalAnalysis(
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter,
        verbose=verbose,
        normalize=normalize,
        fit_baseline_model=False
    )

    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=corr_drop_threshold,
        scale_numerical=scale_numerical,
        random_state=random_state,
        dataset_name=dataset_name,
        test_size=test_size
    )

    lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
    train_times = np.arange(lower, upper)
    tau_train = train_times[-1]

    lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
    test_times = np.arange(lower, upper)
    tau_test = test_times[-1]

    n_features = X_train.shape[1]

    largest_value = 1e+8

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    compress_pickle(os.path.join(final_path, model_file_name), model)

    output_data = {"DistinctRawFeatures": [], "Alpha": [], "TrainError": [], "TestError": [],
                   "TrainHV": [], "TestHV": [],
                   "TrainTime": [], "TrainEvalTime": [], "TestEvalTime": []}

    ref_point = np.array([0.0, n_features])

    for k in range(1, n_features + 1):
        result = get_coxnet_at_k_coefs(model, k)
        if result is None:
            continue
        output_data["TrainTime"].append(training_time)
        alpha = float(result[0])
        output_data['Alpha'].append(alpha)
        output_data['DistinctRawFeatures'].append(k)

        if verbose:
            print('Num Distinct Raw Features ', k)
            print(result)

        start_time = time.time()
        risk_scores = model.predict(X_train, alpha=alpha)
        end_time = time.time()
        train_eval_time = end_time - start_time
        output_data["TrainEvalTime"].append(train_eval_time)
        risk_scores.clip(-largest_value, largest_value, out=risk_scores)
        try:
            error = -1.0 * concordance_index_ipcw(
                survival_train=y_train, survival_test=y_train, estimate=risk_scores, tau=tau_train
            )[0]
        except ValueError:
            error = 0.0
        output_data['TrainError'].append(error)

        start_time = time.time()
        risk_scores = model.predict(X_test, alpha=alpha)
        end_time = time.time()
        test_eval_time = end_time - start_time
        output_data["TestEvalTime"].append(test_eval_time)
        risk_scores.clip(-largest_value, largest_value, out=risk_scores)
        try:
            error = -1.0 * concordance_index_ipcw(
                survival_train=y_train, survival_test=y_test, estimate=risk_scores, tau=tau_test
            )[0]
        except ValueError:
            error = 0.0
        output_data['TestError'].append(error)

        if verbose:
            print('q1: ', error, ' q2: ', k)
            print()

    train_pareto = np.array([[error, size] for error, size in zip(output_data["TrainError"], output_data["DistinctRawFeatures"])])
    train_hv_value = HV(ref_point)(train_pareto)
    output_data["TrainHV"].extend([train_hv_value] * len(output_data["DistinctRawFeatures"]))

    test_pareto = np.array([[error, size] for error, size in zip(output_data["TestError"], output_data["DistinctRawFeatures"])])
    test_hv_value = HV(ref_point)(test_pareto)
    output_data["TestHV"].extend([test_hv_value] * len(output_data["DistinctRawFeatures"]))

    pd.DataFrame(output_data).to_csv(os.path.join(final_path, output_file_name), sep=',', header=True, index=False)


def run_evolution(
        results_path,
        crossovers,
        mutations,
        coeff_opts,
        corr_drop_threshold,
        scale_numerical,
        random_state,
        dataset_name,
        normalize,
        test_size,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio,
        max_iter,
        verbose
) -> None:
    final_path = nsgp_path_string(
        base_path=results_path,
        method='nsgp',
        dataset_name=dataset_name,
        normalize=normalize,
        test_size=test_size,
        pop_size=pop_size,
        num_gen=num_gen,
        max_size=max_size,
        min_depth=min_depth,
        init_max_height=init_max_height,
        tournament_size=tournament_size,
        min_trees_init=min_trees_init,
        max_trees_init=max_trees_init,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter
    )
    if not os.path.isdir(final_path):
        os.makedirs(final_path, exist_ok=True)

    pareto_file_name: str = f'pareto_seed{random_state}'
    output_file_name: str = f'output_seed{random_state}.csv'

    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=corr_drop_threshold,
        scale_numerical=scale_numerical,
        random_state=random_state,
        dataset_name=dataset_name,
        test_size=test_size
    )

    nsgp = NSGP(
        path=final_path,
        pareto_file_name=pareto_file_name,
        output_file_name=output_file_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        crossovers=crossovers,
        mutations=mutations,
        coeff_opts=coeff_opts,
        pop_size=pop_size,
        max_generations=num_gen,
        max_evaluations=-1,
        max_time=-1,
        functions=[Plus(), Minus(), Times(), AnalyticQuotient(), Square(), Log()],
        use_erc=True,
        error_metric='cindex_ipcw',
        size_metric='distinct_raw_features',
        prob_delete_tree=0.05,
        prob_init_tree=0.05,
        prob_mt_crossover=0.1,
        initialization_max_tree_height=init_max_height,
        min_depth=min_depth,
        tournament_size=tournament_size,
        max_tree_size=max_size,
        partition_features=False,
        min_trees_init=min_trees_init,
        max_trees_init=max_trees_init,
        penalize_duplicates=True,
        verbose=verbose,
        alpha=alpha,
        n_iter=max_iter,
        l1_ratio=l1_ratio,
        normalize=normalize
    )
    nsgp.fit(X_train, y_train)

    if verbose:
        # Obtain the front of non-dominated solutions (according to the training set)
        print(str(nsgp))

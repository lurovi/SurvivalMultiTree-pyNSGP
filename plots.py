import json
import statistics
from typing import Any

import os.path

from sklearn.pipeline import Pipeline
from sksurv.metrics import as_concordance_index_ipcw_scorer
from pynsgp.Utils.data import SimpleStdScalerOneHot

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sklearn.model_selection import GridSearchCV

import fastplot
import numpy as np
import pandas as pd
import seaborn as sns
import os

from pymoo.indicators.hv import HV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.tree import SurvivalTree

from methods import set_random_seed, load_preprocess_data, get_coxnet_at_k_coefs
from pynsgp.Utils.pickle_persist import decompress_pickle, decompress_dill
from pynsgp.Utils.data import load_dataset, nsgp_path_string, cox_net_path_string, survival_ensemble_tree_path_string, \
    simple_basic_cast_and_nan_drop
from pynsgp.Utils.stats import is_mannwhitneyu_passed, is_kruskalwallis_passed, perform_mannwhitneyu_holm_bonferroni

import warnings
import yaml

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def filter_non_dominated_only(pairs_pareto):
    new_pareto = []

    for i, pair in enumerate(pairs_pareto):
        error, size = pair[0], pair[1]
        dominated = False
        for j, other_pair in enumerate(pairs_pareto):
            other_error, other_size = other_pair[0], other_pair[1]
            if (other_error <= error and other_size <= size) and (other_error < error or other_size < size):
                dominated = True
                break
        if not dominated:
            new_pareto.append(pair)

    return new_pareto


def callback_scatter_plot(plt, coxnet_n_features_list, coxnet_errors_list, nsgp_n_features_list, nsgp_errors_list):
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')

    coxnet_points = list(zip(coxnet_errors_list, coxnet_n_features_list))
    nsgp_points = list(zip(nsgp_errors_list, nsgp_n_features_list))

    for err, size in coxnet_points:
        # green
        ax.scatter(err, size, c='#31AB0C', marker='o', s=100, edgecolor='black', linewidth=0.8)

    for err, size in nsgp_points:
        # blue
        ax.scatter(err, size, c='#283ADF', marker='v', s=100, edgecolor='black', linewidth=0.8)

    ax.set_ylim(0, 21)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ax.set_xlim(-1.0, -0.0)
    ax.set_xticks([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.set_xlabel('Error')
    ax.set_ylabel('Number of Features')
    # ax.set_title(f'Methods Pareto Front ({metric} across all datasets and repetitions)')
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)


def scatter_plot(
        dataset,
        seed,
        split_type,
        coxnet_n_features_list,
        coxnet_errors_list,
        nsgp_n_features_list,
        nsgp_errors_list
):
    # PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

    preamble = r'''
        \usepackage{amsmath}
        \usepackage{libertine}
        '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    fastplot.plot(
        None, f'pareto_{dataset}_{split_type}_seed{seed}.pdf',
        mode='callback',
        callback=lambda plt: callback_scatter_plot(plt, coxnet_n_features_list, coxnet_errors_list, nsgp_n_features_list, nsgp_errors_list),
        style='latex', **PLOT_ARGS
    )


def take_formula(data, pareto, n_features_to_consider):
    last_pareto = pareto[-1]
    train_errors = list(data['TrainParetoObj1'])[-1]
    test_errors = list(data['TestParetoObj1'])[-1]
    n_features = list(data['ParetoObj2'])[-1]
    train_errors = [float(abc) for abc in train_errors.split(' ')]
    test_errors = [float(abc) for abc in test_errors.split(' ')]
    n_features = [float(abc) for abc in n_features.split(' ')]

    ind = n_features.index(n_features_to_consider)
    multi_tree = last_pareto[ind]
    train_error = train_errors[ind]
    test_error = test_errors[ind]

    latex_expr = multi_tree.latex_expression(round_precision=3, perform_simplification=False)

    return train_error, test_error, latex_expr


def read_coxnet(
    base_path: str,
    method: str,
    dataset_name: str,
    normalize: bool,
    test_size: float,
    n_alphas: int,
    l1_ratio: float,
    alpha_min_ratio: float,
    max_iter: int,
    seed: int
):
    path = cox_net_path_string(
        base_path=base_path,
        method=method,
        dataset_name=dataset_name,
        normalize=normalize,
        test_size=test_size,
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter
    )
    data = pd.read_csv(os.path.join(path, f'output_seed{seed}.csv'), sep=',')
    model = decompress_pickle(os.path.join(path, f'model_seed{seed}.pbz2'))
    return data, model


def read_survivalensembletree(
    base_path: str,
    method: str,
    dataset_name: str,
    normalize: bool,
    test_size: float,
    n_max_depths: int,
    n_folds: int,
    seed: int
):
    path = survival_ensemble_tree_path_string(
        base_path=base_path,
        method=method,
        dataset_name=dataset_name,
        normalize=normalize,
        test_size=test_size,
        n_max_depths=n_max_depths,
        n_folds=n_folds,
    )
    data = pd.read_csv(os.path.join(path, f'output_seed{seed}.csv'), sep=',')
    model = decompress_dill(os.path.join(path, f'model_seed{seed}.pbz2'))
    return data, model


def read_nsgp(
        base_path: str,
        method: str,
        dataset_name: str,
        normalize: bool,
        test_size: float,
        pop_size: int,
        num_gen: int,
        max_size: int,
        min_depth: int,
        init_max_height: int,
        tournament_size: int,
        min_trees_init: int,
        max_trees_init: int,
        alpha: float,
        l1_ratio: float,
        max_iter: int,
        seed: int,
        load_pareto: bool
):
    path = nsgp_path_string(
        base_path=base_path,
        method=method,
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
    data = pd.read_csv(os.path.join(path, f'output_seed{seed}.csv'), sep=',')
    pareto = decompress_pickle(os.path.join(path, f'pareto_seed{seed}.pbz2')) if load_pareto else None
    return data, pareto


def create_survival_function(
        base_path,
        test_size,
        dataset_name,
        seed,
        n_alphas,
        l1_ratio,
        alpha_min_ratio,
        max_iter,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        n_max_depths_st,
        n_folds_st,
        n_max_depths_gb,
        n_folds_gb,
        n_max_depths_rf,
        n_folds_rf,
):

    data: dict[str, list[float]] = {}

    param_grid_survival_tree = {
        "model__estimator__min_samples_split": [2, 5, 8],
        "model__estimator__min_samples_leaf": [1, 4],
        "model__estimator__max_features": [0.5, 1.0],
        "model__estimator__splitter": ["best", "random"],
    }

    param_grid_gradient_boost = {
        "model__estimator__loss": ["coxph"],
        "model__estimator__learning_rate": [0.1, 0.01],
        "model__estimator__n_estimators": [50, 250],
        "model__estimator__min_samples_split": [2, 5, 8],
        "model__estimator__min_samples_leaf": [1, 4],
        "model__estimator__max_depth": [3, 6, 9],
    }

    param_grid_random_forest = {
        "model__estimator__n_estimators": [50, 250],
        "model__estimator__min_samples_split": [2, 5, 8],
        "model__estimator__min_samples_leaf": [1, 4],
        "model__estimator__max_depth": [3, 6, 9],
    }

    # ======================================
    # NSGP
    # ======================================

    nsgpd, pareto_nsgpd = read_nsgp(
        base_path=base_path,
        method='nsgp',
        dataset_name=dataset_name,
        normalize=True,
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
        l1_ratio=l1_ratio_nsgp,
        max_iter=max_iter_nsgp,
        seed=seed,
        load_pareto=True
    )

    last_pareto = pareto_nsgpd[-1]
    n_features = list(nsgpd['ParetoObj2'])[-1]
    n_features = [float(abc) for abc in n_features.split(' ')]
    i = np.argmax(n_features)
    tree = last_pareto[i]

    set_random_seed(seed)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=0.98,
        scale_numerical=True,
        random_state=seed,
        dataset_name=dataset_name,
        test_size=test_size
    )
    largest_value = 1e+8
    output = tree(X_train)
    output.clip(-largest_value, largest_value, out=output)
    cox = CoxnetSurvivalAnalysis(
        n_alphas=1,
        alphas=[alpha],
        max_iter=max_iter_nsgp,
        l1_ratio=l1_ratio,
        normalize=True,
        verbose=False,
        fit_baseline_model=True
    )
    cox.fit(output, y_train)
    output = tree(X_test)
    output.clip(-largest_value, largest_value, out=output)
    data['nsgp_probs'] = np.median(cox.predict_survival_function(output, alpha=alpha, return_array=True), axis=0).tolist()
    data['nsgp_times'] = cox.unique_times_.tolist()

    print('NSGP DONE')

    # ======================================
    # COXNET
    # ======================================

    set_random_seed(seed)

    X_train, X_test, y_train, y_test = load_preprocess_data(
        corr_drop_threshold=0.98,
        scale_numerical=True,
        random_state=seed,
        dataset_name=dataset_name,
        test_size=test_size
    )

    n_features = X_train.shape[1]

    cox = CoxnetSurvivalAnalysis(
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter,
        verbose=False,
        normalize=True,
        fit_baseline_model=True
    )
    cox.fit(X_train, y_train)

    for k in range(1, n_features + 1):
        result = get_coxnet_at_k_coefs(cox, k)
        if result is None:
            continue
        alpha = float(result[0])

    data['coxnet_probs'] = np.median(cox.predict_survival_function(X_test, alpha=alpha, return_array=True), axis=0).tolist()
    data['coxnet_times'] = cox.unique_times_.tolist()

    print('COXNET DONE')

    # ======================================
    # SURVIVAL TREE
    # ======================================

    X, y = load_dataset(dataset_name=dataset_name)
    X, y = simple_basic_cast_and_nan_drop(X, y)

    random_state = seed ** 2
    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=[y_i[0] for y_i in y],
        random_state=random_state
    )

    n_features = X_train.shape[1]
    largest_value = 1e+8

    lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
    train_times = np.arange(lower, upper)
    tau_train = train_times[-1]

    lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
    test_times = np.arange(lower, upper)
    tau_test = test_times[-1]

    folds = StratifiedKFold(
        n_splits=n_folds_st, shuffle=True,
        random_state=random_state + 50,
    ).split(X_train, [y_i[0] for y_i in y_train])

    param_grid = param_grid_survival_tree
    model = SurvivalTree(max_depth=n_max_depths_st, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ('scaler', SimpleStdScalerOneHot(normalize=True)),
            ('model', as_concordance_index_ipcw_scorer(model, tau=tau_train))
        ]
    )

    gcv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=folds,
        error_score=0.0,
        n_jobs=-1,
        refit=True,
        verbose=False
    ).fit(X_train, y_train)

    actual_model = gcv.best_estimator_['model'].estimator
    actual_scaler = gcv.best_estimator_['scaler']

    data['survivaltree_probs'] = np.median(actual_model.predict_survival_function(actual_scaler.transform(X_test), return_array=True), axis=0).tolist()
    data['survivaltree_times'] = actual_model.unique_times_.tolist()

    print('SURVIVALTREE DONE')

    # ======================================
    # GRADIENT BOOST
    # ======================================

    X, y = load_dataset(dataset_name=dataset_name)
    X, y = simple_basic_cast_and_nan_drop(X, y)

    random_state = seed ** 2
    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=[y_i[0] for y_i in y],
        random_state=random_state
    )

    n_features = X_train.shape[1]
    largest_value = 1e+8

    lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
    train_times = np.arange(lower, upper)
    tau_train = train_times[-1]

    lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
    test_times = np.arange(lower, upper)
    tau_test = test_times[-1]

    folds = StratifiedKFold(
        n_splits=n_folds_gb, shuffle=True,
        random_state=random_state + 50,
    ).split(X_train, [y_i[0] for y_i in y_train])

    param_grid = param_grid_gradient_boost
    model = GradientBoostingSurvivalAnalysis(random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ('scaler', SimpleStdScalerOneHot(normalize=True)),
            ('model', as_concordance_index_ipcw_scorer(model, tau=tau_train))
        ]
    )

    gcv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=folds,
        error_score=0.0,
        n_jobs=-1,
        refit=True,
        verbose=False
    ).fit(X_train, y_train)

    actual_model = gcv.best_estimator_['model'].estimator
    actual_scaler = gcv.best_estimator_['scaler']

    data['gradientboost_probs'] = np.median(actual_model.predict_survival_function(actual_scaler.transform(X_test), return_array=True), axis=0).tolist()
    data['gradientboost_times'] = actual_model.unique_times_.tolist()

    print('GRADIENTBOOST DONE')

    # ======================================
    # RANDOM FOREST
    # ======================================

    X, y = load_dataset(dataset_name=dataset_name)
    X, y = simple_basic_cast_and_nan_drop(X, y)

    random_state = seed ** 2
    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=[y_i[0] for y_i in y],
        random_state=random_state
    )

    n_features = X_train.shape[1]
    largest_value = 1e+8

    lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
    train_times = np.arange(lower, upper)
    tau_train = train_times[-1]

    lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
    test_times = np.arange(lower, upper)
    tau_test = test_times[-1]

    folds = StratifiedKFold(
        n_splits=n_folds_rf, shuffle=True,
        random_state=random_state + 50,
    ).split(X_train, [y_i[0] for y_i in y_train])

    param_grid = param_grid_random_forest
    model = RandomSurvivalForest(random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ('scaler', SimpleStdScalerOneHot(normalize=True)),
            ('model', as_concordance_index_ipcw_scorer(model, tau=tau_train))
        ]
    )

    gcv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=folds,
        error_score=0.0,
        n_jobs=-1,
        refit=True,
        verbose=False
    ).fit(X_train, y_train)

    actual_model = gcv.best_estimator_['model'].estimator
    actual_scaler = gcv.best_estimator_['scaler']

    data['randomforest_probs'] = np.median(actual_model.predict_survival_function(actual_scaler.transform(X_test), return_array=True), axis=0).tolist()
    data['randomforest_times'] = actual_model.unique_times_.tolist()

    print('RANDOMFOREST DONE')

    with open(f'survival_function_data_seed{seed}.json', 'w') as f:
        json.dump(data, f, indent=4)


def create_survival_function_all(
        seed_range,
        base_path,
        test_size,
        dataset_name,
        seed,
        n_alphas,
        l1_ratio,
        alpha_min_ratio,
        max_iter,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        n_max_depths_st,
        n_folds_st,
        n_max_depths_gb,
        n_folds_gb,
        n_max_depths_rf,
        n_folds_rf,
):

    # seed method timestep values
    data: dict[str, dict[str, dict[str, list[float]]]] = {}

    param_grid_survival_tree = {
        "model__estimator__min_samples_split": [2, 5, 8],
        "model__estimator__min_samples_leaf": [1, 4],
        "model__estimator__max_features": [0.5, 1.0],
        "model__estimator__splitter": ["best", "random"],
    }

    param_grid_gradient_boost = {
        "model__estimator__loss": ["coxph"],
        "model__estimator__learning_rate": [0.1, 0.01],
        "model__estimator__n_estimators": [50, 250],
        "model__estimator__min_samples_split": [2, 5, 8],
        "model__estimator__min_samples_leaf": [1, 4],
        "model__estimator__max_depth": [3, 6, 9],
    }

    param_grid_random_forest = {
        "model__estimator__n_estimators": [50, 250],
        "model__estimator__min_samples_split": [2, 5, 8],
        "model__estimator__min_samples_leaf": [1, 4],
        "model__estimator__max_depth": [3, 6, 9],
    }

    for seed in seed_range:
        print(seed)
        data[str(seed)] = {'nsgp': {}, 'coxnet': {}, 'survivaltree': {}, 'gradientboost': {}, 'randomforest': {}}

        # ======================================
        # NSGP
        # ======================================

        nsgpd, pareto_nsgpd = read_nsgp(
            base_path=base_path,
            method='nsgp',
            dataset_name=dataset_name,
            normalize=True,
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
            l1_ratio=l1_ratio_nsgp,
            max_iter=max_iter_nsgp,
            seed=seed,
            load_pareto=True
        )

        last_pareto = pareto_nsgpd[-1]
        n_features = list(nsgpd['ParetoObj2'])[-1]
        n_features = [float(abc) for abc in n_features.split(' ')]
        i = np.argmax(n_features)
        tree = last_pareto[i]

        set_random_seed(seed)

        X_train, X_test, y_train, y_test = load_preprocess_data(
            corr_drop_threshold=0.98,
            scale_numerical=True,
            random_state=seed,
            dataset_name=dataset_name,
            test_size=test_size
        )
        largest_value = 1e+8
        output = tree(X_train)
        output.clip(-largest_value, largest_value, out=output)
        cox = CoxnetSurvivalAnalysis(
            n_alphas=1,
            alphas=[alpha],
            max_iter=max_iter_nsgp,
            l1_ratio=l1_ratio,
            normalize=True,
            verbose=False,
            fit_baseline_model=True
        )
        cox.fit(output, y_train)
        output = tree(X_test)
        output.clip(-largest_value, largest_value, out=output)

        times = cox.unique_times_.tolist()
        probs = cox.predict_survival_function(output, alpha=alpha, return_array=True).T.tolist()
        for jj in range(len(times)):
            curr_time = times[jj]
            curr_prob = probs[jj]
            if str(curr_time) not in data[str(seed)]['nsgp']:
                data[str(seed)]['nsgp'][str(curr_time)] = []
            data[str(seed)]['nsgp'][str(curr_time)].extend(curr_prob)

        # ======================================
        # COXNET
        # ======================================

        set_random_seed(seed)

        X_train, X_test, y_train, y_test = load_preprocess_data(
            corr_drop_threshold=0.98,
            scale_numerical=True,
            random_state=seed,
            dataset_name=dataset_name,
            test_size=test_size
        )

        n_features = X_train.shape[1]

        cox = CoxnetSurvivalAnalysis(
            n_alphas=n_alphas,
            l1_ratio=l1_ratio,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter,
            verbose=False,
            normalize=True,
            fit_baseline_model=True
        )
        cox.fit(X_train, y_train)

        for k in range(1, n_features + 1):
            result = get_coxnet_at_k_coefs(cox, k)
            if result is None:
                continue
            alpha = float(result[0])

        times = cox.unique_times_.tolist()
        probs = cox.predict_survival_function(X_test, alpha=alpha, return_array=True).T.tolist()
        for jj in range(len(times)):
            curr_time = times[jj]
            curr_prob = probs[jj]
            if str(curr_time) not in data[str(seed)]['coxnet']:
                data[str(seed)]['coxnet'][str(curr_time)] = []
            data[str(seed)]['coxnet'][str(curr_time)].extend(curr_prob)

        # ======================================
        # SURVIVAL TREE
        # ======================================

        X, y = load_dataset(dataset_name=dataset_name)
        X, y = simple_basic_cast_and_nan_drop(X, y)

        random_state = seed ** 2
        set_random_seed(random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=[y_i[0] for y_i in y],
            random_state=random_state
        )

        n_features = X_train.shape[1]
        largest_value = 1e+8

        lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
        train_times = np.arange(lower, upper)
        tau_train = train_times[-1]

        lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
        test_times = np.arange(lower, upper)
        tau_test = test_times[-1]

        folds = StratifiedKFold(
            n_splits=n_folds_st, shuffle=True,
            random_state=random_state + 50,
        ).split(X_train, [y_i[0] for y_i in y_train])

        param_grid = param_grid_survival_tree
        model = SurvivalTree(max_depth=n_max_depths_st, random_state=random_state)

        pipeline = Pipeline(
            steps=[
                ('scaler', SimpleStdScalerOneHot(normalize=True)),
                ('model', as_concordance_index_ipcw_scorer(model, tau=tau_train))
            ]
        )

        gcv = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=folds,
            error_score=0.0,
            n_jobs=-1,
            refit=True,
            verbose=False
        ).fit(X_train, y_train)

        actual_model = gcv.best_estimator_['model'].estimator
        actual_scaler = gcv.best_estimator_['scaler']

        times = actual_model.unique_times_.tolist()
        probs = actual_model.predict_survival_function(actual_scaler.transform(X_test), return_array=True).T.tolist()
        for jj in range(len(times)):
            curr_time = times[jj]
            curr_prob = probs[jj]
            if str(curr_time) not in data[str(seed)]['survivaltree']:
                data[str(seed)]['survivaltree'][str(curr_time)] = []
            data[str(seed)]['survivaltree'][str(curr_time)].extend(curr_prob)

        # ======================================
        # GRADIENT BOOST
        # ======================================

        X, y = load_dataset(dataset_name=dataset_name)
        X, y = simple_basic_cast_and_nan_drop(X, y)

        random_state = seed ** 2
        set_random_seed(random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=[y_i[0] for y_i in y],
            random_state=random_state
        )

        n_features = X_train.shape[1]
        largest_value = 1e+8

        lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
        train_times = np.arange(lower, upper)
        tau_train = train_times[-1]

        lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
        test_times = np.arange(lower, upper)
        tau_test = test_times[-1]

        folds = StratifiedKFold(
            n_splits=n_folds_gb, shuffle=True,
            random_state=random_state + 50,
        ).split(X_train, [y_i[0] for y_i in y_train])

        param_grid = param_grid_gradient_boost
        model = GradientBoostingSurvivalAnalysis(random_state=random_state)

        pipeline = Pipeline(
            steps=[
                ('scaler', SimpleStdScalerOneHot(normalize=True)),
                ('model', as_concordance_index_ipcw_scorer(model, tau=tau_train))
            ]
        )

        gcv = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=folds,
            error_score=0.0,
            n_jobs=-1,
            refit=True,
            verbose=False
        ).fit(X_train, y_train)

        actual_model = gcv.best_estimator_['model'].estimator
        actual_scaler = gcv.best_estimator_['scaler']

        times = actual_model.unique_times_.tolist()
        probs = actual_model.predict_survival_function(actual_scaler.transform(X_test), return_array=True).T.tolist()
        for jj in range(len(times)):
            curr_time = times[jj]
            curr_prob = probs[jj]
            if str(curr_time) not in data[str(seed)]['gradientboost']:
                data[str(seed)]['gradientboost'][str(curr_time)] = []
            data[str(seed)]['gradientboost'][str(curr_time)].extend(curr_prob)

        # ======================================
        # RANDOM FOREST
        # ======================================

        X, y = load_dataset(dataset_name=dataset_name)
        X, y = simple_basic_cast_and_nan_drop(X, y)

        random_state = seed ** 2
        set_random_seed(random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=[y_i[0] for y_i in y],
            random_state=random_state
        )

        n_features = X_train.shape[1]
        largest_value = 1e+8

        lower, upper = np.percentile([y_i[1] for y_i in y_train], [1, 99])
        train_times = np.arange(lower, upper)
        tau_train = train_times[-1]

        lower, upper = np.percentile([y_i[1] for y_i in y_test], [1, 99])
        test_times = np.arange(lower, upper)
        tau_test = test_times[-1]

        folds = StratifiedKFold(
            n_splits=n_folds_rf, shuffle=True,
            random_state=random_state + 50,
        ).split(X_train, [y_i[0] for y_i in y_train])

        param_grid = param_grid_random_forest
        model = RandomSurvivalForest(random_state=random_state)

        pipeline = Pipeline(
            steps=[
                ('scaler', SimpleStdScalerOneHot(normalize=True)),
                ('model', as_concordance_index_ipcw_scorer(model, tau=tau_train))
            ]
        )

        gcv = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=folds,
            error_score=0.0,
            n_jobs=-1,
            refit=True,
            verbose=False
        ).fit(X_train, y_train)

        actual_model = gcv.best_estimator_['model'].estimator
        actual_scaler = gcv.best_estimator_['scaler']

        times = actual_model.unique_times_.tolist()
        probs = actual_model.predict_survival_function(actual_scaler.transform(X_test), return_array=True).T.tolist()
        for jj in range(len(times)):
            curr_time = times[jj]
            curr_prob = probs[jj]
            if str(curr_time) not in data[str(seed)]['randomforest']:
                data[str(seed)]['randomforest'][str(curr_time)] = []
            data[str(seed)]['randomforest'][str(curr_time)].extend(curr_prob)

    with open(f'survival_function_data_seedALL.json', 'w') as f:
        json.dump(data, f, indent=4)


def create_survival_function_lineplot(surv_func_data, palette_methods):
    # PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

    preamble = r'''
                \usepackage{amsmath}
                \usepackage{libertine}
                '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    fastplot.plot(None, f'surv_func_lineplot.pdf', mode='callback',
                  callback=lambda plt: my_callback_create_survival_function_lineplot(plt, surv_func_data, palette_methods), style='latex',
                  **PLOT_ARGS)


def my_callback_create_survival_function_lineplot(plt, surv_func_data, palette_methods):
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')

    x = []
    for method in palette_methods:
        x.extend(surv_func_data[f'{method}_times'])

    x = sorted(list(set(x)))

    for method in palette_methods:
        ax.plot(surv_func_data[f'{method}_times'], surv_func_data[f'{method}_probs'], label='', color=palette_methods[method], linestyle='-', linewidth=1.5, markersize=10)

    ax.set_xlim(min(x) - 10, max(x) + 10)
    ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
                  labels=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
                  fontsize=30.0)

    ax.set_ylim(0.58, 1.02)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0], labels=[0.6, 0.7, 0.8, 0.9, 1.0], fontsize=30.0)

    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.set_xlabel('Time', fontsize=17.0)
    ax.set_ylabel('Survival Probability', fontsize=17.0)
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)



def print_some_formulae(
        base_path,
        test_size,
        normalize,
        dataset_names,
        seed,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        how_many_pareto_features,
        dataset_names_acronyms
):

    table_string = ''
    for dataset_name in dataset_names:
        to_print_dataset = True
        nsgpd, pareto = read_nsgp(
            base_path=base_path,
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
            l1_ratio=l1_ratio_nsgp,
            max_iter=max_iter_nsgp,
            seed=seed,
            load_pareto=True
        )

        for k in how_many_pareto_features:
            if to_print_dataset:
                table_string += '\\multirow{' + str(len(how_many_pareto_features)) + '}{*}{' + '\\' + dataset_names_acronyms[dataset_name] +'}' + ' & '
            else:
                table_string += ' ' + ' & '
            to_print_dataset = False
            table_string += str(k) + ' & '

            train_error, test_error, latex_expr = take_formula(nsgpd, pareto, k)
            table_string += str(round(-train_error, 3)) + ' & '
            table_string += str(round(-test_error, 3)) + ' & '
            table_string += f'${latex_expr}$' + ' \\\\ \n'

        table_string += '\\midrule \n'
    print(table_string)
    return table_string


def stat_test_print(
        base_path,
        test_size,
        n_alphas,
        l1_ratio,
        alpha_min_ratio,
        max_iter,
        normalizes,
        split_types,
        dataset_names,
        seed_range,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        how_many_pareto_features,
        compare_single_points,
        how_many_pareto_features_boxplot,
        dataset_names_acronyms,
        palette_boxplot,
):

    for split_type in split_types:
        values_for_omnibus_test = {}
        if split_type.lower() == 'test':
            boxplot_data = {'Dataset': [], 'k': [], 'Length of Multi Tree': []}
        for normalize in normalizes:
            values_for_omnibus_test[str(normalize)] = []
            for dataset_name in dataset_names:
                X, y = load_dataset(dataset_name)
                X, y = simple_basic_cast_and_nan_drop(X, y)

                n_features = 100 # X.shape[1]
                ref_point = np.array([0.0, n_features])
                cox_hv_values = []
                nsgp_hv_values = []
                n_trees_in_multi_tree = []
                size_multi_tree = []
                for seed in seed_range:
                    coxd, _ = read_coxnet(
                        base_path=base_path,
                        method='coxnet',
                        dataset_name=dataset_name,
                        normalize=normalize,
                        test_size=test_size,
                        n_alphas=n_alphas,
                        l1_ratio=l1_ratio,
                        alpha_min_ratio=alpha_min_ratio,
                        max_iter=max_iter,
                        seed=seed
                    )

                    coxnet_n_features_list = list(coxd['DistinctRawFeatures'])
                    coxnet_errors_list = list(coxd[split_type + 'Error'])
                    if how_many_pareto_features <= 0:
                        cox_this_hv = list(coxd[split_type + 'HV'])[-1]
                        cox_hv_values.append(cox_this_hv)
                    else:
                        if compare_single_points:
                            compared_c_indexes = [-temp_error for temp_error, temp_feats in zip(coxnet_errors_list, coxnet_n_features_list) if temp_feats == how_many_pareto_features]
                            if len(compared_c_indexes) == 0:
                                compared_c_indexes = [0.0]
                            compared_c_index = compared_c_indexes[0]
                            cox_hv_values.append(compared_c_index)
                        else:
                            cx_pairs_pareto = np.array([[temp_error, temp_feats] for temp_error, temp_feats in zip(coxnet_errors_list, coxnet_n_features_list) if temp_feats <= how_many_pareto_features])
                            if len(cx_pairs_pareto) == 0:
                                cx_pairs_pareto = np.array([ref_point])
                            cox_hv_values.append(HV(ref_point)(cx_pairs_pareto))

                    nsgpd, _ = read_nsgp(
                        base_path=base_path,
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
                        l1_ratio=l1_ratio_nsgp,
                        max_iter=max_iter_nsgp,
                        seed=seed,
                        load_pareto=False
                    )

                    nsgp_n_trees_in_multi_tree_list = [float(val) for val in nsgpd.loc[num_gen - 1, 'ParetoNTrees'].split(' ')]
                    nsgp_size_in_multi_tree_list = [float(val) for val in nsgpd.loc[num_gen - 1, 'ParetoMaxTreeSize'].split(' ')]

                    nsgp_n_features_list = [float(val) for val in nsgpd.loc[num_gen - 1, 'ParetoObj2'].split(' ')]
                    nsgp_errors_list = [float(val) for val in nsgpd.loc[num_gen - 1, split_type + 'ParetoObj1'].split(' ')]
                    if how_many_pareto_features <= 0:
                        nsgp_this_hv = list(nsgpd[split_type + 'HV'])[-1]
                        nsgp_hv_values.append(nsgp_this_hv)
                    else:
                        if compare_single_points:
                            compared_c_indexes = [-temp_error for temp_error, temp_feats in zip(nsgp_errors_list, nsgp_n_features_list) if temp_feats == how_many_pareto_features]
                            if len(compared_c_indexes) == 0:
                                compared_c_indexes = [0.0]
                            compared_c_index = compared_c_indexes[0]
                            nsgp_hv_values.append(compared_c_index)

                            compared_n_trees = [temp_n_trees for temp_n_trees, temp_feats in zip(nsgp_n_trees_in_multi_tree_list, nsgp_n_features_list) if temp_feats == how_many_pareto_features]
                            if len(compared_n_trees) == 0:
                                compared_n_trees = [0]
                            compared_n_tree = compared_n_trees[0]
                            n_trees_in_multi_tree.append(compared_n_tree)

                            for k in how_many_pareto_features_boxplot:
                                compared_n_trees = [temp_n_trees for temp_n_trees, temp_feats in zip(nsgp_n_trees_in_multi_tree_list, nsgp_n_features_list) if temp_feats == k]
                                if len(compared_n_trees) == 0:
                                    compared_n_trees = [0]
                                compared_n_tree = compared_n_trees[0]
                                if not normalize and split_type.lower() == 'test':
                                    boxplot_data['Length of Multi Tree'].append(compared_n_tree)
                                    boxplot_data['k'].append(f'$k = {k}$')
                                    boxplot_data['Dataset'].append(dataset_names_acronyms[dataset_name])

                            compared_sizes = [temp_size for temp_size, temp_feats in zip(nsgp_size_in_multi_tree_list, nsgp_n_features_list) if temp_feats == how_many_pareto_features]
                            if len(compared_sizes) == 0:
                                compared_sizes = [0]
                            compared_size = compared_sizes[0]
                            size_multi_tree.append(compared_size)
                        else:
                            ns_pairs_pareto = np.array([[temp_error, temp_feats] for temp_error, temp_feats in zip(nsgp_errors_list, nsgp_n_features_list) if temp_feats <= how_many_pareto_features])
                            if len(ns_pairs_pareto) == 0:
                                ns_pairs_pareto = np.array([ref_point])
                            nsgp_hv_values.append(HV(ref_point)(ns_pairs_pareto))

                    #scatter_plot(dataset=dataset_name, seed=seed, split_type=split_type.strip(), coxnet_n_features_list=coxnet_n_features_list, coxnet_errors_list=coxnet_errors_list, nsgp_n_features_list=nsgp_n_features_list, nsgp_errors_list=nsgp_errors_list)

                values_for_omnibus_test[str(normalize)].extend(nsgp_hv_values)

                if len(n_trees_in_multi_tree) > 0:
                    print(f'{normalize} {split_type} {dataset_name} NSGP SIZE ', f'median {statistics.median(size_multi_tree)}', ' ', f'mean {statistics.mean(size_multi_tree)}', ' ', f'q1 {np.percentile(size_multi_tree, 25)}', ' ', f'q3 {np.percentile(size_multi_tree, 75)}')
                    print(f'{normalize} {split_type} {dataset_name} NSGP N TREES ', f'median {statistics.median(n_trees_in_multi_tree)}', ' ', f'mean {statistics.mean(n_trees_in_multi_tree)}', ' ', f'q1 {np.percentile(n_trees_in_multi_tree, 25)}', ' ', f'q3 {np.percentile(n_trees_in_multi_tree, 75)}')
                print(f'{normalize} {split_type} {dataset_name} HV COX ', f'median {statistics.median(cox_hv_values)}', ' ', f'mean {statistics.mean(cox_hv_values)}', ' ', f'q1 {np.percentile(cox_hv_values, 25)}', ' ', f'q3 {np.percentile(cox_hv_values, 75)}')
                print(f'{normalize} {split_type} {dataset_name} HV NSGP ', f'median {statistics.median(nsgp_hv_values)}', ' ', f'mean {statistics.mean(nsgp_hv_values)}', ' ', f'q1 {np.percentile(nsgp_hv_values, 25)}', ' ', f'q3 {np.percentile(nsgp_hv_values, 75)}')
                print(f'{normalize} {split_type} {dataset_name} HV MannWhitheyU', is_mannwhitneyu_passed(cox_hv_values, nsgp_hv_values, alternative='less'))
                print()

        print()
        print(f'{split_type} Omnibus Test (Kruskal-Wallis) {is_kruskalwallis_passed(values_for_omnibus_test)}')
        print()

        # PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

        preamble = r'''
            \usepackage{amsmath}
            \usepackage{libertine}
            '''

        PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

        if split_type.lower() == 'test':
            fastplot.plot(None, f'boxplot.pdf', mode='callback', callback=lambda plt: my_callback_boxplot(plt, boxplot_data, palette_boxplot), style='latex', **PLOT_ARGS)


def my_callback_boxplot(plt, data, palette):
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.set_ylim(0, 13)
    ax.set_yticks(list(range(1, 12 + 1)))
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    sns.boxplot(pd.DataFrame(data), x='Dataset', y='Length of Multi Tree', hue='k', palette=palette, legend=False, log_scale=None, fliersize=0.0, showfliers=False, ax=ax)


def create_lineplots_on_single_line_multitree_length(
        base_path,
        test_size,
        dataset_names,
        seed_range,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        how_many_pareto_features,
        dataset_names_acronyms,
):

    # dataset k values
    lineplot_data = {dataset_name: {k: [] for k in how_many_pareto_features} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        for seed in seed_range:
            nsgpd, _ = read_nsgp(
                base_path=base_path,
                method='nsgp',
                dataset_name=dataset_name,
                normalize=False,
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
                l1_ratio=l1_ratio_nsgp,
                max_iter=max_iter_nsgp,
                seed=seed,
                load_pareto=False
            )

            nsgp_n_trees_in_multi_tree_list = [float(val) for val in nsgpd.loc[num_gen - 1, 'ParetoNTrees'].split(' ')]
            nsgp_n_features_list = [float(val) for val in nsgpd.loc[num_gen - 1, 'ParetoObj2'].split(' ')]

            for k in how_many_pareto_features:
                compared_n_trees = [temp_n_trees for temp_n_trees, temp_feats in zip(nsgp_n_trees_in_multi_tree_list, nsgp_n_features_list) if temp_feats == k]
                if len(compared_n_trees) == 0:
                    compared_n_trees = [0]
                compared_n_tree = compared_n_trees[0]
                lineplot_data[dataset_name][k].append(compared_n_tree)

    # PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

    preamble = r'''
    \usepackage{amsmath}
    \usepackage{libertine}
    '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    fastplot.plot(None, f'multitree_length_lineplot.pdf', mode='callback', callback=lambda plt: my_callback_multitree_length_lineplot(plt, lineplot_data, dataset_names, dataset_names_acronyms, how_many_pareto_features), style='latex', **PLOT_ARGS)


def my_callback_multitree_length_lineplot(plt, lineplot_data, dataset_names, dataset_names_acronyms, how_many_pareto_features):
    n, m = 1, len(dataset_names)
    fig, ax = plt.subplots(n, m, figsize=(10, 2), layout='constrained', squeeze=False)
    x = how_many_pareto_features

    for i in range(m):
        dataset_name = dataset_names[i]
        acronym = dataset_names_acronyms[dataset_name]

        all_med, all_q1, all_q3 = [], [], []
        for k in how_many_pareto_features:
            actual_data = lineplot_data[dataset_name][k]
            all_med.append(statistics.median(actual_data))
            all_q1.append(np.percentile(actual_data, 25))
            all_q3.append(np.percentile(actual_data, 75))

        ax[0, i].plot(x, all_med, label='', color='#000080',
                      linestyle='-',
                      linewidth=1.0, markersize=10)
        ax[0, i].fill_between(x, all_q1, all_q3,
                              color='#000080', alpha=0.1)


        ax[0, i].set_xlim(min(how_many_pareto_features), max(how_many_pareto_features))
        ax[0, i].set_xticks(how_many_pareto_features)

        ax[0, i].set_ylim(2, 12)
        ax[0, i].set_yticks([3, 5, 7, 9, 11])

        ax[0, i].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False,
                             right=False)

        ax[0, i].set_title(acronym)

        ax[0, i].set_xlabel('$k$')

        if i == 0:
            ax[0, i].set_ylabel('Multi-Tree Length')
        else:
            ax[0, i].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
            ax[0, i].tick_params(labelleft=False)
            ax[0, i].set_yticklabels([])

        if i == m - 1:
            ax[0, i].tick_params(pad=7)

        ax[0, i].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)


def median_pareto_front_all_methods(
        base_path,
        test_size,
        n_alphas,
        l1_ratio,
        alpha_min_ratio,
        max_iter,
        dataset_names,
        dataset_names_acronyms,
        split_type,
        seed_range,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        n_max_depths_st,
        n_folds_st,
        n_max_depths_gb,
        n_folds_gb,
        n_max_depths_rf,
        n_folds_rf,
        palette_methods,
):

    methods: list[str] = ['coxnet', 'nsgp', 'survivaltree', 'gradientboost', 'randomforest']

    # dataset method values
    values: dict[str, dict[str, list[float]]] = {dataset_name: {method: [] for method in methods} for dataset_name in dataset_names}
    paretos: dict[str, dict[str, list[list[list[float]]]]] = {dataset_name: {method: [] for method in methods} for dataset_name in dataset_names}

    medians: dict[str, dict[str, float]] = {dataset_name: {} for dataset_name in dataset_names}

    n_features = 100 # X.shape[1]
    ref_point = np.array([0.0, n_features])

    for dataset_name in dataset_names:
        for method in methods:
            for seed in seed_range:
                if method == 'coxnet':
                    csv_data, _ = read_coxnet(
                        base_path=base_path,
                        method=method,
                        dataset_name=dataset_name,
                        normalize=True,
                        test_size=test_size,
                        n_alphas=n_alphas,
                        l1_ratio=l1_ratio,
                        alpha_min_ratio=alpha_min_ratio,
                        max_iter=max_iter,
                        seed=seed
                    )

                    n_features_list = list(csv_data['DistinctRawFeatures'])
                    errors_list = list(csv_data[split_type + 'Error'])
                elif method == 'survivaltree':
                    csv_data, _ = read_survivalensembletree(
                        base_path=base_path,
                        method=method,
                        dataset_name=dataset_name,
                        normalize=True,
                        test_size=test_size,
                        n_max_depths=n_max_depths_st,
                        n_folds=n_folds_st,
                        seed=seed
                    )

                    n_features_list = list(csv_data['DistinctRawFeatures'])
                    errors_list = list(csv_data[split_type + 'Error'])
                elif method == 'gradientboost':
                    csv_data, _ = read_survivalensembletree(
                        base_path=base_path,
                        method=method,
                        dataset_name=dataset_name,
                        normalize=True,
                        test_size=test_size,
                        n_max_depths=n_max_depths_gb,
                        n_folds=n_folds_gb,
                        seed=seed
                    )

                    n_features_list = list(csv_data['DistinctRawFeatures'])
                    errors_list = list(csv_data[split_type + 'Error'])
                elif method == 'randomforest':
                    csv_data, _ = read_survivalensembletree(
                        base_path=base_path,
                        method=method,
                        dataset_name=dataset_name,
                        normalize=True,
                        test_size=test_size,
                        n_max_depths=n_max_depths_rf,
                        n_folds=n_folds_rf,
                        seed=seed
                    )

                    n_features_list = list(csv_data['DistinctRawFeatures'])
                    errors_list = list(csv_data[split_type + 'Error'])
                elif method == 'nsgp':
                    csv_data, _ = read_nsgp(
                        base_path=base_path,
                        method=method,
                        dataset_name=dataset_name,
                        normalize=True,
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
                        l1_ratio=l1_ratio_nsgp,
                        max_iter=max_iter_nsgp,
                        seed=seed,
                        load_pareto=False
                    )

                    n_features_list = [float(val) for val in csv_data.loc[num_gen - 1, 'ParetoObj2'].split(' ')]
                    errors_list = [float(val) for val in csv_data.loc[num_gen - 1, split_type + 'ParetoObj1'].split(' ')]
                else:
                    raise ValueError(f'Unrecognized method {method}.')

                if method in ('gradientboost', 'randomforest'):
                    compared_c_indexes = [temp_error for temp_error, temp_feats in zip(errors_list, n_features_list)]
                    if len(compared_c_indexes) == 0:
                        compared_c_indexes = [0.0]
                    else:
                        compared_c_index = compared_c_indexes[0]
                        values[dataset_name][method].append(compared_c_index)
                else:
                    cx_pairs_pareto = [[temp_error, temp_feats] for temp_error, temp_feats in zip(errors_list, n_features_list)]
                    if len(cx_pairs_pareto) == 0:
                        cx_pairs_pareto = [[0.0, n_features]]
                    else:
                        cx_pairs_pareto = filter_non_dominated_only(cx_pairs_pareto)
                        values[dataset_name][method].append(HV(ref_point)(np.array(cx_pairs_pareto)))
                        paretos[dataset_name][method].append(cx_pairs_pareto)

    for dataset_name in dataset_names:
        for method in methods:
            medians[dataset_name][method] = statistics.median(values[dataset_name][method])

    actual_paretos: dict[str, dict[str, list[list[float]]]] = {dataset_name: {} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        for method in methods:
            if method in ('gradientboost', 'randomforest'):
                continue
            i = int(np.argmin([abs(val - medians[dataset_name][method]) for val in values[dataset_name][method]]))
            actual_paretos[dataset_name][method] = paretos[dataset_name][method][i]

    # PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

    preamble = r'''
    \usepackage{amsmath}
    \usepackage{libertine}
    '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    fastplot.plot(None, f'pareto_all_methods.pdf', mode='callback',
                  callback=lambda plt: my_callback_pareto_all_methods_2(plt, medians, actual_paretos, palette_methods, dataset_names, dataset_names_acronyms), style='latex',
                  **PLOT_ARGS)


def my_callback_pareto_all_methods(plt, medians, actual_paretos, palette_methods, dataset_names, dataset_names_acronyms):
    fig, ax = plt.subplots(len(dataset_names), 1, figsize=(6, 12), layout='constrained', squeeze=False)

    markers = {'coxnet': 'o', 'survivaltree': '^', 'nsgp': '*'}

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        acronym = dataset_names_acronyms[dataset_name]
        for method in markers:
            for pair in actual_paretos[dataset_name][method]:
                err, size = pair[0], pair[1]
                ax[i, 0].scatter(err, size, c=palette_methods[method], marker=markers[method], s=50, edgecolor='black', linewidth=0.5)

        ax[i, 0].axvline(medians[dataset_name]['gradientboost'], c=palette_methods['gradientboost'], linewidth=1.2, linestyle='-')
        ax[i, 0].axvline(medians[dataset_name]['randomforest'], c=palette_methods['randomforest'], linewidth=1.2, linestyle='-')

        ax[i, 0].set_ylim(-1, 33)
        ax[i, 0].set_yticks(list(range(1, 33 + 1, 3)))
        #ax[i, 0].set_xlim(-0.8, -0.6)
        #ax[i, 0].set_xticks([-0.75, -0.70, -0.65])
        ax[i, 0].set_xlim(-0.83, -0.52)
        ax[i, 0].set_xticks([-0.80, -0.75, -0.70, -0.65, -0.60, -0.55])
        ax[i, 0].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
        if i == len(dataset_names) - 1:
            ax[i, 0].set_xlabel(r'$\textit{obj}_1$')
        else:
            ax[i, 0].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
            ax[i, 0].tick_params(labelbottom=False)
            ax[i, 0].set_xticklabels([])
        ax[i, 0].set_ylabel(r'$\textit{obj}_2$')

        axttt = ax[i, 0].twinx()
        axttt.set_ylabel(acronym, rotation=270, labelpad=14)
        axttt.yaxis.set_label_position("right")
        axttt.tick_params(labelleft=False)
        axttt.set_yticklabels([])
        axttt.yaxis.tick_right()
        axttt.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
        ax[i, 0].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

        if i == len(dataset_names) - 1:
            ax[i, 0].tick_params(pad=7)

        # ax[i, 0].set_title(f'Methods Pareto Front ({metric} across all datasets and repetitions)')
        ax[i, 0].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)


def my_callback_pareto_all_methods_2(plt, medians, actual_paretos, palette_methods, dataset_names, dataset_names_acronyms):
    fig, ax = plt.subplots(1, len(dataset_names), figsize=(18, 4), layout='constrained', squeeze=False)

    markers = {'coxnet': 'o', 'survivaltree': '^', 'nsgp': '*'}

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        acronym = dataset_names_acronyms[dataset_name]
        for method in markers:
            for pair in actual_paretos[dataset_name][method]:
                err, size = pair[0], pair[1]
                ax[0, i].scatter(err, size, c=palette_methods[method], marker=markers[method], s=50, edgecolor='black', linewidth=0.5)

        ax[0, i].axvline(medians[dataset_name]['gradientboost'], c=palette_methods['gradientboost'], linewidth=1.2, linestyle='-')
        ax[0, i].axvline(medians[dataset_name]['randomforest'], c=palette_methods['randomforest'], linewidth=1.2, linestyle='-')

        ax[0, i].set_ylim(-1, 33)
        ax[0, i].set_yticks(list(range(1, 35 + 1, 5)))
        #ax[0, i].set_xlim(-0.88, -0.52)
        #ax[0, i].set_xticks([-0.80, -0.70, -0.60])
        if dataset_name == 'pbc2':
            ax[0, i].set_xlim(-0.85, -0.65)
            ax[0, i].set_xticks([-0.80, -0.75, -0.70])
        elif dataset_name == 'support2':
            ax[0, i].set_xlim(-0.71, -0.59)
            ax[0, i].set_xticks([-0.70, -0.65, -0.60])
        elif dataset_name == 'framingham':
            ax[0, i].set_xlim(-0.78, -0.62)
            ax[0, i].set_xticks([-0.75, -0.70, -0.65])
        elif dataset_name == 'breast_cancer_metabric':
            ax[0, i].set_xlim(-0.70, -0.50)
            ax[0, i].set_xticks([-0.65, -0.60, -0.55])
        elif dataset_name == 'breast_cancer_metabric_relapse':
            ax[0, i].set_xlim(-0.67, -0.53)
            ax[0, i].set_xticks([-0.65, -0.60, -0.55])
        else:
            raise AttributeError(f'{dataset_name} as dataset not recognized for this plot.')

        ax[0, i].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
        if i == 0:
            ax[0, i].set_ylabel(r'$\textit{obj}_2$', fontsize=18)
        else:
            ax[0, i].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
            ax[0, i].tick_params(labelleft=False)
            ax[0, i].set_yticklabels([])
        ax[0, i].set_xlabel(r'$\textit{obj}_1$', fontsize=18)

        # axttt = ax[0, i].twinx()
        # axttt.set_ylabel(acronym, rotation=270, labelpad=14)
        # axttt.yaxis.set_label_position("right")
        # axttt.tick_params(labelleft=False)
        # axttt.set_yticklabels([])
        # axttt.yaxis.tick_right()
        # axttt.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
        ax[0, i].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

        if i == len(dataset_names) - 1:
            ax[0, i].tick_params(pad=7)
        ax[0, i].set_title(acronym)
        # ax[0, i].set_title(f'Methods Pareto Front ({metric} across all datasets and repetitions)')
        ax[0, i].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)



def stat_test(
        base_path,
        test_size,
        n_alphas,
        l1_ratio,
        alpha_min_ratio,
        max_iter,
        normalizes,
        dataset_names,
        seed_range,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        alpha,
        l1_ratio_nsgp,
        max_iter_nsgp,
        n_max_depths_st,
        n_folds_st,
        n_max_depths_gb,
        n_folds_gb,
        n_max_depths_rf,
        n_folds_rf,
        how_many_pareto_features_table,
        methods,
):

    # split_type metric k normalize dataset method values
    values: dict[str, dict[str, dict[int, dict[bool, dict[str, dict[str, list[float]]]]]]] = {
        'Train': {'HV': {}, 'CI': {}},
        'Test': {'HV': {}, 'CI': {}},
    }
    # split_type metric k normalize dataset method methods-outperformed
    comparisons: dict[str, dict[str, dict[int, dict[bool, dict[str, dict[str, list[str]]]]]]] = {
        'Train': {'HV': {}, 'CI': {}},
        'Test': {'HV': {}, 'CI': {}},
    }

    opaque_methods = ('gradientboost', 'randomforest')

    for split_type in ['Train', 'Test']:
        for k in how_many_pareto_features_table:

            if k not in values[split_type]['HV']:
                values[split_type]['HV'][k] = {}
            if k not in values[split_type]['CI']:
                values[split_type]['CI'][k] = {}
            if k not in comparisons[split_type]['HV']:
                comparisons[split_type]['HV'][k] = {}
            if k not in comparisons[split_type]['CI']:
                comparisons[split_type]['CI'][k] = {}

            for normalize in normalizes:

                if normalize not in values[split_type]['HV'][k]:
                    values[split_type]['HV'][k][normalize] = {}
                if normalize not in values[split_type]['CI'][k]:
                    values[split_type]['CI'][k][normalize] = {}
                if normalize not in comparisons[split_type]['HV'][k]:
                    comparisons[split_type]['HV'][k][normalize] = {}
                if normalize not in comparisons[split_type]['CI'][k]:
                    comparisons[split_type]['CI'][k][normalize] = {}

                for dataset_name in dataset_names:

                    if dataset_name not in values[split_type]['HV'][k][normalize]:
                        values[split_type]['HV'][k][normalize][dataset_name] = {}
                    if dataset_name not in values[split_type]['CI'][k][normalize]:
                        values[split_type]['CI'][k][normalize][dataset_name] = {}
                    if dataset_name not in comparisons[split_type]['HV'][k][normalize]:
                        comparisons[split_type]['HV'][k][normalize][dataset_name] = {}
                    if dataset_name not in comparisons[split_type]['CI'][k][normalize]:
                        comparisons[split_type]['CI'][k][normalize][dataset_name] = {}

                    X, y = load_dataset(dataset_name)
                    X, y = simple_basic_cast_and_nan_drop(X, y)
                    n_features = 100 # X.shape[1]
                    ref_point = np.array([0.0, n_features])

                    for method in methods:

                        if method not in values[split_type]['HV'][k][normalize][dataset_name]:
                            values[split_type]['HV'][k][normalize][dataset_name][method] = []
                        if method not in values[split_type]['CI'][k][normalize][dataset_name]:
                            values[split_type]['CI'][k][normalize][dataset_name][method] = []
                        if method not in comparisons[split_type]['HV'][k][normalize][dataset_name]:
                            comparisons[split_type]['HV'][k][normalize][dataset_name][method] = []
                        if method not in comparisons[split_type]['CI'][k][normalize][dataset_name]:
                            comparisons[split_type]['CI'][k][normalize][dataset_name][method] = []

                        for seed in seed_range:
                            if method == 'coxnet':
                                csv_data, _ = read_coxnet(
                                    base_path=base_path,
                                    method=method,
                                    dataset_name=dataset_name,
                                    normalize=normalize,
                                    test_size=test_size,
                                    n_alphas=n_alphas,
                                    l1_ratio=l1_ratio,
                                    alpha_min_ratio=alpha_min_ratio,
                                    max_iter=max_iter,
                                    seed=seed
                                )

                                n_features_list = list(csv_data['DistinctRawFeatures'])
                                errors_list = list(csv_data[split_type + 'Error'])
                            elif method == 'survivaltree':
                                csv_data, _ = read_survivalensembletree(
                                    base_path=base_path,
                                    method=method,
                                    dataset_name=dataset_name,
                                    normalize=normalize,
                                    test_size=test_size,
                                    n_max_depths=n_max_depths_st,
                                    n_folds=n_folds_st,
                                    seed=seed
                                )

                                n_features_list = list(csv_data['DistinctRawFeatures'])
                                errors_list = list(csv_data[split_type + 'Error'])
                            elif method == 'gradientboost':
                                csv_data, _ = read_survivalensembletree(
                                    base_path=base_path,
                                    method=method,
                                    dataset_name=dataset_name,
                                    normalize=normalize,
                                    test_size=test_size,
                                    n_max_depths=n_max_depths_gb,
                                    n_folds=n_folds_gb,
                                    seed=seed
                                )

                                n_features_list = list(csv_data['DistinctRawFeatures'])
                                errors_list = list(csv_data[split_type + 'Error'])
                            elif method == 'randomforest':
                                csv_data, _ = read_survivalensembletree(
                                    base_path=base_path,
                                    method=method,
                                    dataset_name=dataset_name,
                                    normalize=normalize,
                                    test_size=test_size,
                                    n_max_depths=n_max_depths_rf,
                                    n_folds=n_folds_rf,
                                    seed=seed
                                )

                                n_features_list = list(csv_data['DistinctRawFeatures'])
                                errors_list = list(csv_data[split_type + 'Error'])
                            elif method == 'nsgp':
                                csv_data, _ = read_nsgp(
                                    base_path=base_path,
                                    method=method,
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
                                    l1_ratio=l1_ratio_nsgp,
                                    max_iter=max_iter_nsgp,
                                    seed=seed,
                                    load_pareto=False
                                )

                                n_features_list = [float(val) for val in csv_data.loc[num_gen - 1, 'ParetoObj2'].split(' ')]
                                errors_list = [float(val) for val in csv_data.loc[num_gen - 1, split_type + 'ParetoObj1'].split(' ')]
                            elif method == 'randomsearch':
                                csv_data, _ = read_nsgp(
                                    base_path=base_path,
                                    method=method,
                                    dataset_name=dataset_name,
                                    normalize=normalize,
                                    test_size=test_size,
                                    pop_size=pop_size * num_gen,
                                    num_gen=1,
                                    max_size=max_size,
                                    min_depth=min_depth,
                                    init_max_height=init_max_height,
                                    tournament_size=tournament_size,
                                    min_trees_init=min_trees_init,
                                    max_trees_init=max_trees_init,
                                    alpha=alpha,
                                    l1_ratio=l1_ratio_nsgp,
                                    max_iter=max_iter_nsgp,
                                    seed=seed,
                                    load_pareto=False
                                )

                                n_features_list = [float(val) for val in csv_data.loc[num_gen - 1, 'ParetoObj2'].split(' ')]
                                errors_list = [float(val) for val in csv_data.loc[num_gen - 1, split_type + 'ParetoObj1'].split(' ')]
                            else:
                                raise ValueError(f'Unrecognized method {method}.')

                            compared_c_indexes = [-temp_error for temp_error, temp_feats in zip(errors_list, n_features_list) if temp_feats == (k if k < 1000 and method not in opaque_methods else max(n_features_list))]
                            if len(compared_c_indexes) == 0:
                                compared_c_indexes = [0.0]
                            else:
                                compared_c_index = compared_c_indexes[0]
                                values[split_type]['CI'][k][normalize][dataset_name][method].append(compared_c_index)

                            cx_pairs_pareto = np.array([[temp_error, temp_feats] for temp_error, temp_feats in zip(errors_list, n_features_list) if temp_feats <= k])
                            if len(cx_pairs_pareto) == 0:
                                cx_pairs_pareto = np.array([ref_point])
                            else:
                                values[split_type]['HV'][k][normalize][dataset_name][method].append(HV(ref_point)(cx_pairs_pareto))

    for split_type in ['Train', 'Test']:
        for k in how_many_pareto_features_table:
            for normalize in normalizes:
                for dataset_name in dataset_names:
                    for metric in ['CI', 'HV']:
                        for method in methods:
                            if len(values[split_type][metric][k][normalize][dataset_name][method]) == 0:
                                values[split_type][metric][k][normalize][dataset_name][method] = [0.0] * len(seed_range)


    for split_type in ['Train', 'Test']:
        for k in how_many_pareto_features_table:
            for normalize in normalizes:
                for dataset_name in dataset_names:
                    for metric in ['CI', 'HV']:
                        if is_kruskalwallis_passed(values[split_type][metric][k][normalize][dataset_name], alpha=0.05):
                            bonferroni_dict, mann_dict = perform_mannwhitneyu_holm_bonferroni(values[split_type][metric][k][normalize][dataset_name], alternative='greater', alpha=0.05)
                            for method in methods:
                                if bonferroni_dict[method]:
                                    comparisons[split_type][metric][k][normalize][dataset_name][method].extend(methods)
                                else:
                                    for method_2 in methods:
                                        if method != method_2 and mann_dict[method][method_2]:
                                            comparisons[split_type][metric][k][normalize][dataset_name][method].append(method_2)

    return values, comparisons


def print_table_hv_ci(
        values,
        comparisons,
        methods,
        methods_acronyms,
        how_many_pareto_features_table,
        normalizes,
        dataset_names,
) -> None:

    hv_interpret_table = ''
    ci_interpret_table = ''
    num_methods = len(methods)
    for k in how_many_pareto_features_table:
        is_first = True
        for method in methods:
            if is_first:
                hv_interpret_table += '\\multirow{' + str(num_methods) + '}' + '{*}' + '{' + str(k if k < 1000 else '\\text{max}') + '}'
                ci_interpret_table += '\\multirow{' + str(num_methods) + '}' + '{*}' + '{' + str(k if k < 1000 else '\\text{max}') + '}'
            else:
                hv_interpret_table += ' '
                ci_interpret_table += ' '
            is_first = False
            hv_interpret_table += ' & ' + methods_acronyms[method]
            ci_interpret_table += ' & ' + methods_acronyms[method]
            for normalize in normalizes:
                for dataset_name in dataset_names:
                    hv_median = statistics.median(values['Test']['HV'][str(k)][str(normalize).lower()][dataset_name][method])
                    if len([curr_val for curr_val in values['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method] if curr_val != 0]) == 0:
                        ci_median = '{-}'
                    else:
                        ci_median = statistics.median([curr_val for curr_val in values['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method] if curr_val != 0])
                    hv_num_outperformed_methods = len(comparisons['Test']['HV'][str(k)][str(normalize).lower()][dataset_name][method])
                    ci_num_outperformed_methods = len(comparisons['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method])
                    hv_median_max = max([statistics.median(values['Test']['HV'][str(k)][str(normalize).lower()][dataset_name][method_2]) for method_2 in methods])
                    ci_median_max = max([statistics.median(values['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method_2]) for method_2 in methods])
                    hv_bold = ''
                    ci_bold = ''
                    hv_star = ''
                    ci_star = ''
                    if hv_median == hv_median_max or hv_num_outperformed_methods == num_methods:
                        hv_bold = '\\bfseries '
                    if ci_median == ci_median_max or ci_num_outperformed_methods == num_methods:
                        ci_bold = '\\bfseries '
                    if hv_num_outperformed_methods == num_methods:
                        hv_star = '{$^{\\scalebox{0.90}{\\textbf{\\color{blue}*}}}$}'
                    elif hv_num_outperformed_methods > 0:
                        hv_star = '{$^{\\scalebox{0.90}{\\textbf{\\color{black}*}}}$}'
                    if ci_num_outperformed_methods == num_methods:
                        ci_star = '{$^{\\scalebox{0.90}{\\textbf{\\color{blue}*}}}$}'
                    elif ci_num_outperformed_methods > 0:
                        ci_star = '{$^{\\scalebox{0.90}{\\textbf{\\color{black}*}}}$}'

                    hv_interpret_table += ' & ' + hv_bold + str(round(hv_median, 3)) + hv_star
                    ci_interpret_table += ' & ' + ci_bold + (str(round(ci_median, 3)) if ci_median != '{-}' else ci_median) + ci_star

            hv_interpret_table += ' \\\\ \n'
            ci_interpret_table += ' \\\\ \n'
        hv_interpret_table += ' \\midrule \n'
        ci_interpret_table += ' \\midrule \n'


    print('\n\n\nTABLE INTERPRETABLE MODELS HV\n\n\n')
    print(hv_interpret_table)
    print('\n\n\nTABLE INTERPRETABLE MODELS CI\n\n\n')
    print(ci_interpret_table)
    print()
    print()


def lineplot(
        base_path,
        data,
        test_size,
        normalize,
        dataset_names,
        seed_range,
        pop_size,
        num_gen,
        max_size,
        min_depth,
        init_max_height,
        tournament_size,
        min_trees_init,
        max_trees_init,
        l1_ratio_nsgp,
        max_iter_nsgp,
        alpha,
        dataset_acronyms,
        how_many_pareto_features,
) -> None:
    if data is None or len(data) == 0:

        #dataset k split_type+aggregation+metric values_for_each_generation
        data = {
            dataset_name: {
                str(k): {
                    'TrainMedianHV': [], 'TrainQ1HV': [], 'TrainQ3HV': [],
                    'TestMedianHV': [], 'TestQ1HV': [], 'TestQ3HV': []
                }
                for k in how_many_pareto_features
            }
            for dataset_name in dataset_names
        }

        highest_n_features_ever_found = 0

        for k in how_many_pareto_features:
            print(k)
            for dataset_name in dataset_names:
                print(dataset_name)
                X, y = load_dataset(dataset_name)
                X, y = simple_basic_cast_and_nan_drop(X, y)
                n_features = 100 # X.shape[1]
                ref_point = np.array([0.0, n_features])

                train_single_values = []
                test_single_values = []
                for seed in seed_range:
                    csv_data, _ = read_nsgp(
                        base_path=base_path,
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
                        l1_ratio=l1_ratio_nsgp,
                        max_iter=max_iter_nsgp,
                        seed=seed,
                        load_pareto=False
                    )
                    train_obj1 = list(csv_data['TrainParetoObj1'])
                    test_obj1 = list(csv_data['TestParetoObj1'])
                    obj_2 = list(csv_data['ParetoObj2'])

                    train_hv = []
                    test_hv = []

                    for single_train_obj1, single_test_obj1, single_obj2 in zip(train_obj1, test_obj1, obj_2):
                        single_train_obj1 = [float(val) for val in single_train_obj1.split(' ')]
                        single_test_obj1 = [float(val) for val in single_test_obj1.split(' ')]
                        single_obj2 = [float(val) for val in single_obj2.split(' ')]

                        if max(single_obj2) > highest_n_features_ever_found:
                            highest_n_features_ever_found = max(single_obj2)

                        cx_pairs_pareto = np.array([[temp_error, temp_feats] for temp_error, temp_feats in zip(single_train_obj1, single_obj2) if temp_feats <= k])
                        if len(cx_pairs_pareto) == 0:
                            cx_pairs_pareto = np.array([ref_point])
                        train_hv.append(HV(ref_point)(cx_pairs_pareto))

                        cx_pairs_pareto = np.array([[temp_error, temp_feats] for temp_error, temp_feats in zip(single_test_obj1, single_obj2) if temp_feats <= k])
                        if len(cx_pairs_pareto) == 0:
                            cx_pairs_pareto = np.array([ref_point])
                        test_hv.append(HV(ref_point)(cx_pairs_pareto))

                    train_single_values.append(train_hv)
                    test_single_values.append(test_hv)
                train_single_values = list(map(list, zip(*train_single_values)))
                test_single_values = list(map(list, zip(*test_single_values)))
                for l in train_single_values:
                    data[dataset_name][str(k)]['TrainMedianHV'].append(statistics.median(l))
                    data[dataset_name][str(k)]['TrainQ1HV'].append(float(np.percentile(l, 25)))
                    data[dataset_name][str(k)]['TrainQ3HV'].append(float(np.percentile(l, 75)))
                for l in test_single_values:
                    data[dataset_name][str(k)]['TestMedianHV'].append(statistics.median(l))
                    data[dataset_name][str(k)]['TestQ1HV'].append(float(np.percentile(l, 25)))
                    data[dataset_name][str(k)]['TestQ3HV'].append(float(np.percentile(l, 75)))

        print(f'HIGHEST NUMBER OF FEATURES EVER FOUND: {highest_n_features_ever_found}')

        with open('lineplot_data.json', 'w') as f:
            json.dump(data, f, indent=4)

    # PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

    preamble = r'''
    \usepackage{amsmath}
    \usepackage{libertine}
    '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    fastplot.plot(None, f'lineplot.pdf', mode='callback', callback=lambda plt: my_callback_lineplot(plt, data, how_many_pareto_features, dataset_names, dataset_acronyms), style='latex', **PLOT_ARGS)


def my_callback_lineplot(plt, data, how_many_pareto_features, dataset_names, dataset_acronyms):
    if len(how_many_pareto_features) > 1:
        n, m = len(dataset_names), len(how_many_pareto_features)
        fig, ax = plt.subplots(n, m, figsize=(10, 7), layout='constrained', squeeze=False)
        x = list(range(1, 100 + 1))

        for i in range(n):
            dataset_name = dataset_names[i]
            acronym = dataset_acronyms[dataset_name]
            for j in range(m):
                k = how_many_pareto_features[j]
                actual_data = data[dataset_name][str(k)]

                ax[i, j].plot(x, actual_data['TrainMedianHV'], label='', color='#E51D1D',
                              linestyle='-',
                              linewidth=1.0, markersize=10)
                ax[i, j].fill_between(x, actual_data['TrainQ1HV'], actual_data['TrainQ3HV'],
                                      color='#E51D1D', alpha=0.1)

                ax[i, j].plot(x, actual_data['TestMedianHV'], label='', color='#3B17F2',
                              linestyle='-',
                              linewidth=1.0, markersize=10)
                ax[i, j].fill_between(x, actual_data['TestQ1HV'], actual_data['TestQ3HV'],
                                      color='#3B17F2', alpha=0.1)

                ax[i, j].set_xlim(1, 100)
                ax[i, j].set_xticks([1, 100 // 2, 100])

                if dataset_name == 'pbc2':
                    ax[i, j].set_ylim(72, 84)
                    ax[i, j].set_yticks([75, 78, 81])
                elif dataset_name == 'support2':
                    ax[i, j].set_ylim(62, 74)
                    ax[i, j].set_yticks([65, 68, 71])
                elif dataset_name == 'framingham':
                    ax[i, j].set_ylim(70.5, 75.5)
                    ax[i, j].set_yticks([71, 73, 75])
                elif dataset_name == 'breast_cancer_metabric':
                    ax[i, j].set_ylim(60, 72)
                    ax[i, j].set_yticks([62, 66, 70])
                elif dataset_name == 'breast_cancer_metabric_relapse':
                    ax[i, j].set_ylim(56, 66)
                    ax[i, j].set_yticks([58, 61, 64])

                #ax[i, j].set_ylim(50, 90)
                #ax[i, j].set_yticks([60, 70, 80])

                ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False,
                                     right=False)

                if i == 0:
                    ax[i, j].set_title(f'$k = {k}$' if k < 1000 else '$\\text{max}$')

                if i == n - 1:
                    ax[i, j].set_xlabel('Generation')
                else:
                    ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                    ax[i, j].tick_params(labelbottom=False)
                    ax[i, j].set_xticklabels([])

                if j == 0:
                    ax[i, j].set_ylabel('\\texttt{HV}')
                else:
                    ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                    ax[i, j].tick_params(labelleft=False)
                    ax[i, j].set_yticklabels([])
                    if j == m - 1:
                        # axttt = ax[i, j].twinx()
                        ax[i, j].set_ylabel(acronym, rotation=270, labelpad=14)
                        ax[i, j].yaxis.set_label_position("right")
                        ax[i, j].tick_params(labelleft=False)
                        ax[i, j].set_yticklabels([])
                        # ax[i, j].yaxis.tick_right()

                if i == n - 1 and j == m - 1:
                    ax[i, j].tick_params(pad=7)

                ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    else:
        n, m = 1, len(dataset_names)
        fig, ax = plt.subplots(n, m, figsize=(14, 3), layout='constrained', squeeze=False)
        x = list(range(1, 100 + 1))

        for i in range(m):
            dataset_name = dataset_names[i]
            acronym = dataset_acronyms[dataset_name]

            k = how_many_pareto_features[0]
            actual_data = data[dataset_name][str(k)]

            ax[0, i].plot(x, actual_data['TrainMedianHV'], label='', color='#E51D1D',
                          linestyle='-',
                          linewidth=1.0, markersize=10)
            ax[0, i].fill_between(x, actual_data['TrainQ1HV'], actual_data['TrainQ3HV'],
                                  color='#E51D1D', alpha=0.1)

            ax[0, i].plot(x, actual_data['TestMedianHV'], label='', color='#3B17F2',
                          linestyle='-',
                          linewidth=1.0, markersize=10)
            ax[0, i].fill_between(x, actual_data['TestQ1HV'], actual_data['TestQ3HV'],
                                  color='#3B17F2', alpha=0.1)

            ax[0, i].set_xlim(1, 100)
            ax[0, i].set_xticks([1, 100 // 2, 100])

            if dataset_name == 'pbc2':
                ax[0, i].set_ylim(72, 84)
                ax[0, i].set_yticks([75, 78, 81])
            elif dataset_name == 'support2':
                ax[0, i].set_ylim(62, 74)
                ax[0, i].set_yticks([65, 68, 71])
            elif dataset_name == 'framingham':
                ax[0, i].set_ylim(72, 76)
                ax[0, i].set_yticks([73, 74, 75])
            elif dataset_name == 'breast_cancer_metabric':
                ax[0, i].set_ylim(63, 71)
                ax[0, i].set_yticks([65, 67, 69])
            elif dataset_name == 'breast_cancer_metabric_relapse':
                ax[0, i].set_ylim(58, 66)
                ax[0, i].set_yticks([59, 62, 65])

            # ax[i, j].set_ylim(50, 90)
            # ax[i, j].set_yticks([60, 70, 80])

            ax[0, i].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False,
                                 right=False)

            ax[0, i].set_title(acronym)

            ax[0, i].set_xlabel('Generation')

            if i == 0:
                ax[0, i].set_ylabel('\\texttt{HV}')
            else:
                ax[0, i].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                #ax[0, i].tick_params(labelleft=False)
                #ax[0, i].set_yticklabels([])
            if i == m - 1:
                ax[0, i].tick_params(pad=7)

            ax[0, i].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)


def main():
    base_path: str = '../SurvivalMultiTree-pyNSGP-DATA/results/'

    with open(os.path.join(base_path, 'config_coxnet.yaml'), 'r') as yaml_file:
        try:
            coxnet_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise exc

    with open(os.path.join(base_path, 'config_nsgp.yaml'), 'r') as yaml_file:
        try:
            nsgp_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise exc

    with open(os.path.join(base_path, 'config_survivaltree.yaml'), 'r') as yaml_file:
        try:
            survivaltree_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise exc

    with open(os.path.join(base_path, 'config_gradientboost.yaml'), 'r') as yaml_file:
       try:
           gradientboost_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
       except yaml.YAMLError as exc:
           raise exc

    with open(os.path.join(base_path, 'config_randomforest.yaml'), 'r') as yaml_file:
       try:
           randomforest_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
       except yaml.YAMLError as exc:
           raise exc

    test_size: float = 0.3

    pop_size: int = nsgp_config_dict['pop_size']
    num_gen: int = nsgp_config_dict['num_gen']
    max_size: int = nsgp_config_dict['max_size']
    min_depth: int = nsgp_config_dict['min_depth']
    init_max_height: int = nsgp_config_dict['init_max_height']
    tournament_size: int = nsgp_config_dict['tournament_size']
    min_trees_init: int = nsgp_config_dict['min_trees_init']
    max_trees_init: int = nsgp_config_dict['max_trees_init']
    alpha: float = nsgp_config_dict['alpha']
    max_iter_nsgp: int = nsgp_config_dict['max_iter']
    l1_ratio_nsgp: float = nsgp_config_dict['l1_ratio']

    l1_ratio: float = coxnet_config_dict['l1_ratio']
    n_alphas: int = coxnet_config_dict['n_alphas']
    alpha_min_ratio: float = coxnet_config_dict['alpha_min_ratio']
    max_iter: int = coxnet_config_dict['max_iter']

    n_max_depths_st: int = survivaltree_config_dict['n_max_depths']
    n_folds_st: int = survivaltree_config_dict['n_folds']

    n_max_depths_gb: int = gradientboost_config_dict['n_max_depths']
    n_folds_gb: int = gradientboost_config_dict['n_folds']

    n_max_depths_rf: int = randomforest_config_dict['n_max_depths']
    n_folds_rf: int = randomforest_config_dict['n_folds']

    split_types: list[str] = ['Train', 'Test']
    dataset_names: list[str] = ['pbc2', 'support2', 'framingham', 'breast_cancer_metabric', 'breast_cancer_metabric_relapse']
    seed_range: list[int] = list(range(1, 50 + 1))
    normalizes: list[bool] = [True, False]

    dataset_names_acronyms: dict[str,str] = {
        'pbc2': r'PBC',
        'support2': r'SPP',
        'framingham': r'FRM',
        'breast_cancer_metabric': r'BCM',
        'breast_cancer_metabric_relapse': r'BCR'
    }

    methods_acronyms = {'randomsearch': 'RS', 'coxnet': 'CX', 'nsgp': 'SR', 'survivaltree': 'ST', 'gradientboost': 'GB', 'randomforest': 'RF'}

    palette_boxplot = {'$k = 3$': '#C5F30C',
                       '$k = 4$': '#31AB0C',
                       '$k = 5$': '#283ADF',
                       }
    palette_methods = {'coxnet': '#CBE231',
                       'survivaltree': '#E55E0F',
                       'nsgp': '#33B9F2',
                       'gradientboost': '#0B1DBA',
                       'randomforest': '#6B2106'}

    # create_lineplots_on_single_line_multitree_length(
    #     base_path=base_path,
    #     test_size=test_size,
    #     dataset_names=dataset_names,
    #     seed_range=seed_range,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     alpha=alpha,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     how_many_pareto_features=[3, 4, 5, 6, 7],
    #     dataset_names_acronyms=dataset_names_acronyms,
    # )

    median_pareto_front_all_methods(
        base_path=base_path,
        test_size=test_size,
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter,
        dataset_names=dataset_names,
        dataset_names_acronyms=dataset_names_acronyms,
        split_type='Test',
        seed_range=seed_range,
        pop_size=pop_size,
        num_gen=num_gen,
        max_size=max_size,
        min_depth=min_depth,
        init_max_height=init_max_height,
        tournament_size=tournament_size,
        min_trees_init=min_trees_init,
        max_trees_init=max_trees_init,
        alpha=alpha,
        l1_ratio_nsgp=l1_ratio_nsgp,
        max_iter_nsgp=max_iter_nsgp,
        n_max_depths_st=n_max_depths_st,
        n_folds_st=n_folds_st,
        n_max_depths_gb=n_max_depths_gb,
        n_folds_gb=n_folds_gb,
        n_max_depths_rf=n_max_depths_rf,
        n_folds_rf=n_folds_rf,
        palette_methods=palette_methods,
    )

    # print_some_formulae(
    #     base_path=base_path,
    #     test_size=test_size,
    #     normalize=False,
    #     dataset_names=dataset_names,
    #     seed=13,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     alpha=alpha,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     how_many_pareto_features=[3, 5, 7],
    #     dataset_names_acronyms=dataset_names_acronyms
    # )

    # stat_test_print(
    #     base_path=base_path,
    #     test_size=test_size,
    #     n_alphas=n_alphas,
    #     l1_ratio=l1_ratio,
    #     alpha_min_ratio=alpha_min_ratio,
    #     max_iter=max_iter,
    #     normalizes=normalizes,
    #     split_types=split_types,
    #     dataset_names=dataset_names,
    #     seed_range=seed_range,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     alpha=alpha,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     how_many_pareto_features=2,
    #     compare_single_points=True,
    #     how_many_pareto_features_boxplot=[3, 4, 5],
    #     dataset_names_acronyms=dataset_names_acronyms,
    #     palette_boxplot=palette_boxplot
    # )

    # rs_values, rs_comparisons = stat_test(
    #     base_path=base_path,
    #     test_size=test_size,
    #     n_alphas=n_alphas,
    #     l1_ratio=l1_ratio,
    #     alpha_min_ratio=alpha_min_ratio,
    #     max_iter=max_iter,
    #     normalizes=normalizes,
    #     dataset_names=dataset_names,
    #     seed_range=seed_range,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     alpha=alpha,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     n_max_depths_st=n_max_depths_st,
    #     n_folds_st=n_folds_st,
    #     n_max_depths_gb=n_max_depths_gb,
    #     n_folds_gb=n_folds_gb,
    #     n_max_depths_rf=n_max_depths_rf,
    #     n_folds_rf=n_folds_rf,
    #     how_many_pareto_features_table=[1, 2, 3, 4, 5, 1000],
    #     methods=['randomsearch', 'nsgp'],
    # )
    #
    # with open('rs_values.json', 'w') as f:
    #     json.dump(rs_values, f, indent=4)
    # with open('rs_comparisons.json', 'w') as f:
    #     json.dump(rs_comparisons, f, indent=4)

    # with open('rs_values.json', 'r') as f:
    #     rs_values = json.load(f)
    # with open('rs_comparisons.json', 'r') as f:
    #     rs_comparisons = json.load(f)


    # white_values, white_comparisons = stat_test(
    #     base_path=base_path,
    #     test_size=test_size,
    #     n_alphas=n_alphas,
    #     l1_ratio=l1_ratio,
    #     alpha_min_ratio=alpha_min_ratio,
    #     max_iter=max_iter,
    #     normalizes=normalizes,
    #     dataset_names=dataset_names,
    #     seed_range=seed_range,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     alpha=alpha,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     n_max_depths_st=n_max_depths_st,
    #     n_folds_st=n_folds_st,
    #     n_max_depths_gb=n_max_depths_gb,
    #     n_folds_gb=n_folds_gb,
    #     n_max_depths_rf=n_max_depths_rf,
    #     n_folds_rf=n_folds_rf,
    #     how_many_pareto_features_table=[1, 2, 3, 4, 5, 6, 7, 1000],
    #     methods=['survivaltree', 'coxnet', 'nsgp'],
    # )
    #
    # with open('white_values.json', 'w') as f:
    #     json.dump(white_values, f, indent=4)
    # with open('white_comparisons.json', 'w') as f:
    #     json.dump(white_comparisons, f, indent=4)

    with open('white_values.json', 'r') as f:
        white_values = json.load(f)
    with open('white_comparisons.json', 'r') as f:
        white_comparisons = json.load(f)

    # black_values, black_comparisons = stat_test(
    #     base_path=base_path,
    #     test_size=test_size,
    #     n_alphas=n_alphas,
    #     l1_ratio=l1_ratio,
    #     alpha_min_ratio=alpha_min_ratio,
    #     max_iter=max_iter,
    #     normalizes=normalizes,
    #     dataset_names=dataset_names,
    #     seed_range=seed_range,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     alpha=alpha,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     n_max_depths_st=n_max_depths_st,
    #     n_folds_st=n_folds_st,
    #     n_max_depths_gb=n_max_depths_gb,
    #     n_folds_gb=n_folds_gb,
    #     n_max_depths_rf=n_max_depths_rf,
    #     n_folds_rf=n_folds_rf,
    #     how_many_pareto_features_table=[1, 2, 3, 4, 5, 6, 7, 1000],
    #     methods=['gradientboost', 'randomforest', 'nsgp'],
    # )
    #
    # with open('black_values.json', 'w') as f:
    #     json.dump(black_values, f, indent=4)
    # with open('black_comparisons.json', 'w') as f:
    #     json.dump(black_comparisons, f, indent=4)

    with open('black_values.json', 'r') as f:
        black_values = json.load(f)
    with open('black_comparisons.json', 'r') as f:
        black_comparisons = json.load(f)

    with open('lineplot_data.json', 'r') as f:
        lineplot_data = json.load(f)

    with open(f'survival_function_data_seed{25}.json', 'r') as f:
        surv_func_data = json.load(f)

    # create_survival_function_lineplot(surv_func_data, palette_methods)

    # print_table_hv_ci(
    #     values=black_values,
    #     comparisons=black_comparisons,
    #     methods=['gradientboost', 'randomforest', 'nsgp'],
    #     methods_acronyms=methods_acronyms,
    #     how_many_pareto_features_table=[3, 5, 7, 1000],
    #     normalizes=normalizes,
    #     dataset_names=dataset_names,
    # )

    # lineplot(
    #     base_path=base_path,
    #     data=lineplot_data,
    #     test_size=test_size,
    #     normalize=False,
    #     dataset_names=dataset_names,
    #     seed_range=seed_range,
    #     pop_size=pop_size,
    #     num_gen=num_gen,
    #     max_size=max_size,
    #     min_depth=min_depth,
    #     init_max_height=init_max_height,
    #     tournament_size=tournament_size,
    #     min_trees_init=min_trees_init,
    #     max_trees_init=max_trees_init,
    #     l1_ratio_nsgp=l1_ratio_nsgp,
    #     max_iter_nsgp=max_iter_nsgp,
    #     alpha=alpha,
    #     dataset_acronyms=dataset_names_acronyms,
    #     how_many_pareto_features=[1000],
    # )

   # for seed_surv_func in [1, 10, 25, 50]:
   #     create_survival_function(
   #         base_path=base_path,
   #         test_size=test_size,
   #         dataset_name='framingham',
   #         seed=seed_surv_func,
   #         n_alphas=n_alphas,
   #         l1_ratio=l1_ratio,
   #         alpha_min_ratio=alpha_min_ratio,
   #         max_iter=max_iter,
   #         pop_size=pop_size,
   #         num_gen=num_gen,
   #         max_size=max_size,
   #         min_depth=min_depth,
   #         init_max_height=init_max_height,
   #         tournament_size=tournament_size,
   #         min_trees_init=min_trees_init,
   #         max_trees_init=max_trees_init,
   #         alpha=alpha,
   #         l1_ratio_nsgp=l1_ratio_nsgp,
   #         max_iter_nsgp=max_iter_nsgp,
   #         n_max_depths_st=n_max_depths_st,
   #         n_folds_st=n_folds_st,
   #         n_max_depths_gb=n_max_depths_gb,
   #         n_folds_gb=n_folds_gb,
   #         n_max_depths_rf=n_max_depths_rf,
   #         n_folds_rf=n_folds_rf,
   #     )




if __name__ == '__main__':
    main()

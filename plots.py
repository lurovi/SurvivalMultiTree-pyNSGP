import json
import statistics
from typing import Any

import fastplot
import numpy as np
import pandas as pd
import seaborn as sns
import os

from pymoo.indicators.hv import HV
from pynsgp.Utils.pickle_persist import decompress_pickle, decompress_dill
from pynsgp.Utils.data import load_dataset, nsgp_path_string, cox_net_path_string, survival_ensemble_tree_path_string
from pynsgp.Utils.stats import is_mannwhitneyu_passed, is_kruskalwallis_passed, perform_mannwhitneyu_holm_bonferroni

import warnings
import yaml

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



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
    PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

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

    # offset = 0
    # while True:
    #     try:
    #         ind = n_features.index(n_features_to_consider - offset)
    #         break
    #     except ValueError:
    #         offset += 1
    #         if n_features_to_consider - offset <= 0:
    #             ind = -1
    #             break
    #
    # if ind == -1:
    #     raise ValueError(f'No existing features here.')

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
                X, _ = load_dataset(dataset_name)
                n_features = X.shape[1]
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

        PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

        if split_type.lower() == 'test':
            fastplot.plot(None, f'boxplot.pdf', mode='callback', callback=lambda plt: my_callback_boxplot(plt, boxplot_data, palette_boxplot), style='latex', **PLOT_ARGS)


def my_callback_boxplot(plt, data, palette):
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.set_ylim(0, 13)
    ax.set_yticks(list(range(1, 12 + 1)))
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    sns.boxplot(pd.DataFrame(data), x='Dataset', y='Length of Multi Tree', hue='k', palette=palette, legend=False, log_scale=None, fliersize=0.0, showfliers=False, ax=ax)


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

                    X, _ = load_dataset(dataset_name)
                    n_features = X.shape[1]
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
                            else:
                                raise ValueError(f'Unrecognized method {method}.')

                            compared_c_indexes = [-temp_error for temp_error, temp_feats in zip(errors_list, n_features_list) if temp_feats == (k if k < 1000 else max(n_features_list))]
                            if len(compared_c_indexes) == 0:
                                compared_c_indexes = [0.0]
                            compared_c_index = compared_c_indexes[0]
                            values[split_type]['CI'][k][normalize][dataset_name][method].append(compared_c_index)

                            cx_pairs_pareto = np.array([[temp_error, temp_feats] for temp_error, temp_feats in zip(errors_list, n_features_list) if temp_feats <= k])
                            if len(cx_pairs_pareto) == 0:
                                cx_pairs_pareto = np.array([ref_point])
                            values[split_type]['HV'][k][normalize][dataset_name][method].append(HV(ref_point)(cx_pairs_pareto))

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
                    ci_median = statistics.median(values['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method])
                    hv_num_outperformed_methods = len(comparisons['Test']['HV'][str(k)][str(normalize).lower()][dataset_name][method])
                    ci_num_outperformed_methods = len(comparisons['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method])
                    hv_median_max = max([statistics.median(values['Test']['HV'][str(k)][str(normalize).lower()][dataset_name][method_2]) for method_2 in methods])
                    ci_median_max = max([statistics.median(values['Test']['CI'][str(k)][str(normalize).lower()][dataset_name][method_2]) for method_2 in methods])
                    hv_bold = ''
                    ci_bold = ''
                    hv_star = ''
                    ci_star = ''
                    if hv_median == hv_median_max:
                        hv_bold = '\\bfseries '
                    if ci_median == ci_median_max:
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
                    ci_interpret_table += ' & ' + ci_bold + str(round(ci_median, 3)) + ci_star

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
    #dataset k split_type+aggregation+metric values_for_each_generation
    data = {
        dataset_name: {
            k: {
                'TrainMedianHV': [], 'TrainQ1HV': [], 'TrainQ3HV': [],
                'TestMedianHV': [], 'TestQ1HV': [], 'TestQ3HV': []
            }
            for k in how_many_pareto_features
        }
        for dataset_name in dataset_names
    }

    for k in how_many_pareto_features:
        for dataset_name in dataset_names:

            X, _ = load_dataset(dataset_name)
            n_features = X.shape[1]
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
                data[dataset_name][k]['TrainMedianHV'].append(statistics.median(l))
                data[dataset_name][k]['TrainQ1HV'].append(float(np.percentile(l, 25)))
                data[dataset_name][k]['TrainQ3HV'].append(float(np.percentile(l, 75)))
            for l in test_single_values:
                data[dataset_name][k]['TestMedianHV'].append(statistics.median(l))
                data[dataset_name][k]['TestQ1HV'].append(float(np.percentile(l, 25)))
                data[dataset_name][k]['TestQ3HV'].append(float(np.percentile(l, 75)))

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}

    fastplot.plot(None, f'lineplot.pdf', mode='callback', callback=lambda plt: my_callback_lineplot(plt, data, how_many_pareto_features, dataset_names, dataset_acronyms), style='latex', **PLOT_ARGS)


def my_callback_lineplot(plt, data, how_many_pareto_features, dataset_names, dataset_acronyms):
    n, m = len(dataset_names), len(how_many_pareto_features)
    fig, ax = plt.subplots(n, m, figsize=(10, 10), layout='constrained', squeeze=False)
    x = list(range(1, 100 + 1))

    met_i = 0
    for i in range(n):
        dataset_name = dataset_names[i]
        acronym = dataset_acronyms[dataset_name]
        for j in range(m):
            k = how_many_pareto_features[j]
            actual_data = data[dataset_name][k]

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
                ax[i, j].set_ylim(17, 21)
                ax[i, j].set_yticks([18, 19, 20])
            elif dataset_name == 'support2':
                ax[i, j].set_ylim(27, 33)
                ax[i, j].set_yticks([28, 30, 32])
            elif dataset_name == 'framingham':
                ax[i, j].set_ylim(21, 21.8)
                ax[i, j].set_yticks([21.2, 21.4, 21.6])
            elif dataset_name == 'breast_cancer_metabric':
                ax[i, j].set_ylim(40, 48)
                ax[i, j].set_yticks([41, 45, 47])
            elif dataset_name == 'breast_cancer_metabric_relapse':
                ax[i, j].set_ylim(36, 44)
                ax[i, j].set_yticks([37, 40, 43])

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
                # ax[i, j].tick_params(labelleft=False)
                # ax[i, j].set_yticklabels([])
                if j == m - 1:
                    # axttt = ax[i, j].twinx()
                    ax[i, j].set_ylabel(acronym, rotation=270, labelpad=14)
                    ax[i, j].yaxis.set_label_position("right")
                    # ax[i, j].tick_params(labelleft=False)
                    # ax[i, j].set_yticklabels([])
                    # ax[i, j].yaxis.tick_right()

            if i == n - 1 and j == m - 1:
                ax[i, j].tick_params(pad=7)

            ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)

            met_i += 1



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

    # with open(os.path.join(base_path, 'config_survivaltree.yaml'), 'r') as yaml_file:
    #     try:
    #         survivaltree_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
    #     except yaml.YAMLError as exc:
    #         raise exc

    # with open(os.path.join(base_path, 'config_gradientboost.yaml'), 'r') as yaml_file:
    #    try:
    #        gradientboost_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
    #    except yaml.YAMLError as exc:
    #        raise exc

    # with open(os.path.join(base_path, 'config_randomforest.yaml'), 'r') as yaml_file:
    #    try:
    #        randomforest_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
    #    except yaml.YAMLError as exc:
    #        raise exc

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

    # n_max_depths_st: int = survivaltree_config_dict['n_max_depths']
    # n_folds_st: int = survivaltree_config_dict['n_folds']

    # n_max_depths_gb: int = gradientboost_config_dict['n_max_depths']
    # n_folds_gb: int = gradientboost_config_dict['n_folds']

    # n_max_depths_rf: int = randomforest_config_dict['n_max_depths']
    # n_folds_rf: int = randomforest_config_dict['n_folds']

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

    methods_acronyms = {'coxnet': 'CX', 'nsgp': 'MT', 'survivaltree': 'ST'}

    palette_boxplot = {'$k = 3$': '#C5F30C',
                       '$k = 4$': '#31AB0C',
                       '$k = 5$': '#283ADF',
                       }

    # print_some_formulae(
    #     base_path=base_path,
    #     test_size=test_size,
    #     normalize=False,
    #     dataset_names=dataset_names,
    #     seed=7,
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
    #     how_many_pareto_features=[3, 4, 5],
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

    # values, comparisons = stat_test(
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
    #     methods=['coxnet', 'nsgp'],
    # )
    #
    # with open('values.json', 'w') as f:
    #     json.dump(values, f, indent=4)
    # with open('comparisons.json', 'w') as f:
    #     json.dump(comparisons, f, indent=4)

    with open('values.json', 'r') as f:
        values = json.load(f)
    with open('comparisons.json', 'r') as f:
        comparisons = json.load(f)


    # print_table_hv_ci(
    #     values=values,
    #     comparisons=comparisons,
    #     methods=['coxnet', 'nsgp'],
    #     methods_acronyms=methods_acronyms,
    #     how_many_pareto_features_table=[1, 2, 3, 4, 5, 1000],
    #     normalizes=normalizes,
    #     dataset_names=dataset_names,
    # )

    lineplot(
        base_path=base_path,
        test_size=test_size,
        normalize=False,
        dataset_names=dataset_names,
        seed_range=seed_range,
        pop_size=pop_size,
        num_gen=num_gen,
        max_size=max_size,
        min_depth=min_depth,
        init_max_height=init_max_height,
        tournament_size=tournament_size,
        min_trees_init=min_trees_init,
        max_trees_init=max_trees_init,
        l1_ratio_nsgp=l1_ratio_nsgp,
        max_iter_nsgp=max_iter_nsgp,
        alpha=alpha,
        dataset_acronyms=dataset_names_acronyms,
        how_many_pareto_features=[3, 4, 5, 1000],
    )




if __name__ == '__main__':
    main()

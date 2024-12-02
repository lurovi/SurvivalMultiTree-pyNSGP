from typing import Any

import fastplot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import yaml

from pynsgp.Utils.data import nsgp_path_string, cox_net_path_string



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

    ax.set_ylim(0, 15)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_xlim(-1.0, -0.4)
    ax.set_xticks([-0.9, -0.8, -0.7, -0.6, -0.5])
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



def read_coxnet(
    base_path: str,
    method: str,
    dataset_name: str,
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
        test_size=test_size,
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter
    )
    data = pd.read_csv(os.path.join(path, f'output_seed{seed}.csv'), sep=',')
    return data


def read_nsgp(
        base_path: str,
        method: str,
        dataset_name: str,
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
        seed: int
):
    path = nsgp_path_string(
        base_path=base_path,
        method=method,
        dataset_name=dataset_name,
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
    return data


if __name__ == '__main__':
    base_path: str = 'results/'

    with open('config_coxnet.yaml', 'r') as yaml_file:
        try:
            coxnet_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise exc

    with open('config_nsgp.yaml', 'r') as yaml_file:
        try:
            nsgp_config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise exc

    test_size: int = 0.3

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

    for split_type in ['Train', 'Test']:
        for dataset_name in ['whas500', 'gbsg2', 'veterans_lung_cancer']:
            for seed in range(1, 5 + 1):
                coxd = read_coxnet(
                    base_path=base_path,
                    method='coxnet',
                    dataset_name=dataset_name,
                    test_size=test_size,
                    n_alphas=n_alphas,
                    l1_ratio=l1_ratio,
                    alpha_min_ratio=alpha_min_ratio,
                    max_iter=max_iter,
                    seed=seed
                )

                coxnet_n_features_list = list(coxd['DistinctRawFeatures'])
                coxnet_errors_list = list(coxd[split_type + 'Error'])

                nsgpd = read_nsgp(
                    base_path=base_path,
                    method='nsgp',
                    dataset_name=dataset_name,
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
                    seed=seed
                )

                nsgp_n_features_list = [float(val) for val in nsgpd.loc[num_gen - 1, 'ParetoObj2'].split(' ')]
                nsgp_errors_list = [float(val) for val in nsgpd.loc[num_gen - 1, split_type + 'ParetoObj1'].split(' ')]

                scatter_plot(dataset=dataset_name, seed=seed, split_type=split_type.strip(), coxnet_n_features_list=coxnet_n_features_list, coxnet_errors_list=coxnet_errors_list, nsgp_n_features_list=nsgp_n_features_list, nsgp_errors_list=nsgp_errors_list)


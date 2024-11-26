from genepro.variation import (
    safe_subtree_crossover,
    node_level_crossover,
    subtree_mutation,
    one_point_mutation,
    coeff_mutation,
)
import warnings
import yaml
import argparse
from argparse import ArgumentParser, Namespace

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from methods import run_evolution, run_cox_net


if __name__ == '__main__':
    arg_parser: ArgumentParser = ArgumentParser(description="SurvivalMultiTree-pyNSGP arguments.")
    arg_parser.add_argument("--method", type=str,
                           help=f"The method name.")
    arg_parser.add_argument("--dataset", type=str,
                           help=f"The dataset name.")
    arg_parser.add_argument("--seed", type=int,
                           help=f"The random seed.")
    arg_parser.add_argument("--test_size", type=float,
                           help="Percentage of the dataset to use as test set.")
    arg_parser.add_argument("--config", type=str,
                           help="Path to the .yaml configuration file with the method specific parameters.")
    cmd_args: Namespace = arg_parser.parse_args()
    
    results_path: str = 'results/'

    corr_drop_threshold: float = 0.98
    scale_numerical: bool = True

    method: str = 'coxnet'
    random_state: int = 5
    dataset_name: str = 'whas500'
    test_size: float = 0.3

    pop_size: int = 100
    num_gen: int = 50
    max_size: int = 40
    min_depth: int = 2
    init_max_height: int = 4
    tournament_size: int = 3
    min_trees_init: int = 3
    max_trees_init: int = 6
    alpha: float = float(1e-6)
    max_iter_nsgp: int = 100

    l1_ratio: float = 0.9

    n_alphas: int = 1000
    alpha_min_ratio: float = 0.1
    max_iter: int = 1000000

    verbose: bool = True

    crossovers = [
        {"fun": node_level_crossover, "rate": 0.25},
        {"fun": safe_subtree_crossover, "rate": 0.05, "kwargs": {"max_depth": init_max_height}},
    ]
    mutations = [
        {"fun": subtree_mutation, "rate": 0.25, "kwargs": {"max_depth": init_max_height}},
        {"fun": one_point_mutation, "rate": 0.25},
    ]
    coeff_opts = [
        {
            "fun": coeff_mutation,
            "rate": 0.9,
            "kwargs": {
                "prob_coeff_mut": 0.5,
                "temp": 0.1,
            },
        },
    ]

    if method == 'nsgp':
        run_evolution(
            results_path=results_path,
            crossovers=crossovers,
            mutations=mutations,
            coeff_opts=coeff_opts,
            corr_drop_threshold=corr_drop_threshold,
            scale_numerical=scale_numerical,
            random_state=random_state,
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
            max_iter=max_iter_nsgp,
            verbose=verbose
        )
    elif method == 'coxnet':
        run_cox_net(
            results_path=results_path,
            corr_drop_threshold=corr_drop_threshold,
            scale_numerical=scale_numerical,
            random_state=random_state,
            dataset_name=dataset_name,
            test_size=test_size,
            n_alphas=n_alphas,
            l1_ratio=l1_ratio,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter,
            verbose=verbose
        )
    else:
        raise AttributeError(f'Unrecognized method in main {method}.')

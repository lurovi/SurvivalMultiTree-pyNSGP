import os
import traceback
from typing import Any

from genepro.variation import (
    safe_subtree_crossover,
    node_level_crossover,
    subtree_mutation,
    one_point_mutation,
    coeff_mutation,
)
import warnings
import yaml
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
    arg_parser.add_argument("--seed", type=int,
                           help=f"The random seed.")
    arg_parser.add_argument("--dataset", type=str,
                            help=f"The dataset name.")
    arg_parser.add_argument("--test_size", type=float,
                           help="Percentage of the dataset to use as test set.")
    arg_parser.add_argument("--config", type=str,
                           help="Path to the .yaml configuration file with the method specific parameters.")
    arg_parser.add_argument("--run_id", type=int,
                            help="The run id, used for logging purposes of successful runs. By default is 0.")
    arg_parser.add_argument("--verbose", type=int,
                            help="Whether or not print progress for each generation/iteration.")

    cmd_args: Namespace = arg_parser.parse_args()

    if cmd_args.method is None:
        raise AttributeError(f'Method not provided.')
    if cmd_args.seed is None:
        raise AttributeError(f'Seed not provided.')
    if cmd_args.dataset is None:
        raise AttributeError(f'Dataset not provided.')
    if cmd_args.test_size is None:
        raise AttributeError(f'Test size not provided.')
    if cmd_args.config is None:
        raise AttributeError(f'Configuration .yaml file not provided.')
    if cmd_args.run_id is None:
        run_id: int = 0
    else:
        run_id: int = cmd_args.run_id
    if cmd_args.verbose is None:
        verbose: bool = False
    else:
        verbose: bool = True if cmd_args.verbose != 0 else False

    results_path: str = 'results/'
    run_with_exceptions_path: str = 'run_with_exceptions/'

    if not os.path.isdir(results_path):
        os.makedirs(results_path, exist_ok=True)

    if not os.path.isdir(run_with_exceptions_path):
        os.makedirs(run_with_exceptions_path, exist_ok=True)

    corr_drop_threshold: float = 0.98
    scale_numerical: bool = True

    method: str = cmd_args.method
    random_state: int = cmd_args.seed
    dataset_name: str = cmd_args.dataset
    test_size: float = cmd_args.test_size
    config_file_with_params: str = cmd_args.config

    with open(config_file_with_params, 'r') as yaml_file:
        try:
            config_dict: dict[str, Any] = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise exc

    run_string_descr: str = ','.join([method, str(random_state), dataset_name, str(test_size), config_file_with_params, str(int(verbose)), str(run_id)])

    try:
        if method == 'nsgp':

            pop_size: int = config_dict['pop_size']
            num_gen: int = config_dict['num_gen']
            max_size: int = config_dict['max_size']
            min_depth: int = config_dict['min_depth']
            init_max_height: int = config_dict['init_max_height']
            tournament_size: int = config_dict['tournament_size']
            min_trees_init: int = config_dict['min_trees_init']
            max_trees_init: int = config_dict['max_trees_init']
            alpha: float = config_dict['alpha']
            max_iter_nsgp: int = config_dict['max_iter']
            l1_ratio_nsgp: float = config_dict['l1_ratio']

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
                l1_ratio=l1_ratio_nsgp,
                max_iter=max_iter_nsgp,
                verbose=verbose
            )

        elif method == 'coxnet':

            l1_ratio: float = config_dict['l1_ratio']
            n_alphas: int = config_dict['n_alphas']
            alpha_min_ratio: float = config_dict['alpha_min_ratio']
            max_iter: int = config_dict['max_iter']

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

        with open(os.path.join(results_path, f'completed_run{run_id}.txt'), 'a+') as terminal_std_out:
            terminal_std_out.write(run_string_descr)
            terminal_std_out.write('\n')
    except Exception:
        error_string = str(traceback.format_exc())
        with open(os.path.join(run_with_exceptions_path, f'{run_string_descr.replace(",", "___")}'), 'w') as f:
            f.write(error_string)

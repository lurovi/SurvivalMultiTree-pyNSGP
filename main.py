import os
import traceback
import cProfile
from typing import Any

from genepro.variation import (
    safe_subtree_crossover,
    node_level_crossover,
    subtree_mutation,
    one_point_mutation,
    coeff_mutation,
)
import warnings
import zlib
import threading
import yaml
from argparse import ArgumentParser, Namespace

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from methods import run_evolution, run_cox_net, run_survival_ensemble_tree
from pynsgp.Utils.string_utils import is_valid_filename

completed_csv_lock = threading.Lock()


def main():

    # ========================================================================================================================================
    # PARSING ARGUMENTS FROM COMMAND-LINE
    # ========================================================================================================================================

    arg_parser: ArgumentParser = ArgumentParser(description="SurvivalMultiTree-pyNSGP arguments.")
    arg_parser.add_argument("--method", type=str, help=f"The method name.")
    arg_parser.add_argument("--seed", type=int, help=f"The random seed.")
    arg_parser.add_argument("--dataset", type=str, help=f"The dataset name.")
    arg_parser.add_argument("--normalize", type=int, help=f"Whether or not normalize data when passing it to coxnet.")
    arg_parser.add_argument("--test_size", type=float, help="Percentage of the dataset to use as test set.")
    arg_parser.add_argument("--config", type=str, help="Path to the .yaml configuration file with the method specific parameters.")
    arg_parser.add_argument("--run_id", type=str, default='default', help="The run id, used for logging purposes of successful runs.")
    arg_parser.add_argument("--verbose", required=False, action="store_true", help="Verbose flag.")
    arg_parser.add_argument("--profile", required=False, action="store_true", help="Whether to run and log profiling of code or not.")

    cmd_args: Namespace = arg_parser.parse_args()

    if cmd_args.method is None:
        raise AttributeError(f'Method not provided.')
    if cmd_args.seed is None:
        raise AttributeError(f'Seed not provided.')
    if cmd_args.dataset is None:
        raise AttributeError(f'Dataset not provided.')
    if cmd_args.normalize is None:
        raise AttributeError(f'Normalize not provided.')
    if cmd_args.test_size is None:
        raise AttributeError(f'Test size not provided.')
    if cmd_args.config is None:
        raise AttributeError(f'Configuration .yaml file not provided.')
    if cmd_args.run_id is None:
        run_id: str = 'default'
    else:
        run_id: str = cmd_args.run_id

    # ========================================================================================================================================
    # EVENTUALLY CREATING RESULTS AND EXCEPTIONS DIRECTORIES
    # ========================================================================================================================================
    
    results_path: str = 'results/'
    run_with_exceptions_path: str = 'run_with_exceptions/'

    if not os.path.isdir(results_path):
        os.makedirs(results_path, exist_ok=True)

    if not os.path.isdir(run_with_exceptions_path):
        os.makedirs(run_with_exceptions_path, exist_ok=True)

    # ========================================================================================================================================
    # SETTING GENERAL METHOD-INDEPENDENT PARAMETERS
    # ========================================================================================================================================

    corr_drop_threshold: float = 0.98
    scale_numerical: bool = True

    method: str = cmd_args.method
    random_state: int = cmd_args.seed
    dataset_name: str = cmd_args.dataset
    normalize: bool = True if cmd_args.normalize != 0 else False
    test_size: float = cmd_args.test_size
    config_file_with_params: str = cmd_args.config

    verbose: int = int(cmd_args.verbose)
    profiling: int = int(cmd_args.profile)

    args_string = ";".join(f"{key};{vars(cmd_args)[key]}" for key in sorted(list(vars(cmd_args).keys())) if key not in ('profile', 'verbose'))
    all_items_string = ";".join(f"{key}={value}" for key, value in vars(cmd_args).items())

    # ========================================================================================================================================
    # EXECUTING SELECTED METHOD (EVENTUAL EXCEPTIONS ARE CAPTURED AND LOGGED TO THE EXCEPTIONS FOLDER, RESULTS ARE SAVED IN THE RESULTS DIRECTORY)
    # ========================================================================================================================================

    pr = None
    if profiling != 0:
        pr = cProfile.Profile()
        pr.enable()

    try:
        if not is_valid_filename(run_id):
            raise ValueError(f'run_id {run_id} is not a valid filename.')

        if random_state < 1:
            raise AttributeError(f'seed does not start from 1, it is {random_state}.')

        with open(config_file_with_params, 'r') as yaml_file:
            config_dict: dict[str, Any] = yaml.safe_load(yaml_file) # Loading .YAML configuration file containing method-specific parameters

        # SURVIVAL MULTI-TREE NSGP
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
                {"fun": safe_subtree_crossover, "rate": 0.1, "kwargs": {"max_depth": init_max_height}},
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
                verbose=verbose,
                save_output=True,
            )

        # ELASTIC COXNET
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
                normalize=normalize,
                test_size=test_size,
                n_alphas=n_alphas,
                l1_ratio=l1_ratio,
                alpha_min_ratio=alpha_min_ratio,
                max_iter=max_iter,
                verbose=verbose
            )

        # SURVIVAL TREE WITH BLACK-BOX ENSEMBLE METHODS
        elif method in ('survivaltree', 'gradientboost', 'randomforest'):

            n_max_depths: int = config_dict['n_max_depths']
            n_folds: int = config_dict['n_folds']

            run_survival_ensemble_tree(
                 results_path=results_path,
                 method=method,
                 corr_drop_threshold=corr_drop_threshold,
                 scale_numerical=scale_numerical,
                 random_state=random_state,
                 dataset_name=dataset_name,
                 normalize=normalize,
                 test_size=test_size,
                 n_max_depths=n_max_depths,
                 n_folds=n_folds,
                 verbose=verbose
            )

        else:
            raise AttributeError(f'Unrecognized method in main {method}.')

        # Logging the parameters string of this run if the run was successful
        with completed_csv_lock:
            with open(os.path.join(results_path, f'completed_{run_id}.txt'), 'a+') as terminal_std_out:
                terminal_std_out.write(args_string)
                terminal_std_out.write('\n')
        print(f'Completed run: {all_items_string}.')
    except Exception as e:
        # Capturing the trace of the exception if an exception was raised
        try:
            error_string = str(traceback.format_exc())
            with open(os.path.join(run_with_exceptions_path, f'error_{zlib.adler32(bytes(args_string, "utf-8"))}.txt'), 'w') as f:
                f.write(all_items_string + '\n\n' + error_string)
            print(f'\nException in run: {all_items_string}.\n\n{str(e)}\n\n')
        except Exception as ee:
            with open(os.path.join(run_with_exceptions_path, 'error_in_error.txt'), 'w') as f:
                f.write(str(traceback.format_exc()) + '\n\n')
            print(str(ee))
        
    if profiling != 0:
        pr.disable()
        pr.print_stats(sort='tottime')


if __name__ == '__main__':
    main()


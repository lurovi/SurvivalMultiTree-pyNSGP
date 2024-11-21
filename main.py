from genepro.variation import (
    subtree_crossover,
    node_level_crossover,
    subtree_mutation,
    one_point_mutation,
    coeff_mutation,
)
import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from methods import run_evolution, run_cox_net


if __name__ == '__main__':

    method: str = 'survival_multitree_nsgp'

    crossovers = [
        {"fun": node_level_crossover, "rate": 0.25},
        {"fun": subtree_crossover, "rate": 0.0},
    ]
    mutations = [
        {"fun": subtree_mutation, "rate": 0.25, "kwargs": {"max_depth": 5}},
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

    corr_drop_threshold: float = 0.98
    scale_numerical: bool = True

    random_state: int = 1 ** 2
    dataset_name: str = 'whas500'
    test_size: float = 0.3

    pop_size: int = 500
    num_gen: int = 20

    if method == 'survival_multitree_nsgp':
        run_evolution(
            crossovers=crossovers,
            mutations=mutations,
            coeff_opts=coeff_opts,
            corr_drop_threshold=corr_drop_threshold,
            scale_numerical=scale_numerical,
            random_state=random_state,
            dataset_name=dataset_name,
            test_size=test_size,
            pop_size=pop_size,
            num_gen=num_gen
        )
    elif method == 'cox_net':
        run_cox_net(
            linear_model='coxnet',
            corr_drop_threshold=corr_drop_threshold,
            scale_numerical=scale_numerical,
            random_state=random_state,
            dataset_name=dataset_name,
            test_size=test_size
        )
    else:
        raise AttributeError(f'Unrecognized method in main {method}.')

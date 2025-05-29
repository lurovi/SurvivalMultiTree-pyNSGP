from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP
from genepro.node_impl import *
from genepro.variation import (
    safe_subtree_crossover,
    node_level_crossover,
    subtree_mutation,
    one_point_mutation,
    coeff_mutation,
)
from methods import load_preprocess_data, set_random_seed
import warnings
from sympy.parsing.sympy_parser import parse_expr
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def example():

    random_state = 42
    set_random_seed(random_state)

    X_train, X_test, y_train, y_test = load_preprocess_data( # Or whathever dataset you have locally
        corr_drop_threshold=0.98,
        scale_numerical=True,
        random_state=random_state,
        dataset_name='whas500',
        test_size=0.3
    )
    
    crossovers = [
        {"fun": node_level_crossover, "rate": 0.25},
        {"fun": safe_subtree_crossover, "rate": 0.1, "kwargs": {"max_depth": 2}},
    ]
    
    mutations = [
        {"fun": subtree_mutation, "rate": 0.25, "kwargs": {"max_depth": 2}},
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

    nsgp = NSGP(
        path='results/', # Ignored if save_output is False
        pareto_file_name='pareto', # Ignored if save_output is False
        output_file_name='best.csv', # Ignored if save_output is False
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        crossovers=crossovers,
        mutations=mutations,
        coeff_opts=coeff_opts,
        pop_size=100,
        max_generations=20,
        max_evaluations=-1,
        max_time=-1,
        functions=[Plus(), Minus(), Times(), AnalyticQuotient(), Square(), Log()],
        use_erc=True,
        error_metric='cindex_ipcw',
        size_metric='distinct_raw_features',
        prob_delete_tree=0.05,
        prob_init_tree=0.05,
        prob_mt_crossover=0.1,
        initialization_max_tree_height=2,
        min_depth=1,
        tournament_size=4,
        max_tree_size=7,
        partition_features=False,
        min_trees_init=1,
        max_trees_init=4,
        penalize_duplicates=True,
        verbose=True,
        alpha=0.000001,
        n_iter=1000,
        l1_ratio=0.5,
        normalize=True,
        save_output=False
    )
    nsgp.fit(X_train, y_train) # It actually uses the X_train and y_train you pass in the constructor
    
    output = nsgp.nsgp_
    best_front = sorted(output.latest_front, key=lambda x: -x.objectives[0])
    
    print(f"Number of solutions in the front: {len(best_front)}.")
    print()
    print(nsgp)
    print()
    for i in range(len(best_front)):
        solution = best_front[i]
        print(f'SOLUTION {i}')
        print(f'{str(solution)}')
        print(f'All coefficients (including zeros): {solution.coefficients}.')
        print(f'Offset: {solution.offset}.')
        print(f'Number of trees: {solution.number_of_actual_trees()}.')
        print(f'Size of the tree with the maximum number of nodes: {len(solution)}.')
        print(f'Latex formula:')
        print(f'{solution.latex_expression()}')
        print()
        print()
    print()
    print()
    # Taking the multi-tree in the front with the highest accuracy
    best_solution = best_front[-1]
    # Printing each tree in the best multi-tree
    trees = best_solution.trees
    for i in range(len(trees)):
        tree = trees[i]
        print(f'TREE {i}')
        print(f'{tree.get_readable_repr()}')
        sympy_formula = parse_expr(tree.get_readable_repr().replace('^', '**'), evaluate=True)
        print(f'Sympy Formula: {sympy_formula}.')


if __name__ == '__main__':
    example()


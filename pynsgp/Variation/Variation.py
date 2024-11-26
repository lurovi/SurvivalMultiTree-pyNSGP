from typing import List, Tuple
from numpy.random import randint
from pynsgp.Utils.rand_util import choice
from pynsgp.Nodes.MultiTree import MultiTree

from genepro.variation import *
from genepro.variation import __check_tree_meets_all_constraints
from genepro.node_impl import Plus, Minus, Times, Feature, Constant, OOHRdyFeature, InstantiableConstant


def CreateDummyTree():
    return InstantiableConstant(value=0.0)


def CreateDummyMultiTree():
    mt = MultiTree()
    mt.trees = [CreateDummyTree()]
    return mt


def InstantiateTree(n):
    if isinstance(n, Constant) or isinstance(n, OOHRdyFeature) or isinstance(n, InstantiableConstant):
        n.get_value()
    for child_index in range(n.arity):
        _ = InstantiateTree(n.get_child(child_index))
    return n


def SimplifyConstants(tree: Node, **kwargs) -> Node:
    if CheckIfTreeHasOnlyOperatorsAndConstants(tree):
        result: float = float(tree(np.ones((1, 1)), **kwargs)[0])
        new_node: InstantiableConstant = InstantiableConstant(result, **kwargs)
        parent: Node = tree.parent
        child_id: int = tree.child_id
        if parent is not None:
            parent.replace_child(new_node, child_id)
        return new_node
    for i in range(tree.arity):
        _ = SimplifyConstants(tree.get_child(i), **kwargs)
    return tree


def CheckIfTreeHasOnlyOperatorsAndConstants(tree: Node) -> bool:
    arity: int = tree.arity
    if arity > 0:
        for i in range(arity):
            if not CheckIfTreeHasOnlyOperatorsAndConstants(tree.get_child(i)):
                return False
        return True
    if isinstance(tree, Constant) or isinstance(tree, InstantiableConstant):
        return True
    return False


def ForceNonConstantTree(tree: Node, X_train: np.ndarray, n_trials: int = 5):
    n_features: int = X_train.shape[1]
    possible_feature_nodes: List[OOHRdyFeature] = [OOHRdyFeature(i) for i in range(n_features)]
    tree = SimplifyConstants(tree)
    if isinstance(tree, Constant) or isinstance(tree, InstantiableConstant):
        tree = choice(possible_feature_nodes).create_new_empty_node()

    if n_features == 1:
        return tree

    batch_size: int = min(X_train.shape[0], 100)
    batch: np.ndarray = X_train[:batch_size]

    count: int = 0
    while len(np.unique(tree(batch))) == 1 and count < n_trials:
        chosen_node: Node = choice(MultiTree.extract_feature_nodes(tree))
        new_node = choice([node_to_sample for node_to_sample in possible_feature_nodes if node_to_sample.id != chosen_node.id]).create_new_empty_node()

        parent: Node = chosen_node.parent
        child_id: int = chosen_node.child_id
        if parent is not None:
            parent.replace_child(new_node, child_id)
        else:
            tree = new_node

        count += 1

    return tree


def GenerateRandomSimpleTree(
    internal_nodes: List[Node],
    leaf_nodes: List[Node],
    max_depth: int,
    curr_depth: int = 0,
    parent: Node = None,
):
    # heuristic to generate a semi-normal centered on relatively large trees
    prob_leaf = 0.01 + (curr_depth / max_depth) ** 3

    if curr_depth == max_depth or randu() < prob_leaf:
        n = choice(leaf_nodes).create_new_empty_node()
    else:
        # if at least one parent is not Plus, Minus, or Times, then
        # pick only among Plus, Minus, or Times
        if _check_ancestor_is_not_linear(parent):
            n = choice([Plus(), Times(), Minus()]).create_new_empty_node()
        else:
            n = choice(internal_nodes).create_new_empty_node()

    for _ in range(n.arity):
        c = GenerateRandomSimpleTree(
            internal_nodes, leaf_nodes, max_depth, curr_depth + 1, parent=n
        )
        n.insert_child(c)

    return n


def GenerateRandomTree(functions, terminals, max_height, curr_height=0, method='grow', min_depth=2):

    if curr_height == max_height:
        idx = randint(len(terminals))
        n = deepcopy( terminals[idx] )
    else:
        if method == 'grow' and curr_height >= min_depth:
            term_n_funs = terminals + functions
            idx = randint( len(term_n_funs) )
            n = deepcopy( term_n_funs[idx] )
        elif method == 'full' or (method == 'grow' and curr_height < min_depth):
            idx = randint( len(functions) )
            n = deepcopy( functions[idx] )
        else:
            raise ValueError('Unrecognized tree generation method')

        for i in range(n.arity):
            c = GenerateRandomTree( functions, terminals, max_height, curr_height=curr_height + 1, method=method, min_depth=min_depth )
            n.AppendChild( c )
    return n


def GenerateRandomNonlinearTree(
    internal_nodes: List[Node],
    leaf_nodes: List[Node],
    max_depth: int,
    curr_depth: int = 0,
):
    # heuristic to generate a semi-normal centered on relatively large trees
    prob_leaf = 0.01 + (curr_depth / max_depth) ** 3

    if curr_depth == max_depth or randu() < prob_leaf:
        n = choice(leaf_nodes).create_new_empty_node()
    else:
        if curr_depth == 0:
            # no linear transf
            n = choice(
                    [
                        x
                        for x in internal_nodes
                        if not (isinstance(x, Plus) or isinstance(x, Minus))
                    ]
                ).create_new_empty_node()

        else:
            n = choice(internal_nodes).create_new_empty_node()

    for _ in range(n.arity):
        c = GenerateRandomNonlinearTree(
            internal_nodes, leaf_nodes, max_depth, curr_depth + 1
        )
        n.insert_child(c)

    return n


def GenerateRandomMultitree(
    internal_nodes: List,
    leaf_nodes: List,
    max_depth: int,
    X_train: np.ndarray,
    partition_features: bool = False,
    min_trees_init: int = 1,
    max_trees_init: int = 5
) -> MultiTree:
    mt = MultiTree()

    features = list()
    constants = list()
    for l in leaf_nodes:
        if isinstance(l, Feature) or isinstance(l, OOHRdyFeature):
            features.append(l)
        else:
            constants.append(l)

    num_trees = choice(list(range(min_trees_init, max_trees_init + 1)))
    features_used = set()
    for _ in range(num_trees):
        curr_leaves = leaf_nodes
        if partition_features:
            if len(features_used) == len(features):
                continue
            curr_leaves = [f for f in features if f.id not in features_used] + constants

        #tree = GenerateRandomNonlinearTree(
        #    internal_nodes=internal_nodes, leaf_nodes=curr_leaves, max_depth=max_depth
        #)
        #tree = GenerateRandomTree(
        #    internal_nodes=internal_nodes, leaf_nodes=curr_leaves, max_depth=max_depth
        #)

        tree = GenerateRandomSimpleTree(
            internal_nodes=internal_nodes, leaf_nodes=curr_leaves, max_depth=max_depth,
            curr_depth=0, parent=None,
        )
        tree = ForceNonConstantTree(tree, X_train)

        features_used.update(MultiTree.extract_feature_ids(tree))
        mt.trees.append(tree)

    return mt


def MultitreeLevelCrossover(
    mt: MultiTree,
    donor_mt: MultiTree,
    idx_picked_tree: int | None = None,
) -> MultiTree:

    if idx_picked_tree is None:
        idx_picked_tree = np.random.randint(len(mt.trees))

    donor_tree = deepcopy(choice(donor_mt.trees))

    # drop the tree to be replaced
    mt.trees.pop(idx_picked_tree)
    mt.trees.insert(idx_picked_tree, donor_tree)
    return mt


def GenerateOffspringMultitree(
    parent_mt: MultiTree,
    crossovers: list,
    mutations: list,
    coeff_opts: list,
    donors: list,
    internal_nodes: list,
    leaf_nodes: list,
    max_depth: int,
    X_train: np.ndarray,
    constraints: dict = {"max_tree_size": 100},
    partition_features: bool = False,
    prob_delete_tree: float = 0.05,
    prob_init_tree: float = 0.1,
    prob_mt_crossover: float = 0.0,
    perform_only_one_op: bool = True
) -> MultiTree:

    if prob_mt_crossover > 0 and partition_features:
        raise ValueError(
            "Partition features and multitree crossover are not compatible"
        )

    # set the offspring to a copy (to be modified) of the parent
    offspring_mt = deepcopy(parent_mt)
    idx_picked_tree = choice(range(len(offspring_mt.trees)))

    # Case: delete tree
    if np.random.uniform() < prob_delete_tree and len(offspring_mt.trees) > 1:
        offspring_mt.trees.pop(idx_picked_tree)
        if perform_only_one_op:
            return offspring_mt
        # update idx picked tree
        idx_picked_tree = choice(range(len(offspring_mt.trees)))

    # compute what features can be used
    usable_leaf_nodes = MultiTree.extract_usable_leaves(
        idx_picked_tree, offspring_mt, leaf_nodes, partition_features
    )

    # Case: generate a new tree to add
    if np.random.uniform() < prob_init_tree and len(usable_leaf_nodes) > 0:
        # initialize a new tree
        new_tree = GenerateRandomNonlinearTree(
            internal_nodes=internal_nodes,
            leaf_nodes=usable_leaf_nodes,
            max_depth=max_depth,
        )
        new_tree = ForceNonConstantTree(new_tree, X_train)
        offspring_mt.trees.append(new_tree)
        if perform_only_one_op:
            return offspring_mt
        # update idx picked tree
        idx_picked_tree = choice(range(len(offspring_mt.trees)))

    # Case: multitree crossover
    if np.random.uniform() < prob_mt_crossover:
        # pick a donor
        donor_mt = choice(donors)
        offspring_mt = MultitreeLevelCrossover(
            offspring_mt,
            donor_mt,
            idx_picked_tree=idx_picked_tree,
        )
        if perform_only_one_op:
            return offspring_mt

        idx_picked_tree = choice(range(len(offspring_mt.trees)))

    # Next: undergo node-level variation operators

    # pick the tree to modify
    offspring_tree = deepcopy(offspring_mt.trees[idx_picked_tree])
    # create a backup for constraint violation
    backup_tree = deepcopy(offspring_tree)

    # apply variation operators in a random order
    structural_var_ops = crossovers + mutations
    random_order = np.arange(len(structural_var_ops))
    shuffle(random_order)
    changed = False
    for i in random_order:
        var_op = structural_var_ops[i]
        # randomize donors
        donor_trees = [choice(donor.trees) for donor in donors]
        offspring_tree, changed = __undergo_variation_operator_2(
            X_train,
            var_op,
            offspring_tree,
            crossovers,
            mutations,
            coeff_opts,
            choice(donor_trees),
            internal_nodes,
            usable_leaf_nodes
        )

        # check offspring_tree meets constraints, else revert to backup
        if not __check_tree_meets_all_constraints(offspring_tree, constraints):
            # revert to backup
            offspring_tree = deepcopy(backup_tree)
        else:
            # update backup
            backup_tree = deepcopy(offspring_tree)
            changed = False

        # perform only 1 operator
        if changed and perform_only_one_op:
            break

    # apply coeff mutations
    offspring_tree, changed = __undergo_variation_operator_2(
        X_train,
        choice(coeff_opts),
        offspring_tree,
        [],
        [],
        coeff_opts,
        None,
        internal_nodes,
        usable_leaf_nodes
    )

    # print("len of offspring tree", len(offspring_tree.get_subtree()))
    # update with offspring tree
    offspring_mt.trees.pop(idx_picked_tree)
    offspring_mt.trees.insert(idx_picked_tree, offspring_tree)

    # assert len is not violated
    # for off_tree in offspring_mt.trees:
    #    if len(off_tree.get_subtree()) > constraints["max_tree_size"]:
    #        print(f"Starting from {len(backup_tree.get_subtree())} nodes")
    #        print("mt crossover happened:", mt_crossover_happened)
    #        raise ValueError(f"Tree size constraint violated")

    return offspring_mt


def CoeffMutationInclOohFeatures(
    tree: Node, prob_coeff_mut: float = 0.25, temp: float = 0.25
) -> Node:
    """
    Applies random coefficient mutations to constant nodes

    Parameters
    ----------
    tree : Node
      the tree to which coefficient mutations are applied
    prob_coeff_mut : float, optional
      the probability with which coefficients are mutated (default is 0.25)
    temp : float, optional
      "temperature" that indicates the strength of coefficient mutation, it is relative to the current value (i.e., v' = v + temp*abs(v)*N(0,1))

    Returns
    -------
    Node
      the tree after coefficient mutation (it is the same as the tree in input)
    """
    coeffs = [
        n
        for n in tree.get_subtree()
        if isinstance(n, Constant) or isinstance(n, OOHRdyFeature) or isinstance(n, InstantiableConstant)
    ]
    for c in coeffs:
        # decide wheter it should be applied
        if randu() < prob_coeff_mut:
            v = c.get_value()
            # update the value by +- temp relative to current value
            new_v = v + temp * np.abs(v) * randn()
            c.set_value(new_v)

    return tree

def _check_ancestor_is_not_linear(ancestor: Node | None) -> bool:
    while ancestor is not None:
        if not (
            isinstance(ancestor, Plus)
            or isinstance(ancestor, Minus)
            or isinstance(ancestor, Times)
        ):
            return True
        ancestor = ancestor.parent
    return False

def __undergo_variation_operator_2(
    X_train: np.ndarray,
    var_op: dict,
    offspring: Node,
    crossovers,
    mutations,
    coeff_opts,
    donor,
    internal_nodes,
    leaf_nodes,
) -> Tuple[Node, bool]:
    if "kwargs" not in var_op:
        var_op["kwargs"] = {}

    # decide whether to actually do something
    if var_op["rate"] < randu():
        # nope
        return offspring, False

    # prepare the function to call
    var_op_fun = var_op["fun"]
    # next, we need to provide the right arguments based on the type of ops
    if var_op in crossovers:
        # we need a donor
        offspring = var_op_fun(offspring, deepcopy(donor), **var_op["kwargs"])
        offspring = ForceNonConstantTree(offspring, X_train)
    elif var_op in mutations:
        # we need to provide node types
        offspring = var_op_fun(offspring, deepcopy(internal_nodes), deepcopy(leaf_nodes), **var_op["kwargs"])
        offspring = ForceNonConstantTree(offspring, X_train)
    elif var_op in coeff_opts:
        offspring = var_op_fun(offspring, **var_op["kwargs"])
        offspring = ForceNonConstantTree(offspring, X_train)
    return offspring, True

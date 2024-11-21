from copy import deepcopy
import numpy as np
from typing import List, Tuple
from numpy.random import randint
from numpy.random import random
from sympy import simplify

from pynsgp.Nodes.MultiTree import MultiTree, extract_usable_leaves, extract_feature_ids
import numpy as np
from numpy.random import shuffle
from copy import deepcopy

from genepro.variation import *
from genepro.variation import __check_tree_meets_all_constraints
from genepro.node_impl import Plus, Minus, Times, Feature, Constant


from pynsgp.Nodes.more_node_impl import OOHRdyFeature, InstantiableConstant


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
        n = deepcopy(randc(leaf_nodes)[0])
    else:
        # if at least one parent is not Plus, Minus, or Times, then
        # pick only among Plus, Minus, or Times
        if _check_ancestor_is_not_linear(parent):
            n = deepcopy(randc([Plus(), Times(), Minus()])[0])
        else:
            n = deepcopy(randc(internal_nodes)[0])

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
        n = deepcopy(randc(leaf_nodes)[0])
    else:
        if curr_depth == 0:
            # no linear transf
            n = deepcopy(
                randc(
                    [
                        x
                        for x in internal_nodes
                        if not (isinstance(x, Plus) or isinstance(x, Minus))
                    ]
                )[0]
            )
        else:
            n = deepcopy(randc(internal_nodes)[0])

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
    partition_features: bool = False,
    min_trees_init: int = 1,
    max_trees_init: int = 5,
) -> MultiTree:
    mt = MultiTree()

    features = list()
    constants = list()
    for l in leaf_nodes:
        if isinstance(l, Feature) or isinstance(l, OOHRdyFeature):
            features.append(l)
        else:
            constants.append(l)

    num_trees = np.random.choice(list(range(min_trees_init, max_trees_init + 1)))
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

        generated_tree_is_constant = True
        generations_trials = 0
        while generated_tree_is_constant and generations_trials < 5:
            tree = GenerateRandomSimpleTree(
                internal_nodes=internal_nodes, leaf_nodes=curr_leaves, max_depth=max_depth,
                curr_depth=0, parent=None,
            )
            feature_ids_extracted_from_tree = extract_feature_ids(tree)
            if len(feature_ids_extracted_from_tree) > 0:
                generated_tree_is_constant = False
            else:
                generations_trials += 1

        features_used.update(feature_ids_extracted_from_tree)
        mt.trees.append(tree)

    return mt


def OnePointMutation( individual, functions, terminals ):

    arity_functions = {}
    for f in functions:
        arity = f.arity
        if arity not in arity_functions:
            arity_functions[arity] = [f]
        else:
            arity_functions[arity].append(f)

    nodes = individual.GetSubtree()
    prob = 1.0/len(nodes)

    for i in range(len(nodes)):
        if random() < prob:
            arity = nodes[i].arity
            if arity == 0:
                idx = randint( len(terminals) )
                n = deepcopy( terminals[idx] )
            else:
                idx = randint(len(arity_functions[arity]))
                n = deepcopy(arity_functions[arity][idx])

            # update link to children
            for child in nodes[i]._children:
                n.AppendChild(child)

            # update link to parent node
            p = nodes[i].parent
            if p:
                idx = p.DetachChild( nodes[i] )
                p.InsertChildAtPosition(idx, n)
            else:
                nodes[i] = n
                individual = n


    return individual


def SubtreeMutation( individual, functions, terminals, max_height=4 ):

    mutation_branch = GenerateRandomTree( functions, terminals, max_height )

    nodes = individual.GetSubtree()

    #nodes = __GetCandidateNodesAtUniformRandomDepth( nodes )

    to_replace = nodes[randint(len(nodes))]

    if not to_replace.parent:
        del individual
        return mutation_branch


    p = to_replace.parent
    idx = p.DetachChild(to_replace)
    p.InsertChildAtPosition(idx, mutation_branch)

    return individual


def SubtreeCrossover( individual, donor ):

    # this version of crossover returns 1 child

    nodes1 = individual.GetSubtree()
    nodes2 = donor.GetSubtree()	# no need to deep copy all nodes of parent2

    #nodes1 = __GetCandidateNodesAtUniformRandomDepth( nodes1 )
    #nodes2 = __GetCandidateNodesAtUniformRandomDepth( nodes2 )

    to_swap1 = nodes1[ randint(len(nodes1)) ]
    to_swap2 = deepcopy( nodes2[ randint(len(nodes2)) ] )	# we deep copy now, only the sutbree from parent2
    to_swap2.parent = None

    p1 = to_swap1.parent

    if not p1:
        return to_swap2

    idx = p1.DetachChild(to_swap1)
    p1.InsertChildAtPosition(idx, to_swap2)

    return individual


def __GetCandidateNodesAtUniformRandomDepth( nodes ):

    depths = np.unique( [x.GetDepth() for x in nodes] )
    chosen_depth = depths[randint(len(depths))]
    candidates = [x for x in nodes if x.GetDepth() == chosen_depth]

    return candidates


def MultitreeLevelCrossover(
    mt: MultiTree,
    donor_mt: MultiTree,
    idx_picked_tree: int | None = None,
) -> MultiTree:

    if idx_picked_tree is None:
        idx_picked_tree = np.random.randint(len(mt.trees))

    donor_tree = deepcopy(np.random.choice(donor_mt.trees))

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
    idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # Case: delete tree
    if np.random.uniform() < prob_delete_tree and len(offspring_mt.trees) > 1:
        offspring_mt.trees.pop(idx_picked_tree)
        if perform_only_one_op:
            return offspring_mt
        # update idx picked tree
        idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # compute what features can be used
    usable_leaf_nodes = extract_usable_leaves(
        idx_picked_tree, offspring_mt, leaf_nodes, partition_features
    )

    # Case: generate a new tree to add
    if np.random.uniform() < prob_init_tree and len(usable_leaf_nodes) > 0:
        # initialize a new tree
        generated_tree_is_constant = True
        generations_trials = 0
        while generated_tree_is_constant and generations_trials < 5:
            new_tree = GenerateRandomNonlinearTree(
                internal_nodes=internal_nodes,
                leaf_nodes=usable_leaf_nodes,
                max_depth=max_depth,
            )
            feature_ids_extracted_from_tree = extract_feature_ids(new_tree)
            if len(feature_ids_extracted_from_tree) > 0:
                generated_tree_is_constant = False
            else:
                generations_trials += 1

        offspring_mt.trees.append(new_tree)
        if perform_only_one_op:
            return offspring_mt
        # update idx picked tree
        idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

    # Case: multitree crossover
    if np.random.uniform() < prob_mt_crossover:
        # pick a donor
        donor_mt = np.random.choice(donors)
        offspring_mt = MultitreeLevelCrossover(
            offspring_mt,
            donor_mt,
            idx_picked_tree=idx_picked_tree,
        )
        if perform_only_one_op:
            return offspring_mt

        idx_picked_tree = np.random.choice(range(len(offspring_mt.trees)))

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
        donor_trees = [np.random.choice(donor.trees) for donor in donors]
        offspring_tree, changed = __undergo_variation_operator_2(
            var_op,
            offspring_tree,
            crossovers,
            mutations,
            coeff_opts,
            np.random.choice(donor_trees),
            internal_nodes,
            usable_leaf_nodes,
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
        np.random.choice(coeff_opts),
        offspring_tree,
        [],
        [],
        coeff_opts,
        None,
        internal_nodes,
        usable_leaf_nodes,
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
        offspring = var_op_fun(offspring, donor, **var_op["kwargs"])
    elif var_op in mutations:
        # we need to provide node types
        generated_tree_is_constant = True
        generations_trials = 0
        while generated_tree_is_constant and generations_trials < 5:
            offspring = var_op_fun(
                offspring, internal_nodes, leaf_nodes, **var_op["kwargs"]
            )
            feature_ids_extracted_from_tree = extract_feature_ids(offspring)
            if len(feature_ids_extracted_from_tree) > 0:
                generated_tree_is_constant = False
            else:
                generations_trials += 1
    elif var_op in coeff_opts:
        generated_tree_is_constant = True
        generations_trials = 0
        while generated_tree_is_constant and generations_trials < 5:
            offspring = var_op_fun(offspring, **var_op["kwargs"])
            feature_ids_extracted_from_tree = extract_feature_ids(offspring)
            if len(feature_ids_extracted_from_tree) > 0:
                generated_tree_is_constant = False
            else:
                generations_trials += 1

    return offspring, True

"""
This file contains utils functions to encode a Sci-kit learn RandomForestClassifier in SAT as explained in "On
Explaining Random Forests with SAT" by Yacine Izza and Joao Marques-Silva
"""

import numpy as np
from pysat.card import CardEnc
from pysat.pb import PBEnc
from sklearn.tree import _tree
from pysat.formula import CNF, CNFPlus


def get_features(model, feature_names):
    """
    Finds all the thresholds for each feature of the entire random forest

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        The random forest model
    feature_names : str list
        The names of the features of the model

    Returns
    -------
    split : dict
        key: name of the feature
        value: list of thresholds related to the feature
    """

    features = feature_names
    split = dict(zip(features, [[] for _ in features]))

    for temp_model in model.estimators_:
        temp_tree = temp_model.tree_
        temp_split = temp_tree.feature
        for i, node in enumerate(temp_split):
            if node == -2:
                continue
            if temp_tree.threshold[i] not in split[features[node]]:
                split[features[node]].append(temp_tree.threshold[i])

    return split


def create_variables(feature):
    """
    Starts from a dictionary of features, containing all the thresholds for each feature, and creates the variables
    needed for the thresholds and the intervals between the thresholds

    Parameters
    ----------
    feature : dict
        The dictionary of features with the lists of thresholds created by "get_features" function

    Returns
    -------
    variables : dict
        key: name of the feature
        value: a dictionary containing variables related to thresholds and intervals
            key: variable related to the threshold
            value: list of two elements, the first element is the list of variables related to the intervals lower than
                or equal to the threshold, the second element is the list of variables related to the intervals greater
                than the threshold
    tresh_to_var : dict
        correspondence between the threshold and the variable related to it
        key: name of the feature
        value: list of variables related to the thresholds of the feature. The values of the list have the same indexes
            of the values of the lists contained in the "feature" parameter
    counter : int
        last defined variable
    """

    counter = 0
    variables = dict()
    tresh_to_var = dict()

    for key, values in feature.items():
        values.sort()
        if len(values) == 0:
            continue
        if len(values) == 1:
            counter += 1
            variables[key] = create_binary(tresh_to_var, key, counter)

        else:
            counter += 1
            variables[key] = create_real_value(values, tresh_to_var, key, counter)
            counter += len(values) + len(values)

    return variables, tresh_to_var, counter


def create_binary(tresh_to_var, key, counter):
    """
    Function used inside "create_variables", creates the variables for a single-threshold feature.
    For single-threshold features no interval variables are created, the threshold var is used to represent the left
    interval and the negated threshold var is used to represent the right interval.

    Parameters
    ----------
    tresh_to_var : dict
        see "create_variables" Returns section
    key : str
        name of the single-threshold feature
    counter : int
        see "create_variables" Returns section

    Returns
    -------
    variable : dict
        key: variable related to the threshold
        value: list of two elements, the first element is a list containing the threshold variable, the second
            element is a list containing the negated threshold variable
    """

    variable = {}
    variable[counter] = [[counter], [-counter]]
    tresh_to_var[key] = [counter]

    return variable


def create_real_value(values, tresh_to_var, key, counter):
    """
    Function used inside "create_variables", creates the variables for a multi-threshold feature.
    For multi-threshold features N+1 interval variables are created, where N is the number of thresholds.

    Parameters
    ----------
    values: float list
        values of the thresholds
    tresh_to_var : dict
        see "create_binary"
    key : str
        see "create_binary"
    counter : int
        see "create_binary"

    Returns
    -------
    variable : dict
        key: variable related to the threshold
        value: list of two elements, the first element is the list of variables related to the intervals lower than
            or equal to the threshold, the second element is the list of variables related to the intervals greater
            than the threshold
    """

    n_thresholds = len(values)
    var_tresholds = list(range(counter, counter + n_thresholds))
    var_intervals = list(range(counter + n_thresholds, counter + n_thresholds + n_thresholds + 1))
    variable = {}
    tresh_to_var[key] = var_tresholds

    for i, var in enumerate(var_tresholds):
        temp_left = []
        temp_right = []
        for interval in var_intervals[:i + 1]:
            temp_left.append(interval)
        for interval in var_intervals[i + 1:]:
            temp_right.append(interval)
        variable[var] = [temp_left, temp_right]

    return variable


def create_class_vars(model, last_var):
    """
    Creates the variables related to the classes predicted by each decision tree of the random forest.
    For each decision tree, for each class a variable is defined.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        The random forest model
    last_var : int
        integer value related to the last defined variable

    Returns
    -------
    vars : dict
        key: tree index
        value: list of variables related to the classes predicted by the single tree
    """
    n_trees = len(model.estimators_)
    n_classes = len(model.classes_)

    vars = dict()
    for i in range(n_trees):
        vars[i] = []
        for j in range(n_classes):
            vars[i].append(last_var + 1)
            last_var += 1
    return vars


def get_paths(trees, feature_names):
    """
    Retrieve all the paths of all the trees of the random forest in a human-readable way

    Parameters
    ----------
    trees : sklearn.ensemble.DecisionTreeClassifier list
        List of all the decision trees of the random forest
    feature_names : str list
        The names of the features of the model

    Returns
    -------
    paths_total : dict
        key: tree index
        value: list of strings representing each path of the decision tree
    """

    paths_total = {}
    for i, tree in enumerate(trees.estimators_):

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                vars = name.upper()
                p1 += [f"{vars} <= {threshold}"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"{vars} > {threshold}"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                class_assigned = np.argmax(tree_.value[node])
                path += [class_assigned]
                paths += [path]

        recurse(0, path, paths)

        paths_total[i] = paths

    return paths_total


def encode_paths(trees, feature_names, feature_split, corr, class_vars):
    """
    Encode all the paths of all the trees of the random forest as pysat.formula.CNF objects following the procedure
    explained in "On Explaining Random Forests with SAT" by Yacine Izza and Joao Marques-Silva

    Parameters
    ----------
    trees : sklearn.ensemble.DecisionTreeClassifier list
        List of all the decision trees of the random forest
    feature_names : str list
        The names of the features of the model
    feature_split : dict
        value returned by "get_features" function
    corr : dict
        value returned by "create_variables" function in "tresh_to_var" variable
    class_vars : dict
        value returned by "create_class_vars" function

    Returns
    -------
    forest_cnf_paths : dict
        key: tree index
        value: list of pysat.formula.CNF objects representing each path of the decision tree
    """

    forest_cnf_paths = {}
    for tree_index, tree in enumerate(trees.estimators_):

        tree_ = tree.tree_
        feature_name = [
            feature_names[tree_index] if tree_index != _tree.TREE_UNDEFINED else "undefined!"
            for tree_index in tree_.feature
        ]

        single_tree_cnf_paths = []
        thresh_vars_list = []

        # explore the tree
        def recurse(node, thresh_vars_list, single_tree_cnf_paths):

            # the current node is a non-terminal node:
            # save the threshold variable
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]  # feature name (used as index)
                threshold = tree_.threshold[node]  # threshold value
                left_p, right_p = list(thresh_vars_list), list(thresh_vars_list)

                var_index = feature_split[name].index(threshold)  # threshold variable index
                var = corr[name][var_index]  # threshold variable

                left_p.append(var)
                recurse(tree_.children_left[node], left_p, single_tree_cnf_paths)
                right_p.append(-var)
                recurse(tree_.children_right[node], right_p, single_tree_cnf_paths)

            # the current node is a terminal node:
            # use the list of threshold variables to encode the path
            else:
                # _ : not
                # _l1 and l2 and l3 -> l12
                # l1 or _l2 or _l3 or l12
                clause = []
                class_assigned = np.argmax(tree_.value[node])
                class_var = class_vars[tree_index][class_assigned]

                clause.append(class_var)
                for j in range(len(thresh_vars_list)):
                    thresh_var = thresh_vars_list[j]
                    clause.append(-thresh_var)

                single_cnf_path = CNF(from_clauses=[clause])

                single_tree_cnf_paths.append(single_cnf_path)

        recurse(0, thresh_vars_list, single_tree_cnf_paths)
        forest_cnf_paths[tree_index] = single_tree_cnf_paths

    return forest_cnf_paths


def encode_thresholds_and_intervals(features_thresholds_and_intervals):
    """
    Encode the corresponce between the thresholds variables and the intervals variables following the procedure
    explained in "On Explaining Random Forests with SAT" by Yacine Izza and Joao Marques-Silva

    Parameters
    ----------
    features_thresholds_and_intervals : dict
        value returned by "create_variables" function in "variables" variable

    Returns
    -------
    forest_cnf_paths : pysat.formula.CNF list
        list of pysat.formula.CNF objects representing the correspondence between thresholds vars and interval vars
    """
    cnf_threshold_and_intervals = []
    for single_feature_vars in features_thresholds_and_intervals.values():
        for thresh_var, intervals in single_feature_vars.items():
            # left encoding: positive thresh_var
            left_intervals = intervals[0]
            left_encoding = encode_double_implication(thresh_var, left_intervals)

            cnf_threshold_and_intervals.append(left_encoding)

            # right encoding: negative thresh_var
            right_intervals = intervals[1]
            right_encoding = encode_double_implication(-thresh_var, right_intervals)

            cnf_threshold_and_intervals.append(right_encoding)

    return cnf_threshold_and_intervals


def encode_double_implication(thresh_var, list_of_intervals):
    """
    Encode in CNF a double implication of the following type:
        _ : not
        a <-> b1 or b2 or b3
        (a -> b1 or b2 or b3) and ((b1 or b2 or b3) -> a)
        (_a or b1 or b2 or b3) and ((_b1 and _b2 and _b3) or a)
        (_a or b1 or b2 or b3) and (_b1 or a) and (_b2 or a) and (_b3 or a)

    Parameters
    ----------
    thresh_var : int
        variable on the left-hand side of the double implication
    list_of_intervals : int list
        list of variables in OR on the right-hand side of the double implication

    Returns
    -------
    cnf_formula : pysat.formula.CNF
        pysat.formula.CNF object representing the CNF formula encoding the double implication
    """

    cnf_formula = CNF()
    first_clause = [-thresh_var]
    for interval_var in list_of_intervals:
        curr_clause = [thresh_var, -interval_var]
        cnf_formula.append(curr_clause)

        first_clause.append(interval_var)

    cnf_formula.append(first_clause)
    return cnf_formula


def encode_card_net_constr(class_vars):
    """
    Encode the cardinality network constraints explained in "On Explaining Random Forests with SAT" by Yacine Izza
    and Joao Marques-Silva

    Parameters
    ----------
    class_vars : dict
        value returned by "create_class_vars" function

    Returns
    -------
    forest_card_net_constr : pysat.formula.CNFPlus list
        list of pysat.formula.CNFPlus objects encoding the cardinality network constraint of each decision tree of the
        random forest
    """

    forest_card_net_constr = []
    for tree_classes in class_vars.values():
        tree_card_net_constr = CardEnc.equals(lits=tree_classes, bound=1)
        forest_card_net_constr.append(tree_card_net_constr)
    return forest_card_net_constr


def encode_majority_voting(predicted_class, class_vars):
    """
    Encode the majority voting constraints explained in "On Explaining Random Forests with SAT" by Yacine Izza and
    Joao Marques-Silva

    Parameters
    ----------
    predicted_class : int
        the class predicted by the random forest
    class_vars : dict
        value returned by "create_class_vars" function

    Returns
    -------
    constr_list : pysat.formula.CNF list
        list of pysat.formula.CNF objects encoding the majority voting constraint of each not predicted class of the
        random forest
    """

    # using the same indexes as the reference paper
    j = predicted_class
    n_classes = len(class_vars[0])
    constr_list = []
    for k in range(n_classes):
        if j != k:
            constr_list.append(encode_mv(k, j, class_vars))
    return constr_list


def encode_mv(k, j, class_vars):
    """
    Function called by "encode_majority_voting" to encode one between (5) and (6) formulas presented in the reference
    paper for each not predicted class

    Parameters
    ----------
    k : int
        the index of the predicted class
    j : int
        the index of the current not predicted class
    class_vars : dict
        value returned by "create_class_vars" function

    Returns
    -------
    mv_formula : pysat.formula.CNF
        encoding of the majority voting constraint related to a single not predicted class
    """

    # M number of trees, i goes from 1 to M, j is the current not predicted class, k is the predicted class
    # lhs -> left-hand side
    # rhs -> right-hand side

    M = len(class_vars)
    ik = []
    lhs_ij = []
    rhs_ij = []
    lhs_ik_weights = []
    lhs_ij_weights = []
    rhs_weights = []
    for i in range(M):
        ik.append(class_vars[i][k])
        lhs_ij.append(class_vars[i][j])
        rhs_ij.append(-class_vars[i][j])
        lhs_ik_weights.append(1)
        lhs_ij_weights.append(-1)
        rhs_weights.append(1)

    lhs_lits = ik + lhs_ij
    rhs_lits = ik + rhs_ij
    lhs_weights = lhs_ik_weights + lhs_ij_weights
    rhs_weights = rhs_weights + rhs_weights

    # left-hand side of the double implication
    lhs = PBEnc.atleast(lits=lhs_lits, weights=lhs_weights, bound=1)
    last_var = lhs.nv

    # right-hand side of the double implication
    if k < j:
        rhs = PBEnc.atleast(lits=rhs_lits, weights=rhs_weights, bound=M, top_id=last_var)
    elif j < k:
        rhs = PBEnc.atleast(lits=rhs_lits, weights=rhs_weights, bound=(M + 1), top_id=last_var)

    if rhs.nv > last_var:
        last_var = rhs.nv

    # apply double implication and get majority voting formula
    mv_formula = double_implication(lhs, rhs, last_var)

    return mv_formula


def double_implication(lhs, rhs, last_var):
    """
    Encodes the CNF double implication between two CNF

    Parameters
    ----------
    lhs : pysat.formula.CNF
        left-hand side formula of the double implication
    rhs : pysat.formula.CNF
        right-hand side formula of the double implication
    last_var : int
        top variable identifier used so far (to be passed to the negate() method)

    Returns
    -------
    final_cnf : pysat.formula.CNF
        CNF encoding of the double implication
    """

    # F1 <-> F2
    # (_F1 or F2) and (_F2 or F1)
    final_cnf = CNF()
    first_implication = disjunction(lhs.negate(topv=last_var), rhs)
    second_implication = disjunction(rhs.negate(topv=last_var), lhs)
    final_cnf.extend(first_implication.clauses + second_implication.clauses)
    return final_cnf


def disjunction(f1, f2):
    """
    Encodes the CNF disjunction between two CNF

    Parameters
    ----------
    f1 : pysat.formula.CNF
        first formula
    f2 : pysat.formula.CNF
        second formula

    Returns
    -------
    final_cnf : pysat.formula.CNF
        CNF encoding of disjunction
    """

    final_cnf = CNF()
    for f1_clause in f1:
        for f2_clause in f2:
            new_clause = list(set(f1_clause + f2_clause))
            final_cnf.append(new_clause)
    return final_cnf


def final_encoding(cnf_paths, cnf_thresholds_and_intervals, cnf_card_net_constr, cnf_majority_voting_constr):
    """
    Encode the entire random forest putting all together as explained in "On Explaining Random Forests with SAT" by
    Yacine Izza and Joao Marques-Silva

    Parameters
    ----------
    cnf_paths : dict
        value returned by "encode_path" function
    cnf_thresholds_and_intervals : dict
        value returned by "encode_thresholds_and_intervals" function
    cnf_card_net_constr : pysat.formula.CNFPlus list
        value returned by "encode_card_net_constr" function
    cnf_majority_voting_constr : pysat.formula.CNF

    Returns
    -------
    total_formula : pysat.formula.CNFPlus
        pysat.formula.CNFPlus object encoding the entire random forest
    """

    total_formula = CNFPlus()

    for tree_paths in cnf_paths.values():
        for path in tree_paths:
            total_formula.extend(path.clauses)

    for formula in cnf_thresholds_and_intervals:
        total_formula.extend(formula.clauses)

    for card_net_constr in cnf_card_net_constr:
        total_formula.extend(card_net_constr)

    for majority_voting_constr in cnf_majority_voting_constr:
        total_formula.extend(majority_voting_constr)

    return total_formula

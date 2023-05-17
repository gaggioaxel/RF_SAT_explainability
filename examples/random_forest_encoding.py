"""
Example of the encoding of random forests using "encode_rf_utils.py" module
"""

import joblib
from pysat.solvers import Solver

import encode_rf_utils as utils
import mock_model as mm


def create_mock_model():
    classes = [0, 1, 2]

    features1 = [0, 1, 2, -2, -2, -2, -2]
    features2 = [1, -2, 2, -2, -2]

    threshold1 = [2.1, 1.2, 3.4, -2, -2, -2, -2]
    threshold2 = [1.9, -2, 6.5, -2, -2]

    values1 = [[], [], [], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    values2 = [[], [1, 0, 0], [], [0, 1, 0], [0, 0, 1]]

    children_left1 = [1, 3, 5, -1, -1, -1, -1]
    children_right1 = [2, 4, 6, -1, -1, -1, -1]

    children_left2 = [1, -1, 3, -1, -1]
    children_right2 = [2, -1, 4, -1, -1]

    mock_tree1 = mm.MockTree(features1, threshold1, values1, children_left1, children_right1)
    mock_tree2 = mm.MockTree(features2, threshold2, values2, children_left2, children_right2)

    forest_estimators = [mock_tree1, mock_tree2]

    mock_forest = mm.MockForest(forest_estimators, classes)

    return mock_forest


def create_mock_model_paper():
    classes = [0, 1]  # 0 is NO and 1 is YES

    features1 = [0, 2, -2, -2, -2]
    features2 = [1, -2, 3, -2, -2]
    features3 = [1, 0, 2, -2, -2, -2, -2]

    threshold1 = [0, 0, -2, -2, -2]
    threshold2 = [0, -2, 75, -2, -2]
    threshold3 = [0, 0, 0, -2, -2, -2, -2]

    values1 = [[], [], [1, 0], [0, 1], [1, 0]]
    values2 = [[], [1, 0], [], [0, 1], [1, 0]]
    values3 = [[], [], [], [0, 1], [1, 0], [0, 1], [1, 0]]

    children_left1 = [1, 3, -1, -1, -1]
    children_right1 = [2, 4, -1, -1, -1]

    children_left2 = [1, -1, 3, -1, -1]
    children_right2 = [2, -1, 4, -1, -1]

    children_left3 = [1, 3, 5, -1, -1, -1, -1]
    children_right3 = [2, 4, 6, -1, -1, -1, -1]

    mock_tree1 = mm.MockTree(features1, threshold1, values1, children_left1, children_right1)
    mock_tree2 = mm.MockTree(features2, threshold2, values2, children_left2, children_right2)
    mock_tree3 = mm.MockTree(features3, threshold3, values3, children_left3, children_right3)

    forest_estimators = [mock_tree1, mock_tree2, mock_tree3]

    mock_forest = mm.MockForest(forest_estimators, classes)

    return mock_forest


def encode_iris():
    with open('../model/iris_model', 'rb') as r:
        model = joblib.load(r)

    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    mock_predicted_class = 1

    features_split = utils.get_features(model, feature_names)
    variables, corr, last_var = utils.create_variables(features_split)
    class_vars = utils.create_class_vars(model, last_var)
    cnf_paths = utils.encode_paths(model, feature_names, features_split, corr, class_vars)
    cnf_thresholds_and_intervals = utils.encode_thresholds_and_intervals(variables)
    cnf_card_net_constr = utils.encode_card_net_constr(class_vars)
    cnf_majority_voting_constr = utils.encode_majority_voting(mock_predicted_class, class_vars)
    cnf_total_formula = utils.final_encoding(cnf_paths, cnf_thresholds_and_intervals, cnf_card_net_constr,
                                             cnf_majority_voting_constr)
    print('ok')


def encode_mock_model():
    model = create_mock_model()

    feature_names = ['feature0', 'feature1', 'feature2']

    mock_predicted_class = 0

    features_split = utils.get_features(model, feature_names)
    variables, corr, last_var = utils.create_variables(features_split)
    class_vars = utils.create_class_vars(model, last_var)
    cnf_paths = utils.encode_paths(model, feature_names, features_split, corr, class_vars)
    cnf_thresholds_and_intervals = utils.encode_thresholds_and_intervals(variables)
    cnf_card_net_constr = utils.encode_card_net_constr(class_vars)
    cnf_majority_voting_constr = utils.encode_majority_voting(mock_predicted_class, class_vars)
    cnf_total_formula = utils.final_encoding(cnf_paths, cnf_thresholds_and_intervals, cnf_card_net_constr,
                                             cnf_majority_voting_constr)

    # check if the final formula is satisfiable
    s = Solver()
    for clause in cnf_total_formula.clauses:
        s.add_clause(clause)
    print(s.solve())


def encode_mock_model_paper():
    model = create_mock_model_paper()

    feature_names = ['blocked arteries', 'good-blood circulation', 'chest-pain', 'weight']

    mock_predicted_class = 1

    features_split = utils.get_features(model, feature_names)
    variables, corr, last_var = utils.create_variables(features_split)
    class_vars = utils.create_class_vars(model, last_var)
    cnf_paths = utils.encode_paths(model, feature_names, features_split, corr, class_vars)
    cnf_thresholds_and_intervals = utils.encode_thresholds_and_intervals(variables)
    cnf_card_net_constr = utils.encode_card_net_constr(class_vars)
    cnf_majority_voting_constr = utils.encode_majority_voting(mock_predicted_class, class_vars)
    cnf_total_formula = utils.final_encoding(cnf_paths, cnf_thresholds_and_intervals, cnf_card_net_constr,
                                             cnf_majority_voting_constr)

    # check if the final formula is satisfiable
    s = Solver()
    for clause in cnf_total_formula.clauses:
        s.add_clause(clause)
    print(s.solve())


# encode_iris()
encode_mock_model()
encode_mock_model_paper()

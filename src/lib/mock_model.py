class MockForest:
    """
    A class used to represent a mock random forest with the same structure of Sci-kit learn RandomForestClassifier to
    create tests (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

    Attributes
    ----------
    estimators: MockTree list
        mock decision trees of the random forest
    classes: int list
        list of the classes that can be predicted by the random forest
    """

    def __init__(self, estimators, classes):
        self.estimators_ = estimators
        self.classes_ = classes


class MockTree:
    """
    A class used to represent a mock decision tree with the same structure of Sci-kit learn DecisionTreeClassifier to
    create tests (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn
    .tree.DecisionTreeClassifier)

    Attributes
    ----------
    tree_: MockTreeStructure
        mock internal structure of the model's tree
    """

    def __init__(self, features, thresholds, values, children_left, children_right):
        self.tree_ = MockTreeStructure(features, thresholds, values, children_left, children_right)


class MockTreeStructure:
    """
    A class used to represent the same internal structure of a Sci-kit learn DecisionTreeClassifier (reference:
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree
    -plot-unveil-tree-structure-py)

    Attributes: to understand the format of the attributes see the reference linked above
    ----------
    features: int list
    thresholds: float list
    values: list of int lists
    children_left: int list
    children_right: int list
    """

    def __init__(self, features, thresholds, values, children_left, children_right):
        self.feature = features
        self.threshold = thresholds
        self.value = values
        self.children_left = children_left
        self.children_right = children_right

from IPython.display import Image
import pydotplus
from sklearn import tree


def responsible_features(estimators, X):
    """Return an array of integers indicating which features (columns) in X contributed to the classificaiton decsion

    >>> responsible_features(RandomForestClassifier().fit(X, y), X)
    """
    estimators = getattr(estimators, 'estimators_', estimators)
    estimators = [estimators] if not hasattr(estimators, '__len__') else estimators
    y = []
    for est in estimators:
        y += [est.predict(X)]
    return y


def plot_decision_tree(dt):
    dot_data = tree.export_graphviz(dt, out_file='tree.dot')
    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())

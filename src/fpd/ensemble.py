from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from fpd.dataset import Dataset

# FROM: https://www.datacamp.com/tutorial/ensemble-learning-python
def bagged_decision_tree_classifier():
    fp = Dataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
    )
    X = fp.as_numpy_array()
    Y = fp.target()
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(
        base_estimator=cart, n_estimators=num_trees, random_state=7
    )
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("Bagged Decision Tree Ensemble Classifier accuracy:", results.mean())


def adaboost():
    fp = Dataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
    )
    X = fp.as_numpy_array()
    Y = fp.target()
    seed = 7
    num_trees = 70
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("ADAboost Ensemble Classifier accuracy:", results.mean())


def voting_ensemble():
    fp = Dataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
    )
    X = fp.as_numpy_array()
    Y = fp.target()
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    # create the sub models
    estimators = []
    model1 = LogisticRegression(max_iter=250)
    estimators.append(("logistic", model1))
    model2 = DecisionTreeClassifier()
    estimators.append(("cart", model2))
    model3 = SVC()
    estimators.append(("svm", model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
    print("Voting Ensemble Classifier accuracy:", results.mean())

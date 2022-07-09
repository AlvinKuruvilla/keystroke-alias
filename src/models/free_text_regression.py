# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from .regression_utils import compare_regression
from features.feature_lists import get_desktop_features

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=ConvergenceWarning)

# function to call the compare_regression function for the specified model, feature_type and task
def regression_results(problem, feature_type, model):
    num_features = []
    mae = []
    hyper = []
    val_score = []
    for i in range(5, 105, 5):
        res, setup, val = compare_regression(problem, feature_type, i, model)
        num_features.append(i)
        mae.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(mae)
    print(hyper)
    # print(val_score)


class_problems = ["Height"]
models = ["XGBoost"]

for model in models:
    print(
        "###########################################################################################"
    )
    print(model)
    for class_problem in class_problems:
        print(class_problem)
        print("Desktop")
        regression_results(class_problem, get_desktop_features(), model)
        print()
        print(
            "-----------------------------------------------------------------------------------------"
        )

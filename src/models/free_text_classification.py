from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from features.feature_lists import get_desktop_features

from .classification_utils import compare_classification

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=ConvergenceWarning)

# function to call the compare_classification function for the specified model, feature_type and task
def classification_results(problem, feature_type, model):
    num_features = []
    accuracy = []
    hyper = []
    val_score = []
    for i in range(5, 105, 5):
        res, setup, val = compare_classification(problem, feature_type, i, model)
        num_features.append(i)
        accuracy.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(accuracy)
    print(hyper)
    # print(val_score)


def run_free_text_xgboost_gender_ml_model():
    class_problems = ["Gender"]
    models = ["XGBoost"]

    for model in models:
        print(
            "###########################################################################################"
        )
        print(model)
        for class_problem in class_problems:
            print(class_problem)
            print("Desktop")
            classification_results(class_problem, get_desktop_features(), model)
            print()
            print(
                "-----------------------------------------------------------------------------------------"
            )

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from modeling.modeling import Modeling

CLASSIFICATION_METHODS = [
    "logistic_regression_lbfgs",
    "logistic_regression_saga",
    "svm",
    "random_forest",
]
CLASSIFICATION_PARAMS = {
    "logistic_regression_lbfgs": {
        "solver": ["lbfgs"],
        "penalty": ["l2"],
        "max_iter": [1000],
    },
    "logistic_regression_saga": {
        "solver": ["saga"],
        "penalty": ["l1", "l2", "elasticnet"],
        "l1_ratio": [0.5],
        "max_iter": [1000],
    },
    "random_forest": {
        "max_depth": [10, 100, None],
        # "max_features": ["auto", "sqrt"],
        # "min_samples_leaf": [1, 2, 4],
        # "min_samples_split": [2, 5, 10],
        "n_estimators": [10, 100, 1000],
    },
    "svm": {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    },
}


class Classification(Modeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def classify(self):
        return self.create_model()

    def select_classifier(self, classifier_name):
        if classifier_name.startswith("logistic_regression"):
            options = {
                "classifier": [LogisticRegression(random_state=self.random_seed)]
            }
        elif classifier_name == "random_forest":
            options = {
                "classifier": [RandomForestClassifier(random_state=self.random_seed)]
            }
        elif classifier_name == "svm":
            options = {
                "classifier": [SVC(random_state=self.random_seed, probability=True)]
            }

        for key, value in CLASSIFICATION_PARAMS[classifier_name].items():
            options[f"classifier__{key}"] = value

        return options

    def encode_labels(self, labels):
        encoder = LabelEncoder()

        fitted_encoder = encoder.fit(labels)
        labels_encoded = encoder.transform(labels)

        return labels_encoded, fitted_encoder

    def get_cv(self, n_splits=10, n_repeats=1):
        return RepeatedStratifiedKFold(
            random_state=self.random_seed, n_splits=n_splits, n_repeats=n_repeats
        )

    def get_scoring(self):
        return {
            "auc": "roc_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "specificity": make_scorer(recall_score, pos_label=0),
        }

    def get_pipeline(self):
        return Pipeline([("preprocessor", None), ("classifier", None)])

    def get_parameter_grid(self):
        methods = []
        for classification_method in CLASSIFICATION_METHODS:
            methods.append(
                {
                    **self.preprocessor,
                    **self.select_classifier(classification_method),
                }
            )

        return methods

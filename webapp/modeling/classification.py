from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from quantimage2_backend_common.utils import CV_SPLITS, CV_REPEATS
from modeling.modeling import Modeling
from service.feature_transformation import OUTCOME_FIELD_CLASSIFICATION

CLASSIFICATION_METHODS = [
    "logistic_regression_lbfgs",
    # "logistic_regression_saga",
    # "svm",
    # "random_forest",
]
CLASSIFICATION_PARAMS = {
    "logistic_regression_lbfgs": {
        "solver": ["lbfgs"],
        "penalty": ["l2"],
        "max_iter": [100],
    },
    # "logistic_regression_saga": {
    #     "solver": ["saga"],
    #     "penalty": ["l1", "l2", "elasticnet"],
    #     "l1_ratio": [0.5],
    #     "max_iter": [1000],
    # },
    # "random_forest": {
    #     "max_depth": [10, 100, None],
    #     # "max_features": ["auto", "sqrt"],
    #     # "min_samples_leaf": [1, 2, 4],
    #     # "min_samples_split": [2, 5, 10],
    #     "n_estimators": [10, 100, 1000],
    # },
    # "svm": {
    #     "C": [0.01, 0.1, 1, 10, 100],
    #     "gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001],
    #     "kernel": ["linear", "rbf", "poly", "sigmoid"],
    # },
}


class Classification(Modeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if a positive label is defined for the current label category
        if self.label_category.pos_label:
            available_classes = list(
                set(
                    [
                        l.label_content[OUTCOME_FIELD_CLASSIFICATION]
                        for l in self.label_category.labels
                    ]
                )
            )
            sorted_classes = sorted(
                available_classes,
                key=lambda i: 1 if i == self.label_category.pos_label else 0,
            )
            self.classes = sorted_classes
        else:
            self.classes = None

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
                "classifier": [
                    # Using CalibratedClassifierCV as a work-around to strange behaviour when using "probability=True"
                    # on SVC directly, as documented here : https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities
                    # This allows still getting probabilities & having consistent behaviour of predict & predict_proba
                    CalibratedClassifierCV(
                        SVC(random_state=self.random_seed, probability=False)
                    )
                ]
            }

        for key, value in CLASSIFICATION_PARAMS[classifier_name].items():
            # Parameters need to be passed differently for the CalibratedClassifierCV used for SVM, as they are nested
            # within the "estimator" attribute of the CalibratedClassifierCV object
            if classifier_name == "svm":
                options[f"classifier__estimator__{key}"] = value
            else:
                options[f"classifier__{key}"] = value

        return options

    def encode_labels(self, labels):
        # Classes must be sorted so that the negative class is first & the positive class is second
        if self.classes is not None:
            labels_binarized = label_binarize(labels, classes=self.classes)
            labels_encoded = [l[0] for l in labels_binarized]
        else:
            labels_encoded = [
                int(l) for l in labels[OUTCOME_FIELD_CLASSIFICATION].values
            ]
        return labels_encoded

    def get_cv(self, n_splits=CV_SPLITS, n_repeats=CV_REPEATS):
        return RepeatedStratifiedKFold(
            random_state=self.random_seed, n_splits=n_splits, n_repeats=n_repeats
        )

    def get_scoring(self):
        return {
            "auc": "roc_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "sensitivity": "recall",
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

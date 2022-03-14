import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    recall_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    get_scorer,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.svm import SVC
from ttictoc import tic, toc

# Lists of supported normalization & classification methods
from sklearn.utils import resample

NORMALIZATION_METHODS = ["standardization", "l2norm"]
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
        "max_iter": [10000],
    },
}


def preprocess_features(features):
    features = features.drop("PatientID", axis=1)
    features.sort_index(inplace=True)
    return features


def preprocess_labels(labels, training_patients, test_patients):

    if training_patients and test_patients:
        labels = labels.loc[training_patients + test_patients]

    labels.sort_index(inplace=True)

    return labels


def split_dataset(X, y, training_patients, test_patients):
    X_train = X.loc[training_patients]
    X_test = X.loc[test_patients]

    y_train = y.loc[training_patients]
    y_test = y.loc[test_patients]

    return X_train, X_test, y_train, y_test


def get_cv(random_seed, n_splits=5, n_repeats=1):
    return RepeatedStratifiedKFold(
        random_state=random_seed, n_splits=n_splits, n_repeats=n_repeats
    )


def get_labelencoder():
    return LabelEncoder()


def get_scoring():
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "specificity": make_scorer(recall_score, pos_label=0),
        "auc": "roc_auc",
    }


def select_normalizer(normalization_name):
    scaler = None

    if normalization_name == "standardization":
        scaler = StandardScaler()
    elif normalization_name == "l2norm":
        scaler = Normalizer()

    return scaler


def select_classifier(classifier_name, random_seed):
    if classifier_name.startswith("logistic_regression"):
        options = {"classifier": [LogisticRegression(random_state=random_seed)]}
        for key, value in CLASSIFICATION_PARAMS[classifier_name].items():
            options[f"classifier__{key}"] = value
        return options
    elif classifier_name == "random_forest":
        return {"classifier": [RandomForestClassifier(random_state=random_seed)]}
    elif classifier_name == "svm":
        return {"classifier": [SVC(random_state=random_seed)]}


def run_bootstrap(
    X_test, y_test, model, random_seed, n_bootstrap=100, scoring=get_scoring()
):

    all_scores = []
    i = 0

    for _ in range(n_bootstrap):
        print(f"Running bootstrap n°{i}...")
        tic()
        X_test_resampled, y_test_resampled = resample(
            X_test,
            y_test,
            replace=True,
            n_samples=len(y_test),
            stratify=y_test,
            random_state=random_seed,
        )

        y_pred = model.predict(X_test_resampled)

        scores = calculate_scores(y_test_resampled, y_pred, scoring)

        all_scores.append(scores)

        elapsed = toc()
        print(f"Bootstrap iteration took {int(elapsed * 1000)}ms")

        if i % 100 == 0:
            print(f"Ran {i}/{n_bootstrap} iterations of the Bootstrap run")
        i += 1

    return all_scores, n_bootstrap


def mean_confidence_interval(data, confidence=0.95):
    alpha = 1 - confidence
    return {
        "mean": np.mean(data),
        "inf_value": np.quantile(data, alpha / 2),
        "sup_value": np.quantile(data, 1 - alpha / 2),
    }


def calculate_scores(y_true, y_pred, scoring):
    scores = {}

    for score_name, scorer in scoring.items():
        if type(scorer) == str:
            scorer = get_scorer(scorer)

        scores[score_name] = scorer._score_func(y_true, y_pred)

    return scores


def calculate_test_metrics(scores):

    metrics = {}

    for metric in get_scoring().keys():
        metric_scores = [score[metric] for score in scores]
        metrics[metric] = mean_confidence_interval(metric_scores)

    return metrics


def calculate_training_metrics(cv_results):
    metrics = {}

    for metric in get_scoring().keys():
        metrics[metric] = mean_confidence_interval(cv_results[f"mean_test_{metric}"])

    return metrics

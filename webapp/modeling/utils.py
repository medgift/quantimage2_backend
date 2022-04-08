import numpy as np
from scipy.stats import t
from sklearn.metrics import (
    get_scorer,
)
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder

# Lists of supported normalization & classification methods
from sklearn.utils import resample

NORMALIZATION_METHODS = ["standardization", "l2norm"]


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


def select_normalizer(normalization_name):
    scaler = None

    if normalization_name == "standardization":
        scaler = StandardScaler()
    elif normalization_name == "l2norm":
        scaler = Normalizer()

    return scaler


def get_random_seed(extraction_id=None, collection_id=None):
    return 100000 + collection_id if collection_id else extraction_id


def generate_normalization_methods():
    methods = []
    for normalization_method in NORMALIZATION_METHODS:
        methods.append(select_normalizer(normalization_method))

    return methods


def run_bootstrap(X_test, y_test, model, random_seed, scoring, n_bootstrap=1000):

    all_test_predictions = model.predict(X_test)
    all_test_predictions_probabilities = None

    try:
        all_test_predictions_probabilities = model.predict_proba(X_test)
    except AttributeError:
        pass  # The model can't predict probabilities, ignore this

    all_scores = []

    probabilities_resampled = None

    for i in range(n_bootstrap):

        if all_test_predictions_probabilities is not None:
            predictions_resampled, probabilities_resampled, y_test_resampled = resample(
                all_test_predictions,
                all_test_predictions_probabilities,
                y_test,
                replace=True,
                n_samples=len(y_test),
                random_state=random_seed + i,
                stratify=y_test,
            )
        else:
            predictions_resampled, y_test_resampled = resample(
                all_test_predictions,
                y_test,
                replace=True,
                n_samples=len(y_test),
                random_state=random_seed + i,
                stratify=[o[0] for o in y_test],
            )

        scores = calculate_scores(
            y_test_resampled, predictions_resampled, probabilities_resampled, scoring
        )

        all_scores.append(scores)

        if i % 100 == 0:
            print(f"Ran {i}/{n_bootstrap} iterations of the Bootstrap run")

    return all_scores, n_bootstrap


def mean_confidence_interval(data, confidence=0.95):
    alpha = 1 - confidence
    return {
        "mean": np.mean(data),
        "inf_value": np.quantile(data, alpha / 2),
        "sup_value": np.quantile(data, 1 - alpha / 2),
    }


def calculate_scores(y_true, y_pred, y_pred_proba, scoring):
    scores = {}

    for score_name, scorer in scoring.items():
        if type(scorer) == str:
            scorer = get_scorer(scorer)

        if y_pred_proba is not None:
            try:
                # For metrics that need the probabilities (such as roc_auc_score)
                # Calculate only with the "greater label" probability
                # See https://scikit-learn.org/stable/modules/model_evaluation.html#roc-auc-binary
                scores[score_name] = scorer._score_func(y_true, y_pred_proba[:, 1])
            except ValueError:
                # For metric that only need the binary predictions
                scores[score_name] = scorer._score_func(y_true, y_pred)
        else:
            scores[score_name] = scorer._score_func(y_true, y_pred)

    return scores


def calculate_test_metrics(scores, scoring):

    metrics = {}

    for metric in scoring.keys():
        metric_scores = [score[metric] for score in scores]
        metrics[metric] = mean_confidence_interval(metric_scores)

    return metrics


def calculate_training_metrics(cv_results, scoring):
    metrics = {}

    for metric in scoring.keys():
        metrics[metric] = mean_confidence_interval(cv_results[f"mean_test_{metric}"])

    return metrics


def corrected_std(differences, n_train, n_test):
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    # p_val = t.sf(t_stat, df)
    return t_stat, p_val


def compare_score(s1, s2, n_train, n_test, rope_interval=[-0.01, 0.01]):
    n = s1.shape[0]
    t_stat, p_val = compute_corrected_ttest(s1 - s2, n - 1, n_train, n_test)
    t_post = t(
        n - 1, loc=np.mean(s1 - s2), scale=corrected_std(s1 - s2, n_train, n_test)
    )
    ri = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
    return {
        "t_stat": t_stat,
        "p_value": p_val,
        "proba M1 > M2": 1 - t_post.cdf(rope_interval[1]),
        "proba M1 == M2": ri,
        "proba M1 < M2": t_post.cdf(rope_interval[0]),
    }


def corrected_ci(s, n_train, n_test, alpha=0.95):
    n = s.shape[0]
    mean = np.mean(s)
    std = corrected_std(s, n_train, n_test)
    return t.interval(alpha, n - 1, loc=mean, scale=std)

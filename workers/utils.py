import os
import math
import re
from time import time

import numpy as np
import pandas as pd
from flask import jsonify
from numpy import argmax
from sklearn.metrics import get_scorer, roc_curve
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from scipy import stats

from config_workers import MODELS_BASE_DIR
from quantimage2_backend_common.utils import MessageType


def mean_confidence_interval_student(mean, std, n_samples, confidence=0.95):
    inf_value, sup_value = stats.t.interval(
        alpha=confidence, df=n_samples - 1, loc=mean, scale=std
    )

    return {
        "mean": mean,
        "inf_value": inf_value if not math.isnan(inf_value) else mean,
        "sup_value": sup_value if not math.isnan(sup_value) else mean,
    }


def calculate_training_metrics(
    best_index, cv_results, scoring, random_seed, confidence=0.95
):
    metrics = {}

    for index, metric in enumerate(scoring.keys()):

        # Filter the CV results to keep only the splits for the current metric
        metric_splits = {
            k: v
            for k, v in cv_results.items()
            if re.match(rf"split\d_test_{metric}", k)
        }

        # Get the metric values for the best index
        split_values = [v[best_index] for v in metric_splits.values()]

        # Run bootstrap on the split values
        bootstrapped_values = bootstrap_on_results(
            split_values, random_seed, n_bootstrap=20
        )

        # Calculate mean for each of the boostrap repetitions
        bootstrapped_means = [np.mean(values) for values in bootstrapped_values]

        # Calculate confidence interval based on the bootstrapped means
        metrics[metric] = get_confidence_interval_quartiles(
            split_values, bootstrapped_means, confidence
        )
        metrics[metric]["order"] = index

    return metrics


def calculate_test_metrics(scores, scoring, random_seed, confidence=0.95):

    metrics = {}
    metrics_values = {}

    for index, metric in enumerate(scoring.keys()):
        metric_scores = [score[metric] for score in scores]

        # Run bootstrap on the bootstrapped values
        bootstrapped_values = bootstrap_on_results(
            metric_scores, random_seed, n_bootstrap=50
        )
        # Calculate mean for each of the 2nd-level boostrap repetitions
        bootstrapped_means = [np.mean(values) for values in bootstrapped_values]

        # Save the mean of bootstrapped values for each metric
        metrics_values[metric] = bootstrapped_means

        # Calculate the confidence interval for the metric
        metrics[metric] = get_confidence_interval_quartiles(
            metric_scores, bootstrapped_means, confidence
        )
        metrics[metric]["order"] = index

    return metrics, metrics_values


def get_confidence_interval_quartiles(metric_scores, bootstrapped_means, confidence):
    # Calculate confidence interval based on the bootstrapped means
    inf_value = np.quantile(bootstrapped_means, ((1 - confidence) / 2))
    sup_value = np.quantile(bootstrapped_means, 1 - ((1 - confidence) / 2))

    # Report the true mean of the metric scores
    mean_value = np.mean(metric_scores)

    return {
        "mean": mean_value,
        "inf_value": inf_value,
        "sup_value": sup_value,
    }


def bootstrap_on_results(results, random_seed, n_bootstrap=20):
    bootstrapped_results = []

    for i in range(n_bootstrap):
        resampled_results = resample(
            results,
            replace=True,
            n_samples=len(results),
            random_state=random_seed + i,
        )
        bootstrapped_results.append(resampled_results)

    return bootstrapped_results


def run_bootstrap(
    X_test,
    y_test,
    model,
    random_seed,
    scoring,
    n_bootstrap=1000,
    training_id=None,
    socket_io=None,
):

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

        if i % (n_bootstrap / 10) == 0:
            print(f"Ran {i}/{n_bootstrap} iterations of the Bootstrap run")

            # Status update
            if training_id and socket_io:
                socketio_body = {
                    "training-id": training_id,
                    "phase": "testing",
                    "current": i,
                    "total": n_bootstrap,
                }
                socket_io.emit(
                    MessageType.TRAINING_STATUS.value, jsonify(socketio_body).get_json()
                )

    # Final status update
    if training_id and socket_io:
        socketio_body = {
            "training-id": training_id,
            "phase": "testing",
            "current": n_bootstrap,
            "total": n_bootstrap,
        }
        socket_io.emit(
            MessageType.TRAINING_STATUS.value, jsonify(socketio_body).get_json()
        )

    return all_scores, n_bootstrap


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
                scores[score_name] = scorer._score_func(
                    y_true, y_pred_proba[:, 1], **scorer._kwargs
                )
            except ValueError:
                # For metric that only need the binary predictions
                scores[score_name] = scorer._score_func(
                    y_true, y_pred, **scorer._kwargs
                )
        else:
            scores[score_name] = scorer._score_func(y_true, y_pred, **scorer._kwargs)

    return scores


def get_model_path(user_id, album_id, model_type):
    # Define features path for storing the results
    models_dir = os.path.join(MODELS_BASE_DIR, user_id, album_id)

    models_filename = f"model_{model_type}"

    models_filename += f"_{str(int(time()))}"
    models_filename += ".joblib"
    models_path = os.path.join(models_dir, models_filename)

    return models_path


# TODO - This function might be used in the future if it seems necessary
# Calculate the optimal decision threshold based on Youden's index
def get_optimal_threshold(y, positive_probabilities):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, positive_probabilities)

    # get the best threshold
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = thresholds[ix]
    print("Best Threshold=%f" % (best_thresh))

    return best_thresh


def compute_feature_importance(
        X_test,
        y_test,
        model,
        random_seed
    ):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_seed)
    return pd.Series(result.importances_mean, index=X_test.columns).to_dict()
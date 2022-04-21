import os
import math
from time import time

import numpy as np
from flask import jsonify
from sklearn.metrics import get_scorer
from sklearn.utils import resample
from scipy import stats

from config import MODELS_BASE_DIR
from imaginebackend_common.utils import MessageType


def mean_confidence_interval_student(data, confidence=0.95):
    mean = np.mean(data)
    inf_value, sup_value = stats.t.interval(
        confidence, len(data) - 1, loc=np.mean(data), scale=stats.sem(data)
    )

    return {
        "mean": mean,
        "inf_value": inf_value if not math.isnan(inf_value) else 0,
        "sup_value": sup_value if not math.isnan(sup_value) else 0,
    }


def mean_confidence_interval_normal(data, confidence=0.95):
    mean = np.mean(data)
    stderrmean = stats.sem(data)

    # If the standard error is 0, we only have one value, take the mean as inf & sup value
    if stderrmean == 0:
        inf_value, sup_value = mean, mean
    else:
        inf_value, sup_value = stats.norm.interval(
            confidence, loc=mean, scale=stats.sem(data)
        )

    return {
        "mean": mean,
        "inf_value": inf_value if not math.isnan(inf_value) else 0,
        "sup_value": sup_value if not math.isnan(sup_value) else 0,
    }


def calculate_training_metrics(cv_results, scoring):
    metrics = {}

    for index, metric in enumerate(scoring.keys()):
        metrics[metric] = mean_confidence_interval_student(
            cv_results[f"mean_test_{metric}"]
        )
        metrics[metric]["order"] = index

    return metrics


def calculate_test_metrics(scores, scoring):

    metrics = {}

    for index, metric in enumerate(scoring.keys()):
        metric_scores = [score[metric] for score in scores]
        metrics[metric] = mean_confidence_interval_normal(metric_scores)
        metrics[metric]["order"] = index

    return metrics


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
                scores[score_name] = scorer._score_func(y_true, y_pred_proba[:, 1])
            except ValueError:
                # For metric that only need the binary predictions
                scores[score_name] = scorer._score_func(y_true, y_pred)
        else:
            scores[score_name] = scorer._score_func(y_true, y_pred)

    return scores


def get_model_path(user_id, album_id, model_type):
    # Define features path for storing the results
    models_dir = os.path.join(MODELS_BASE_DIR, user_id, album_id)

    models_filename = f"model_{model_type}"

    models_filename += f"_{str(int(time()))}"
    models_filename += ".joblib"
    models_path = os.path.join(models_dir, models_filename)

    return models_path

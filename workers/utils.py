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


def mean_confidence_interval_student(mean, std, n_samples, confidence=0.95):
    inf_value, sup_value = stats.t.interval(
        alpha=confidence, df=n_samples - 1, loc=mean, scale=std
    )

    return {
        "mean": mean,
        "inf_value": inf_value if not math.isnan(inf_value) else mean,
        "sup_value": sup_value if not math.isnan(sup_value) else mean,
    }


def calculate_training_metrics(best_index, cv_results, n_splits, scoring):
    metrics = {}

    for index, metric in enumerate(scoring.keys()):
        metrics[metric] = mean_confidence_interval_student(
            cv_results[f"mean_test_{metric}"][best_index],
            cv_results[f"std_test_{metric}"][best_index],
            n_splits,
        )
        metrics[metric]["order"] = index

    return metrics


def calculate_test_metrics(scores, scoring, confidence=0.95):

    metrics = {}

    for index, metric in enumerate(scoring.keys()):
        metric_scores = [score[metric] for score in scores]

        inf_value = np.quantile(metric_scores, ((1 - confidence) / 2))
        sup_value = np.quantile(metric_scores, 1 - ((1 - confidence) / 2))
        mean_value = np.mean(metric_scores)

        metrics[metric] = {
            "mean": mean_value,
            "inf_value": inf_value if not math.isnan(inf_value) else 0,
            "sup_value": sup_value if not math.isnan(sup_value) else 0,
        }
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

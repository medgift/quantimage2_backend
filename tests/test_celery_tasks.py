"""
Tests for Celery tasks and worker utility functions.

Covers worker utility functions (metrics calculation, bootstrap, model paths,
scoring), and Celery task dispatch/mock patterns.
"""

import os
import math
import importlib
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score

# ---------------------------------------------------------------------------
# Worker utilities (workers/utils.py)
# ---------------------------------------------------------------------------


class TestMeanConfidenceInterval:
    def test_normal_case(self):
        from utils import mean_confidence_interval_student

        result = mean_confidence_interval_student(
            mean=0.85, std=0.05, n_samples=100, confidence=0.95
        )
        assert "mean" in result
        assert "inf_value" in result
        assert "sup_value" in result
        assert result["mean"] == 0.85
        assert result["inf_value"] < result["mean"]
        assert result["sup_value"] > result["mean"]

    def test_single_sample_returns_mean(self):
        """With n=1 (df=0), t-distribution is undefined — should handle gracefully."""
        from utils import mean_confidence_interval_student

        result = mean_confidence_interval_student(
            mean=0.75, std=0.1, n_samples=1, confidence=0.95
        )
        # With 0 degrees of freedom, inf/sup may be nan → replaced by mean
        assert result["mean"] == 0.75

    def test_zero_std(self):
        from utils import mean_confidence_interval_student

        result = mean_confidence_interval_student(
            mean=0.9, std=0.0, n_samples=10, confidence=0.95
        )
        assert result["inf_value"] == result["sup_value"] == result["mean"]


class TestBootstrapOnResults:
    def test_bootstrap_returns_correct_count(self):
        from utils import bootstrap_on_results

        results = [0.8, 0.82, 0.85, 0.79, 0.81]
        bootstrapped = bootstrap_on_results(results, random_seed=42, n_bootstrap=10)
        assert len(bootstrapped) == 10
        for sample in bootstrapped:
            assert len(sample) == len(results)

    def test_bootstrap_is_reproducible(self):
        from utils import bootstrap_on_results

        results = [0.1, 0.2, 0.3, 0.4, 0.5]
        b1 = bootstrap_on_results(results, random_seed=42, n_bootstrap=5)
        b2 = bootstrap_on_results(results, random_seed=42, n_bootstrap=5)
        for s1, s2 in zip(b1, b2):
            np.testing.assert_array_equal(s1, s2)


class TestGetConfidenceIntervalQuartiles:
    def test_basic_interval(self):
        from utils import get_confidence_interval_quartiles

        metric_scores = [0.80, 0.82, 0.85, 0.79, 0.81]
        bootstrapped_means = [
            0.81,
            0.82,
            0.80,
            0.83,
            0.82,
            0.81,
            0.84,
            0.80,
            0.82,
            0.81,
        ]

        result = get_confidence_interval_quartiles(
            metric_scores, bootstrapped_means, confidence=0.95
        )
        assert "mean" in result
        assert "inf_value" in result
        assert "sup_value" in result
        assert result["inf_value"] <= result["mean"]
        assert result["sup_value"] >= result["mean"]


class TestCalculateTrainingMetrics:
    def test_training_metrics_structure(self):
        from utils import calculate_training_metrics

        # Simulate CV results with 2 metrics and 3 splits
        cv_results = {
            "split0_test_auc": np.array([0.85, 0.80]),
            "split1_test_auc": np.array([0.82, 0.78]),
            "split2_test_auc": np.array([0.88, 0.83]),
            "split0_test_accuracy": np.array([0.90, 0.85]),
            "split1_test_accuracy": np.array([0.88, 0.82]),
            "split2_test_accuracy": np.array([0.92, 0.87]),
        }
        scoring = {"auc": "roc_auc", "accuracy": "accuracy"}
        best_index = 0

        metrics = calculate_training_metrics(
            best_index, cv_results, scoring, random_seed=42
        )
        assert "auc" in metrics
        assert "accuracy" in metrics
        for metric_name in metrics:
            assert "mean" in metrics[metric_name]
            assert "inf_value" in metrics[metric_name]
            assert "sup_value" in metrics[metric_name]
            assert "order" in metrics[metric_name]


class TestCalculateTestMetrics:
    def test_test_metrics_structure(self):
        from utils import calculate_test_metrics

        scores = [
            {"auc": 0.85, "accuracy": 0.90},
            {"auc": 0.82, "accuracy": 0.88},
            {"auc": 0.88, "accuracy": 0.92},
            {"auc": 0.84, "accuracy": 0.89},
            {"auc": 0.86, "accuracy": 0.91},
        ]
        scoring = {"auc": "roc_auc", "accuracy": "accuracy"}

        metrics, values = calculate_test_metrics(scores, scoring, random_seed=42)
        assert "auc" in metrics
        assert "accuracy" in metrics
        assert "auc" in values
        assert "accuracy" in values


class TestGetModelPath:
    def test_model_path_format(self):
        from utils import get_model_path

        path = get_model_path("user-123", "album-456", "Classification")
        assert "user-123" in path
        assert "album-456" in path
        assert "Classification" in path
        assert path.endswith(".joblib")


class TestCalculateScores:
    def test_classification_scores(self):
        from utils import calculate_scores

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_pred_proba = np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]
        )

        scoring = {
            "auc": "roc_auc",
            "accuracy": "accuracy",
            "specificity": make_scorer(recall_score, pos_label=0),
        }

        scores = calculate_scores(y_true, y_pred, y_pred_proba, scoring)
        assert "auc" in scores
        assert "accuracy" in scores
        assert "specificity" in scores
        assert 0 <= scores["auc"] <= 1
        assert 0 <= scores["accuracy"] <= 1

    def test_scores_without_probabilities(self):
        from utils import calculate_scores

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        scoring = {"accuracy": "accuracy"}

        scores = calculate_scores(y_true, y_pred, None, scoring)
        assert "accuracy" in scores
        assert scores["accuracy"] == 0.75


# ---------------------------------------------------------------------------
# Compute predictions
# ---------------------------------------------------------------------------


class TestComputePredictions:
    def test_compute_predictions(self):
        from utils import compute_predictions

        # Create a simple trained model
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])
        model.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        )

        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        patients = ["P1", "P2", "P3"]

        predictions, probabilities = compute_predictions(X, model, patients)
        assert "P1" in predictions
        assert "P2" in predictions
        assert "P3" in predictions
        assert predictions["P1"]["prediction"] == 0
        assert predictions["P2"]["prediction"] == 1
        assert "probabilities" in probabilities["P1"]


class TestComputeRiskScores:
    def test_compute_risk_scores_cox(self):
        from utils import compute_risk_scores

        model = MagicMock()
        model.predict.return_value = np.array([1.5, -0.5, 0.8])
        # Simulate CoxPH analyzer
        analyzer_mock = MagicMock()
        analyzer_mock.__class__.__name__ = "CoxPHSurvivalAnalysis"
        model.best_estimator_ = {"analyzer": analyzer_mock}

        X = pd.DataFrame({"f1": [1, 2, 3]})
        patients = ["P1", "P2", "P3"]

        risk_scores = compute_risk_scores(X, model, patients)
        assert risk_scores["P1"]["risk_score"] == 1.5
        assert risk_scores["P2"]["risk_score"] == -0.5

    def test_compute_risk_scores_ipc_ridge(self):
        from utils import compute_risk_scores

        model = MagicMock()
        model.predict.return_value = np.array([100.0, 200.0, 50.0])
        # Simulate IPCRidge analyzer
        analyzer_mock = MagicMock()
        analyzer_mock.__class__.__name__ = "IPCRidge"
        model.best_estimator_ = {"analyzer": analyzer_mock}

        X = pd.DataFrame({"f1": [1, 2, 3]})
        patients = ["P1", "P2", "P3"]

        risk_scores = compute_risk_scores(X, model, patients)
        # IPCRidge negates the predictions
        assert risk_scores["P1"]["risk_score"] == -100.0
        assert risk_scores["P2"]["risk_score"] == -200.0


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


class TestComputeFeatureImportance:
    @patch("utils.permutation_importance")
    def test_classification_importance(self, mock_pi):
        from utils import compute_feature_importance

        # compute_feature_importance passes a single scorer to
        # permutation_importance, so sklearn returns one result object - not a
        # dict keyed by metric name, which is what it returns for several.
        mock_result = MagicMock()
        mock_result.importances_mean = np.array([0.05, 0.15, 0.10])
        mock_pi.return_value = mock_result

        X = pd.DataFrame(
            {"feat_a": [1, 2, 3], "feat_b": [4, 5, 6], "feat_c": [7, 8, 9]}
        )
        y = np.array([0, 1, 0])
        model = MagicMock()
        scoring = {"auc": "roc_auc"}

        importances = compute_feature_importance(
            X, y, model, scoring, "Classification", 42
        )
        assert "feat_a" in importances
        assert "feat_b" in importances
        assert importances["feat_b"] == 0.15

        # Guard the single-scorer contract itself: passing the whole scoring
        # dict is what makes this call expensive, and it also changes the shape
        # of what comes back.
        assert mock_pi.call_args.kwargs["scoring"] == "roc_auc"

    @staticmethod
    def _inputs():
        X = pd.DataFrame(
            {"feat_a": [1, 2, 3], "feat_b": [4, 5, 6], "feat_c": [7, 8, 9]}
        )
        y = np.array([0, 1, 0])
        return X, y

    @staticmethod
    def _result():
        mock_result = MagicMock()
        mock_result.importances_mean = np.array([0.05, 0.15, 0.10])
        return mock_result

    @patch("utils.permutation_importance")
    def test_feature_importance_runs_single_core_by_default(self, mock_pi):
        """Without an explicit budget, permutation importance stays on 1 core.

        Callers that don't know the worker's CPU budget must not accidentally
        fan out (e.g. n_jobs=-1 would oversubscribe concurrent trainings).
        """
        from utils import compute_feature_importance

        mock_pi.return_value = self._result()
        X, y = self._inputs()

        compute_feature_importance(
            X, y, MagicMock(), {"auc": "roc_auc"}, "Classification", 42
        )

        assert mock_pi.call_count == 1
        assert "n_jobs" in mock_pi.call_args.kwargs
        assert mock_pi.call_args.kwargs["n_jobs"] == 1

    @patch("utils.permutation_importance")
    def test_feature_importance_forwards_explicit_n_jobs(self, mock_pi):
        from utils import compute_feature_importance

        mock_pi.return_value = self._result()
        X, y = self._inputs()

        importances = compute_feature_importance(
            X, y, MagicMock(), {"auc": "roc_auc"}, "Classification", 42, n_jobs=6
        )

        assert mock_pi.call_args.kwargs["n_jobs"] == 6
        # The rest of the call is unchanged by the new keyword
        assert mock_pi.call_args.kwargs["scoring"] == "roc_auc"
        assert mock_pi.call_args.kwargs["random_state"] == 42
        assert importances == {"feat_a": 0.05, "feat_b": 0.15, "feat_c": 0.10}

    @patch("utils.permutation_importance")
    def test_survival_feature_importance_forwards_n_jobs(self, mock_pi):
        from utils import compute_feature_importance

        mock_pi.return_value = self._result()
        X, y = self._inputs()
        c_index_scorer = MagicMock(name="c_index_scorer")

        compute_feature_importance(
            X,
            y,
            MagicMock(),
            {"c-index": c_index_scorer},
            "Survival",
            7,
            n_jobs=3,
        )

        assert mock_pi.call_args.kwargs["n_jobs"] == 3
        assert mock_pi.call_args.kwargs["scoring"] is c_index_scorer


# ---------------------------------------------------------------------------
# Celery task registration (basic sanity checks)
# ---------------------------------------------------------------------------


class TestCeleryTaskRegistration:
    """Verify task names and signatures exist.

    These tests require the workers environment (okapy, pyradiomics, etc.).
    They are skipped when running in the backend container.
    """

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_train_task_exists(self):
        """The train task should be importable."""
        from tasks import train_model

        assert train_model is not None
        assert train_model.name == "quantimage2tasks.train"

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_extract_task_exists(self):
        """The extract task should be importable."""
        from tasks import run_extraction

        assert run_extraction is not None
        assert run_extraction.name == "quantimage2tasks.extract"

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_extract_task_has_time_limits(self):
        """Extraction task should have hard and soft time limits."""
        from tasks import run_extraction

        # These are set as decorator arguments
        assert run_extraction.time_limit == 7200
        assert run_extraction.soft_time_limit == 6600

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_extract_task_still_acks_late(self):
        """Extraction keeps acks_late so a crashed extraction is redelivered."""
        from tasks import run_extraction

        assert run_extraction.acks_late is True

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_train_task_does_not_ack_late(self):
        """Regression #54: a long training must not be redelivered.

        With acks_late, the message of a training running longer than the
        broker visibility timeout is handed to another worker slot and the
        same training runs twice concurrently.
        """
        from tasks import train_model

        assert train_model.acks_late is False

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_train_task_has_finite_time_limits(self):
        """Training must have a soft limit that fires before the hard kill."""
        from tasks import train_model

        assert isinstance(train_model.soft_time_limit, (int, float))
        assert isinstance(train_model.time_limit, (int, float))
        assert 0 < train_model.soft_time_limit < train_model.time_limit

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_visibility_timeout_exceeds_every_acks_late_task_time_limit(self):
        """No acks_late task may outlive the broker visibility timeout.

        If it does, Redis redelivers the still-running message and the task
        runs twice. Checks the configuration as applied to the real app.
        """
        import tasks

        conf = tasks.celery.conf
        if os.environ.get("CELERY_CONFIG_MODULE") != "celeryconfig":
            pytest.skip("celeryconfig not wired via CELERY_CONFIG_MODULE here")

        visibility_timeout = conf.broker_transport_options["visibility_timeout"]
        own_tasks = {
            name: task
            for name, task in tasks.celery.tasks.items()
            if name.startswith("quantimage2tasks.")
        }
        assert "quantimage2tasks.extract" in own_tasks
        assert "quantimage2tasks.train" in own_tasks

        checked = []
        for name, task in own_tasks.items():
            if task.acks_late:
                assert task.time_limit is not None, name
                assert visibility_timeout > task.time_limit, name
                checked.append(name)
        assert "quantimage2tasks.extract" in checked

    @pytest.mark.skipif(
        not importlib.util.find_spec("okapy"),
        reason="okapy not available (workers-only dependency)",
    )
    def test_worker_app_uses_redelivery_safe_config(self):
        """The real worker app picks up the fixed celeryconfig values."""
        import tasks

        if os.environ.get("CELERY_CONFIG_MODULE") != "celeryconfig":
            pytest.skip("celeryconfig not wired via CELERY_CONFIG_MODULE here")

        conf = tasks.celery.conf
        assert conf.worker_prefetch_multiplier == 1
        assert conf.task_acks_late is True
        assert conf.broker_transport_options["visibility_timeout"] == 21600
        assert "visibility_timeout" not in conf.result_backend_transport_options


# ---------------------------------------------------------------------------
# Celery configuration (workers/celeryconfig.py)
# ---------------------------------------------------------------------------

_KEEPALIVE_OPTIONS = {
    "retry_on_timeout": True,
    "socket_keepalive": True,
    "socket_keepalive_options": {4: 60, 5: 10, 6: 5},
    "health_check_interval": 30,
}


@pytest.fixture
def celeryconfig_module(monkeypatch):
    """Import a fresh ``celeryconfig`` and restore sys.modules afterwards."""
    import sys

    monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "4")
    saved = sys.modules.pop("celeryconfig", None)
    try:
        yield importlib.import_module("celeryconfig")
    finally:
        if saved is not None:
            sys.modules["celeryconfig"] = saved
        else:
            sys.modules.pop("celeryconfig", None)


class TestCeleryConfig:
    """Broker/worker settings that prevent redelivery of long tasks (#54)."""

    def test_broker_visibility_timeout_is_six_hours(self, celeryconfig_module):
        opts = celeryconfig_module.broker_transport_options
        assert opts.get("visibility_timeout") == 21600

    def test_broker_visibility_timeout_exceeds_extraction_hard_limit(
        self, celeryconfig_module
    ):
        """Extraction (acks_late, hard limit 7200 s) must never be redelivered
        while still running."""
        opts = celeryconfig_module.broker_transport_options
        assert opts.get("visibility_timeout", 3600) > 7200

    def test_result_backend_has_no_visibility_timeout(self, celeryconfig_module):
        opts = celeryconfig_module.result_backend_transport_options
        assert "visibility_timeout" not in opts

    def test_broker_and_result_backend_options_are_separate_objects(
        self, celeryconfig_module
    ):
        assert (
            celeryconfig_module.broker_transport_options
            is not celeryconfig_module.result_backend_transport_options
        )

    @pytest.mark.parametrize(
        "attr", ["broker_transport_options", "result_backend_transport_options"]
    )
    def test_keepalive_options_preserved(self, celeryconfig_module, attr):
        opts = getattr(celeryconfig_module, attr)
        for key, value in _KEEPALIVE_OPTIONS.items():
            assert opts.get(key) == value, f"{attr}[{key!r}]"

    def test_prefetch_multiplier_is_one(self, celeryconfig_module):
        """Regression #54: reserved-but-unstarted messages are exposed to the
        visibility timeout too; a slot must only hold the task it runs."""
        assert celeryconfig_module.worker_prefetch_multiplier == 1

    def test_task_acks_late_stays_enabled_globally(self, celeryconfig_module):
        assert celeryconfig_module.task_acks_late is True

    def test_acks_on_failure_or_timeout_kept(self, celeryconfig_module):
        assert celeryconfig_module.task_acks_on_failure_or_timeout is True

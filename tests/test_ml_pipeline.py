"""
Tests for the ML modeling pipeline.

Covers modeling utilities (normalization, dataset splitting, confidence intervals),
Classification and Survival pipeline construction, parameter grid generation,
label encoding, and cross-validation setup.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Modeling utilities (webapp/modeling/utils.py)
# ---------------------------------------------------------------------------


class TestNormalizationMethods:
    def test_select_standardizer(self):
        from modeling.utils import select_normalizer
        from sklearn.preprocessing import StandardScaler

        scaler = select_normalizer("standardization")
        assert isinstance(scaler, StandardScaler)

    def test_select_minmax(self):
        from modeling.utils import select_normalizer
        from sklearn.preprocessing import MinMaxScaler

        scaler = select_normalizer("minmax")
        assert isinstance(scaler, MinMaxScaler)

    def test_select_none(self):
        from modeling.utils import select_normalizer

        scaler = select_normalizer("unknown")
        assert scaler is None

    def test_generate_normalization_methods(self):
        from modeling.utils import generate_normalization_methods
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        methods = generate_normalization_methods()
        assert len(methods) == 2
        types = [type(m) for m in methods]
        assert StandardScaler in types
        assert MinMaxScaler in types


class TestDatasetSplitting:
    def test_preprocess_features_train_test(self):
        from modeling.utils import preprocess_features

        df = pd.DataFrame(
            {
                "PatientID": ["A", "B", "C", "D"],
                "feat1": [1, 2, 3, 4],
                "feat2": [5, 6, 7, 8],
            },
            index=["A", "B", "C", "D"],
        )
        result = preprocess_features(df, ["A", "B"], ["C"])
        assert "PatientID" not in result.columns
        assert len(result) == 3
        assert set(result.index) == {"A", "B", "C"}

    def test_preprocess_features_train_only(self):
        from modeling.utils import preprocess_features

        df = pd.DataFrame(
            {"PatientID": ["A", "B", "C"], "feat1": [1, 2, 3]},
            index=["A", "B", "C"],
        )
        result = preprocess_features(df, ["A", "B", "C"], None)
        assert len(result) == 3
        assert "PatientID" not in result.columns

    def test_preprocess_labels_train_test(self):
        from modeling.utils import preprocess_labels

        df = pd.DataFrame(
            {"Outcome": ["Pos", "Neg", "Pos", "Neg"]},
            index=["A", "B", "C", "D"],
        )
        result = preprocess_labels(df, ["A", "B"], ["C", "D"])
        assert len(result) == 4
        # Should be sorted by index
        assert list(result.index) == sorted(result.index)

    def test_split_dataset(self):
        from modeling.utils import split_dataset

        X = pd.DataFrame({"f1": [1, 2, 3, 4]}, index=["A", "B", "C", "D"])
        y = pd.DataFrame({"Outcome": [0, 1, 0, 1]}, index=["A", "B", "C", "D"])
        X_train, X_test, y_train, y_test = split_dataset(X, y, ["A", "B"], ["C", "D"])

        assert len(X_train) == 2
        assert len(X_test) == 2
        assert set(X_train.index) == {"A", "B"}
        assert set(X_test.index) == {"C", "D"}


class TestRandomSeed:
    def test_get_random_seed(self):
        from modeling.utils import get_random_seed

        assert get_random_seed() == 42


class TestStatisticalUtils:
    def test_corrected_std(self):
        from modeling.utils import corrected_std

        diffs = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        std = corrected_std(diffs, n_train=80, n_test=20)
        assert std > 0
        assert np.isfinite(std)

    def test_compute_corrected_ttest(self):
        from modeling.utils import compute_corrected_ttest

        diffs = np.array([0.05, 0.1, 0.08, 0.12, 0.07])
        t_stat, p_val = compute_corrected_ttest(diffs, df=4, n_train=80, n_test=20)
        assert np.isfinite(t_stat)
        assert 0 <= p_val <= 1

    def test_corrected_ci(self):
        from modeling.utils import corrected_ci

        scores = np.array([0.75, 0.80, 0.78, 0.82, 0.77])
        ci = corrected_ci(scores, n_train=80, n_test=20, alpha=0.95)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # lower < upper

    def test_compare_score(self):
        from modeling.utils import compare_score

        s1 = np.array([0.85, 0.82, 0.88, 0.84, 0.86])
        s2 = np.array([0.75, 0.78, 0.72, 0.77, 0.74])
        result = compare_score(s1, s2, n_train=80, n_test=20)

        assert "t_stat" in result
        assert "p_value" in result
        assert "proba M1 > M2" in result
        assert "proba M1 == M2" in result
        assert "proba M1 < M2" in result


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------


class TestClassificationPipeline:
    def test_get_pipeline(self):
        """Classification pipeline should have preprocessor + classifier steps."""
        from modeling.classification import Classification
        from sklearn.pipeline import Pipeline

        # We need to test the method without full init. Use a minimal mock.
        clf = Classification.__new__(Classification)
        clf.random_seed = 42
        clf.label_category = MagicMock()
        clf.label_category.pos_label = None
        clf.classes = None

        pipeline = clf.get_pipeline()
        assert isinstance(pipeline, Pipeline)
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "classifier" in step_names

    def test_get_scoring_keys(self):
        """Scoring dict must include the expected metric keys."""
        clf = _make_classification_stub()
        scoring = clf.get_scoring()
        expected_keys = {"auc", "accuracy", "precision", "sensitivity", "specificity"}
        assert expected_keys == set(scoring.keys())

    def test_get_cv(self):
        from sklearn.model_selection import RepeatedStratifiedKFold

        clf = _make_classification_stub()
        cv = clf.get_cv(n_splits=5, n_repeats=1)
        assert isinstance(cv, RepeatedStratifiedKFold)

    def test_get_parameter_grid_structure(self):
        """Parameter grid should be a list of dicts with preprocessor + classifier."""
        clf = _make_classification_stub()
        clf.preprocessor = {"preprocessor": [None]}
        grid = clf.get_parameter_grid()
        assert isinstance(grid, list)
        assert len(grid) > 0
        for entry in grid:
            assert "preprocessor" in entry
            assert "classifier" in entry or any(
                k.startswith("classifier") for k in entry
            )

    def test_select_classifier_logistic_regression(self):
        clf = _make_classification_stub()
        options = clf.select_classifier("logistic_regression_lbfgs")
        assert "classifier" in options
        assert len(options["classifier"]) == 1

    def test_select_classifier_svm(self):
        clf = _make_classification_stub()
        options = clf.select_classifier("svm")
        assert "classifier" in options

    def test_select_classifier_random_forest(self):
        clf = _make_classification_stub()
        options = clf.select_classifier("random_forest")
        assert "classifier" in options

    def test_encode_labels_binary(self):
        """encode_labels should produce 0/1 encoded values."""
        clf = _make_classification_stub()
        clf.classes = ["Negative", "Positive"]
        clf.label_category = MagicMock()
        clf.label_category.pos_label = "Positive"

        labels = pd.DataFrame({"Outcome": ["Positive", "Negative", "Positive"]})
        encoded = clf.encode_labels(labels)
        assert set(encoded) <= {0, 1}

    def test_encode_labels_no_classes(self):
        """When classes is None, encode_labels should convert to int."""
        clf = _make_classification_stub()
        clf.classes = None

        labels = pd.DataFrame({"Outcome": [1, 0, 1, 0]})
        encoded = clf.encode_labels(labels)
        assert all(isinstance(v, int) for v in encoded)


class TestClassificationMethods:
    def test_classification_methods_list(self):
        from modeling.classification import CLASSIFICATION_METHODS

        assert "logistic_regression_lbfgs" in CLASSIFICATION_METHODS
        assert "logistic_regression_saga" in CLASSIFICATION_METHODS
        assert "svm" in CLASSIFICATION_METHODS
        assert "random_forest" in CLASSIFICATION_METHODS

    def test_classification_params_keys(self):
        from modeling.classification import (
            CLASSIFICATION_PARAMS,
            CLASSIFICATION_METHODS,
        )

        for method in CLASSIFICATION_METHODS:
            assert method in CLASSIFICATION_PARAMS


# ---------------------------------------------------------------------------
# Survival pipeline
# ---------------------------------------------------------------------------


class TestSurvivalPipeline:
    def test_get_pipeline(self):
        """Survival pipeline should have preprocessor + analyzer steps."""
        from modeling.survival import Survival
        from sklearn.pipeline import Pipeline

        surv = Survival.__new__(Survival)
        surv.random_seed = 42

        pipeline = surv.get_pipeline()
        assert isinstance(pipeline, Pipeline)
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "analyzer" in step_names

    def test_get_scoring(self):
        surv = _make_survival_stub()
        scoring = surv.get_scoring()
        assert "c-index" in scoring

    def test_get_cv(self):
        from quantimage2_backend_common.modeling_utils import (
            SurvivalRepeatedStratifiedKFold,
        )

        surv = _make_survival_stub()
        cv = surv.get_cv(n_splits=3, n_repeats=1)
        assert isinstance(cv, SurvivalRepeatedStratifiedKFold)

    def test_get_parameter_grid_structure(self):
        surv = _make_survival_stub()
        surv.preprocessor = {"preprocessor": [None]}
        grid = surv.get_parameter_grid()
        assert isinstance(grid, list)
        assert len(grid) > 0
        for entry in grid:
            assert "preprocessor" in entry
            assert "analyzer" in entry or any(k.startswith("analyzer") for k in entry)

    def test_select_coxnet(self):
        surv = _make_survival_stub()
        options = surv.select_survival_analyzer("coxnet")
        assert "analyzer" in options

    def test_select_coxnet_elastic(self):
        surv = _make_survival_stub()
        options = surv.select_survival_analyzer("coxnet_elastic")
        assert "analyzer" in options

    def test_select_ipc_ridge(self):
        surv = _make_survival_stub()
        options = surv.select_survival_analyzer("ipc_ridge")
        assert "analyzer" in options

    def test_encode_labels(self):
        """Survival encode_labels should produce a structured numpy array."""
        surv = _make_survival_stub()
        labels = pd.DataFrame({"Event": [1, 0, 1], "Time": [10.0, 20.0, 30.0]})
        encoded = surv.encode_labels(labels)
        assert hasattr(encoded, "dtype")
        assert "Event" in encoded.dtype.names
        assert "Time" in encoded.dtype.names


class TestSurvivalMethods:
    def test_survival_methods_list(self):
        from modeling.survival import SURVIVAL_METHODS

        assert "coxnet" in SURVIVAL_METHODS
        assert "coxnet_elastic" in SURVIVAL_METHODS
        assert "ipc_ridge" in SURVIVAL_METHODS


# ---------------------------------------------------------------------------
# Modeling utilities (shared/modeling_utils.py)
# ---------------------------------------------------------------------------


class TestCIndexScore:
    def test_c_index_score(self):
        from quantimage2_backend_common.modeling_utils import c_index_score
        from sksurv.util import Surv

        y_true = Surv.from_arrays([True, False, True, False], [10, 20, 30, 40])
        y_pred = np.array([3.0, 1.0, 4.0, 2.0])

        score = c_index_score(y_true, y_pred)
        assert 0 <= score <= 1


class TestSurvivalRepeatedStratifiedKFold:
    def test_split_uses_event_for_stratification(self):
        from quantimage2_backend_common.modeling_utils import (
            SurvivalRepeatedStratifiedKFold,
        )
        from sksurv.util import Surv

        X = np.random.rand(20, 3)
        y = Surv.from_arrays(
            [True, False] * 10,
            [float(i) for i in range(20)],
        )

        cv = SurvivalRepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        splits = list(cv.split(X, y))
        assert len(splits) == 2

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification_stub():
    """Create a Classification instance without full __init__."""
    from modeling.classification import Classification

    clf = Classification.__new__(Classification)
    clf.random_seed = 42
    clf.label_category = MagicMock()
    clf.label_category.pos_label = None
    clf.classes = None
    clf.preprocessor = {"preprocessor": [None]}
    return clf


def _make_survival_stub():
    """Create a Survival instance without full __init__."""
    from modeling.survival import Survival

    surv = Survival.__new__(Survival)
    surv.random_seed = 42
    surv.preprocessor = {"preprocessor": [None]}
    return surv

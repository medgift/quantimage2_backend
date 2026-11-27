"""
Tests for user-facing errors when a survival test set has too few events.

Covers:
- ``utils.TrainingDataError``: ValueError subclass carrying a user message
- ``utils.run_bootstrap``: zero-event resamples raise TrainingDataError
  instead of sksurv's bare ``ValueError: All samples are censored``
- ``utils.validate_metrics_no_nan``: raises TrainingDataError
- ``tasks.train_model``: fails fast (before grid search) when a survival test
  set has no events, and reports TrainingDataError messages without a
  traceback while other errors keep the full traceback
"""

import importlib.util
import math
import re

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from quantimage2_backend_common.const import FAKE_SCORER_KEY
from quantimage2_backend_common.modeling_utils import c_index_score
from quantimage2_backend_common.utils import MessageType

requires_okapy = pytest.mark.skipif(
    not importlib.util.find_spec("okapy"),
    reason="okapy not available (workers-only dependency)",
)

TRAINING_STATUS = MessageType.TRAINING_STATUS.value

# Field names used by webapp/modeling/survival.py (Surv.from_dataframe) and a
# second, unrelated pair to prove the code reads names from the dtype.
FIELD_NAMES = [("Event", "Time"), ("status", "survival_days")]


def _survival_y(events, times=None, names=("Event", "Time")):
    events = np.asarray(events, dtype=bool)
    if times is None:
        times = np.arange(1, len(events) + 1, dtype=float) * 10.0
    return Surv.from_arrays(events, times, name_event=names[0], name_time=names[1])


def _survival_scoring():
    """Same scoring dict as production (webapp/modeling/survival.py)."""
    return {"c-index": make_scorer(c_index_score)}


class _RiskModel:
    """Survival-like model: predict() returns a risk score, no predict_proba."""

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def _no_comparable_pairs_test_set(names=("Event", "Time"), n=10):
    """n test patients, 1 event at the latest time: zero comparable pairs."""
    events = np.zeros(n, dtype=bool)
    events[-1] = True
    times = np.arange(1, n + 1, dtype=float) * 10.0
    X_test = np.column_stack([np.arange(n, dtype=float), np.ones(n)])
    return X_test, _survival_y(events, times, names=names)


def _mentions_larger_test_set(message):
    m = message.lower()
    return "test set" in m and any(
        word in m for word in ("larger", "increase", "bigger", "more patients")
    )


# ---------------------------------------------------------------------------
# TrainingDataError
# ---------------------------------------------------------------------------


class TestTrainingDataError:
    def test_is_value_error_subclass(self):
        from utils import TrainingDataError

        assert issubclass(TrainingDataError, ValueError)

    def test_str_is_the_user_message(self):
        from utils import TrainingDataError

        exc = TrainingDataError("Please use a larger test set.")

        assert str(exc) == "Please use a larger test set."


# ---------------------------------------------------------------------------
# validate_metrics_no_nan
# ---------------------------------------------------------------------------


class TestValidateMetricsNoNanError:
    def test_nan_metrics_raise_training_data_error(self):
        from utils import TrainingDataError, validate_metrics_no_nan

        with pytest.raises(TrainingDataError) as exc_info:
            validate_metrics_no_nan(
                {"c-index": {"mean": float("nan")}}, None, None, n_test_patients=7
            )

        message = str(exc_info.value)
        # Existing message content is preserved
        assert "test_metrics (1 NaN values)" in message
        assert "current size: 7 patients" in message
        assert "Please increase the test set size and try again." in message

    def test_still_catchable_as_value_error(self):
        from utils import validate_metrics_no_nan

        with pytest.raises(ValueError):
            validate_metrics_no_nan(None, [float("nan")], None, n_test_patients=3)

    def test_finite_metrics_do_not_raise(self):
        from utils import validate_metrics_no_nan

        validate_metrics_no_nan(
            {"c-index": {"mean": 0.7}}, {"c-index": [0.7]}, [{"c-index": 0.7}], 10
        )


# ---------------------------------------------------------------------------
# run_bootstrap
# ---------------------------------------------------------------------------


class TestRunBootstrapSurvivalEvents:
    @pytest.mark.parametrize("names", FIELD_NAMES)
    def test_resample_without_events_raises_training_data_error(self, names):
        """A resample with no events must not surface sksurv's bare error.

        The bootstrap stratifies on the event indicator, so every resample
        keeps the test set's event count: a zero-event resample happens
        exactly when the test set itself has no events.
        """
        from utils import TrainingDataError, run_bootstrap

        rng = np.random.RandomState(0)
        n = 20
        X_test = rng.normal(size=(n, 2))
        y_test = _survival_y(np.zeros(n, dtype=bool), names=names)

        with pytest.raises(TrainingDataError) as exc_info:
            run_bootstrap(
                X_test,
                y_test,
                _RiskModel(),
                42,
                _survival_scoring(),
                n_bootstrap=100,
            )

        message = str(exc_info.value)
        lower = message.lower()
        assert "event" in lower
        assert "bootstrap" in lower
        assert "censored" in lower
        assert _mentions_larger_test_set(message), message
        assert "Traceback" not in message

    def test_survival_bootstrap_with_events_runs(self):
        """Regression: a test set with plenty of events bootstraps fine."""
        from utils import run_bootstrap

        rng = np.random.RandomState(1)
        n = 30
        events = np.zeros(n, dtype=bool)
        events[::2] = True  # 15 events
        times = rng.uniform(1, 100, size=n)
        y_test = _survival_y(events, times)
        X_test = rng.normal(size=(n, 2))

        scores, n_bootstrap = run_bootstrap(
            X_test, y_test, _RiskModel(), 42, _survival_scoring(), n_bootstrap=100
        )

        assert n_bootstrap == 100
        assert len(scores) == 100
        values = [s["c-index"] for s in scores]
        assert all(0.0 <= v <= 1.0 and not math.isnan(v) for v in values)

    def test_classification_bootstrap_unaffected(self):
        """Regression: classification test sets don't go through the check."""
        from utils import run_bootstrap

        rng = np.random.RandomState(2)
        n = 20
        X_test = rng.normal(size=(n, 2))
        y_test = np.array([0, 1] * 10)
        model = LogisticRegression().fit(X_test, y_test)

        scores, n_bootstrap = run_bootstrap(
            X_test, y_test, model, 42, {"auc": "roc_auc"}, n_bootstrap=20
        )

        assert n_bootstrap == 20
        assert len(scores) == 20
        assert all(0.0 <= s["auc"] <= 1.0 for s in scores)

    @pytest.mark.parametrize("names", FIELD_NAMES)
    def test_no_comparable_pairs_raises_training_data_error(self, names):
        """1 event at the latest time: no comparable pairs in any resample.

        sksurv raises NoComparablePairException; the user must get a clear
        message instead.
        """
        from utils import TrainingDataError, run_bootstrap

        X_test, y_test = _no_comparable_pairs_test_set(names)

        with pytest.raises(TrainingDataError) as exc_info:
            run_bootstrap(
                X_test, y_test, _RiskModel(), 42, _survival_scoring(), n_bootstrap=100
            )

        message = str(exc_info.value)
        lower = message.lower()
        assert "event" in lower
        assert "c-index" in lower or "concordance" in lower
        assert _mentions_larger_test_set(message), message
        assert "Traceback" not in message
        assert "no comparable pairs, cannot estimate" not in lower


# ---------------------------------------------------------------------------
# train_model (heavy work and I/O mocked)
# ---------------------------------------------------------------------------


class _GridSearchRecorder:
    def __init__(self, fit_error=None, estimator_step="analyzer"):
        self.calls = []
        self.fits = 0
        self.fit_error = fit_error
        self.estimator_step = estimator_step

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        scoring = kwargs.get("scoring")
        fitted = MagicMock(name="fitted_grid")
        fitted.cv_results_ = {
            "mean_test_c-index": np.array([0.7]),
            f"mean_test_{FAKE_SCORER_KEY}": np.array([0.0]),
        }
        fitted.scoring = scoring
        fitted.scorer_ = dict(scoring)
        fitted.best_index_ = 0
        fitted.best_params_ = {
            self.estimator_step: LogisticRegression(),
            "preprocessor": StandardScaler(),
        }
        fitted.best_estimator_ = MagicMock(name="best_estimator")

        recorder = self

        def _fit(*a, **k):
            recorder.fits += 1
            if recorder.fit_error is not None:
                raise recorder.fit_error
            return fitted

        grid = MagicMock(name="grid")
        grid.fit.side_effect = _fit
        return grid


def _label_category(label_type):
    lc = MagicMock(name="label_category")
    lc.label_type = label_type
    lc.id = 9
    return lc


def _survival_kwargs(n_test=13, test_events=0, names=("Event", "Time"), **over):
    cv = MagicMock(name="cv")
    cv.cvargs = {"n_splits": 5}
    cv.n_repeats = 2
    n_train = 20
    train_events = np.zeros(n_train, dtype=bool)
    train_events[::2] = True
    test_events_arr = np.zeros(n_test, dtype=bool)
    test_events_arr[:test_events] = True
    kwargs = dict(
        feature_extraction_id=1,
        collection_id=None,
        album={"album_id": "album-1", "name": "Album"},
        feature_selection=None,
        feature_names=["f1", "f2"],
        pipeline=MagicMock(name="pipeline"),
        parameter_grid=[{}],
        estimator_step="analyzer",
        scoring=_survival_scoring(),
        refit_metric="c-index",
        cv=cv,
        n_jobs=1,
        X_train=pd.DataFrame(
            {"f1": np.arange(n_train, dtype=float), "f2": np.ones(n_train)}
        ),
        X_test=pd.DataFrame(
            {"f1": np.arange(n_test, dtype=float), "f2": np.ones(n_test)}
        ),
        label_category=_label_category("Survival"),
        data_splitting_type="trainTestSplit",
        train_test_splitting_type="automatic",
        training_patients=[f"tr{i}" for i in range(n_train)],
        test_patients=[f"te{i}" for i in range(n_test)],
        y_train_encoded=_survival_y(train_events, names=names),
        y_test_encoded=_survival_y(test_events_arr, names=names),
        is_train_test=True,
        random_seed=42,
        training_id=321,
        user_id="user-1",
    )
    kwargs.update(over)
    return kwargs


@pytest.fixture
def train_env(monkeypatch, tmp_path):
    import tasks

    grid = _GridSearchRecorder()
    mocks = {
        "db": MagicMock(name="db"),
        "socketio": MagicMock(name="socketio"),
        "SocketIO": MagicMock(name="SocketIO"),
        "run_bootstrap": MagicMock(
            name="run_bootstrap", return_value=([{"c-index": 0.7}], 100)
        ),
        "calculate_training_metrics": MagicMock(
            name="calculate_training_metrics",
            return_value={"c-index": {"mean": 0.7}},
        ),
        "calculate_test_metrics": MagicMock(
            name="calculate_test_metrics",
            return_value=({"c-index": {"mean": 0.7}}, {"c-index": [0.7]}),
        ),
        "compute_feature_importance": MagicMock(
            name="compute_feature_importance", return_value={"f1": 0.1}
        ),
        "compute_predictions": MagicMock(
            name="compute_predictions", return_value=({}, {})
        ),
        "compute_risk_scores": MagicMock(name="compute_risk_scores", return_value={}),
        "get_model_path": MagicMock(
            name="get_model_path", return_value=str(tmp_path / "models" / "m.joblib")
        ),
        "joblib": MagicMock(name="joblib"),
        "Model": MagicMock(name="Model"),
        "format_model": MagicMock(name="format_model", return_value={"id": 1}),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(tasks, name, mock, raising=False)
    monkeypatch.setattr(tasks, "GridSearchCV", grid)
    # validate_metrics_no_nan is deliberately NOT mocked: it is part of the
    # behavior under test (it raises TrainingDataError on NaN metrics).
    mocks["grid"] = grid
    mocks["tasks"] = tasks
    return mocks


def _training_status_bodies(socketio_mock):
    bodies = []
    for call in socketio_mock.emit.call_args_list:
        args, kwargs = call.args, call.kwargs
        event = args[0] if args else kwargs.get("event")
        body = args[1] if len(args) > 1 else kwargs.get("data")
        if event == TRAINING_STATUS:
            bodies.append(body)
    return bodies


def _final_body(env):
    bodies = _training_status_bodies(env["socketio"])
    assert bodies, "no training-status emitted on tasks.socketio"
    return bodies[-1]


def _assert_nothing_saved(env):
    env["Model"].assert_not_called()
    env["Model"].return_value.save_to_db.assert_not_called()
    env["joblib"].dump.assert_not_called()


@requires_okapy
class TestTrainModelSurvivalTestSetWithoutEvents:
    @pytest.mark.parametrize("names", FIELD_NAMES)
    def test_fails_fast_before_grid_search(self, train_env, names):
        train_env["tasks"].train_model.run(
            **_survival_kwargs(n_test=13, test_events=0, names=names)
        )

        grid = train_env["grid"]
        assert grid.calls == [], "GridSearchCV must not be built"
        assert grid.fits == 0
        _assert_nothing_saved(train_env)
        train_env["run_bootstrap"].assert_not_called()

    def test_reports_clear_message_without_traceback(self, train_env):
        train_env["tasks"].train_model.run(
            **_survival_kwargs(n_test=13, test_events=0, training_id=321)
        )

        body = _final_body(train_env)
        assert set(body) == {"training-id", "failed", "error"}
        assert body["training-id"] == 321
        assert body["failed"] is True

        error = body["error"]
        assert isinstance(error, str)
        assert "Traceback" not in error
        lower = error.lower()
        assert "event" in lower or "censored" in lower
        assert "test" in lower
        assert re.search(r"\b13\b", error), error
        assert _mentions_larger_test_set(error) or "split" in lower, error
        assert "All samples are censored" not in error
        assert not any(
            b.get("complete") for b in _training_status_bodies(train_env["socketio"])
        )

    def test_error_is_exactly_the_training_data_error_message(
        self, train_env, monkeypatch
    ):
        """The emitted error is str(exc) of the TrainingDataError raised."""
        import utils

        raised = []
        original = utils.TrainingDataError

        class _Spy(original):
            def __init__(self, *args):
                super().__init__(*args)
                raised.append(self)

        tasks = train_env["tasks"]
        monkeypatch.setattr(utils, "TrainingDataError", _Spy)
        if hasattr(tasks, "TrainingDataError"):
            monkeypatch.setattr(tasks, "TrainingDataError", _Spy)

        tasks.train_model.run(**_survival_kwargs(n_test=13, test_events=0))

        body = _final_body(train_env)
        assert raised, "fail-fast check must raise utils.TrainingDataError"
        assert body["error"] == str(raised[-1])
        assert body["failed"] is True
        assert "Traceback" not in body["error"]

    def test_test_set_with_one_event_proceeds(self, train_env):
        train_env["tasks"].train_model.run(**_survival_kwargs(n_test=13, test_events=1))

        assert len(train_env["grid"].calls) == 1
        assert train_env["grid"].fits == 1
        body = _final_body(train_env)
        assert body.get("complete") is True
        train_env["Model"].assert_called_once()

    def test_survival_without_train_test_split_not_checked(self, train_env):
        """Cross-validation only: there is no test set to check."""
        train_env["tasks"].train_model.run(
            **_survival_kwargs(
                is_train_test=False,
                X_test=None,
                y_test_encoded=None,
                test_patients=None,
            )
        )

        assert len(train_env["grid"].calls) == 1
        assert _final_body(train_env).get("complete") is True

    def test_classification_with_single_class_test_set_not_checked(self, train_env):
        """Classification trainings are unaffected by the survival check."""
        grid = _GridSearchRecorder(estimator_step="classifier")
        train_env["tasks"].GridSearchCV = grid  # restored by monkeypatch

        train_env["tasks"].train_model.run(
            **_survival_kwargs(
                estimator_step="classifier",
                scoring={"auc": "roc_auc"},
                refit_metric="auc",
                label_category=_label_category("Classification"),
                y_train_encoded=np.array([0, 1] * 10),
                y_test_encoded=np.zeros(13, dtype=int),
            )
        )

        assert len(grid.calls) == 1
        assert _final_body(train_env).get("complete") is True


@requires_okapy
class TestTrainModelErrorReporting:
    def test_training_data_error_reported_without_traceback(self, train_env):
        from utils import TrainingDataError

        message = "The test set is too small: please use a larger test set."
        grid = _GridSearchRecorder(fit_error=TrainingDataError(message))
        train_env["tasks"].GridSearchCV = grid  # restored by monkeypatch

        train_env["tasks"].train_model.run(
            **_survival_kwargs(test_events=3, training_id=88)
        )

        body = _final_body(train_env)
        assert body == {"training-id": 88, "failed": True, "error": message}
        _assert_nothing_saved(train_env)

    def test_nan_metrics_reported_as_user_message(self, train_env):
        """validate_metrics_no_nan's TrainingDataError reaches the UI as-is."""
        train_env["calculate_test_metrics"].return_value = (
            {"c-index": {"mean": float("nan")}},
            {"c-index": [float("nan")]},
        )

        train_env["tasks"].train_model.run(
            **_survival_kwargs(n_test=13, test_events=2, training_id=89)
        )

        body = _final_body(train_env)
        assert body["training-id"] == 89
        assert body["failed"] is True
        assert "Traceback" not in body["error"]
        assert "current size: 13 patients" in body["error"]
        assert "Please increase the test set size" in body["error"]
        _assert_nothing_saved(train_env)

    def test_other_errors_keep_full_traceback(self, train_env):
        """Unexpected errors still ship the traceback for debugging."""
        grid = _GridSearchRecorder(fit_error=RuntimeError("solver exploded"))
        train_env["tasks"].GridSearchCV = grid  # restored by monkeypatch

        train_env["tasks"].train_model.run(
            **_survival_kwargs(test_events=3, training_id=90)
        )

        body = _final_body(train_env)
        assert body["training-id"] == 90
        assert body["failed"] is True
        assert "Traceback" in body["error"]
        assert "RuntimeError: solver exploded" in body["error"]

    def test_plain_value_error_keeps_full_traceback(self, train_env):
        """Only TrainingDataError is user-facing, not every ValueError."""
        grid = _GridSearchRecorder(fit_error=ValueError("internal bug"))
        train_env["tasks"].GridSearchCV = grid  # restored by monkeypatch

        train_env["tasks"].train_model.run(
            **_survival_kwargs(test_events=3, training_id=91)
        )

        body = _final_body(train_env)
        assert body["failed"] is True
        assert "Traceback" in body["error"]
        assert "ValueError: internal bug" in body["error"]

    def test_no_comparable_pairs_in_bootstrap_reported_without_traceback(
        self, train_env, monkeypatch
    ):
        """End to end: real run_bootstrap on a test set with 1 event that
        has no comparable pairs (fail-fast does not fire)."""
        import utils

        tasks = train_env["tasks"]
        monkeypatch.setattr(tasks, "run_bootstrap", utils.run_bootstrap)
        X_test, y_test = _no_comparable_pairs_test_set()

        original_call = train_env["grid"].__call__

        def _grid_with_risk_model(*args, **kwargs):
            grid = original_call(*args, **kwargs)
            fit = grid.fit.side_effect

            def _fit(*a, **k):
                fitted = fit(*a, **k)
                fitted.predict.side_effect = lambda X: np.asarray(X, dtype=float)[:, 0]
                fitted.predict_proba.side_effect = AttributeError("no proba")
                return fitted

            grid.fit.side_effect = _fit
            return grid

        monkeypatch.setattr(tasks, "GridSearchCV", _grid_with_risk_model)

        tasks.train_model.run(
            **_survival_kwargs(
                n_test=10,
                X_test=pd.DataFrame(X_test, columns=["f1", "f2"]),
                y_test_encoded=y_test,
                training_id=92,
            )
        )

        assert train_env["grid"].fits == 1, "fail-fast must not fire (1 event)"
        body = _final_body(train_env)
        assert body["training-id"] == 92
        assert body["failed"] is True
        assert "Traceback" not in body["error"]
        assert "event" in body["error"].lower()
        _assert_nothing_saved(train_env)

"""
Audit tests for issue #54 (training redelivery and CPU oversubscription).

These cover behavior the frozen tests in ``test_training_concurrency.py`` and
``test_celery_tasks.py`` do not reach because they mock ``GridSearchCV`` and
the Socket.IO client:

- the progress scorer running for real inside loky worker processes
- the soft time limit firing inside an in-process (``n_jobs=1``) grid search,
  where scikit-learn's ``error_score`` handling can swallow it
- reserved (prefetched) ``acks_late=False`` training messages, which stay
  unacknowledged in Redis while they wait for a free slot
- a CPU budget that cannot be resolved must still be reported to the user
"""

import importlib.util
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantimage2_backend_common.const import FAKE_SCORER_KEY
from quantimage2_backend_common.utils import MessageType

requires_okapy = pytest.mark.skipif(
    not importlib.util.find_spec("okapy"),
    reason="okapy not available (workers-only dependency)",
)

TRAINING_STATUS = MessageType.TRAINING_STATUS.value


def _training_status_bodies(socketio_mock):
    bodies = []
    for call in socketio_mock.emit.call_args_list:
        args, kwargs = call.args, call.kwargs
        event = args[0] if args else kwargs.get("event")
        body = args[1] if len(args) > 1 else kwargs.get("data")
        if event == TRAINING_STATUS:
            bodies.append(body)
    return bodies


def _set_machine(monkeypatch, *, cores, concurrency):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(cores)), raising=False
    )
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", str(concurrency))


# ---------------------------------------------------------------------------
# fake_score inside real loky worker processes
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_loky_workers(monkeypatch):
    """Loky workers spawned with an in-memory Socket.IO message queue.

    Workers inherit the environment at spawn time, so any reusable executor
    from an earlier test is shut down first.
    """
    from joblib.externals.loky import get_reusable_executor

    monkeypatch.setenv("SOCKET_MESSAGE_QUEUE", "memory://")
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    yield
    get_reusable_executor().shutdown(wait=True, kill_workers=True)


class TestFakeScoreInLokyWorkers:
    def test_grid_search_with_progress_scorer_runs_in_parallel_workers(
        self, fresh_loky_workers
    ):
        """The scorer must unpickle and emit inside loky workers, with no
        Flask app context and a real (non-mocked) Socket.IO client.

        ``error_score="raise"`` makes a scorer failure visible: train_model
        uses the default ``np.nan``, which would hide it behind a warning and
        silently drop every progress event.
        """
        import utils

        rng = np.random.RandomState(0)
        X = rng.normal(size=(40, 3))
        y = np.array([0, 1] * 20)
        grid = GridSearchCV(
            DummyClassifier(),
            {"strategy": ["most_frequent", "prior"]},
            scoring={
                "accuracy": "accuracy",
                FAKE_SCORER_KEY: make_scorer(utils.fake_score, training_id=7),
            },
            refit="accuracy",
            cv=StratifiedKFold(n_splits=2),
            n_jobs=2,
            error_score="raise",
        )

        fitted = grid.fit(X, y)

        assert np.all(fitted.cv_results_[f"mean_test_{FAKE_SCORER_KEY}"] == 0)

    def test_fake_score_with_real_client_outside_app_context(self, monkeypatch):
        """No Flask app context exists in the joblib workers any more."""
        import flask
        import utils

        monkeypatch.setenv("SOCKET_MESSAGE_QUEUE", "memory://")
        monkeypatch.setattr(utils, "_socketio", None)
        assert not flask.has_app_context()

        assert utils.fake_score([0, 1], [0, 1], training_id=1) == 0
        client = utils._socketio
        assert client is not None
        assert utils.fake_score([0, 1], [0, 1], training_id=2) == 0
        assert utils._socketio is client


# ---------------------------------------------------------------------------
# train_model with a real, in-process grid search
# ---------------------------------------------------------------------------


class _SoftLimitOnFirstFit(BaseEstimator, ClassifierMixin):
    """Raises SoftTimeLimitExceeded on the first fit, like the SIGUSR1 handler
    billiard installs in the pool process does when the soft limit fires."""

    fired = False

    def fit(self, X, y):
        from celery.exceptions import SoftTimeLimitExceeded

        if not type(self).fired:
            type(self).fired = True
            raise SoftTimeLimitExceeded()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, 0] = 1.0
        return proba


@pytest.fixture
def real_grid_training(monkeypatch, tmp_path):
    """train_model with a real GridSearchCV; everything after it is mocked."""
    import tasks
    import utils

    monkeypatch.setenv("SOCKET_MESSAGE_QUEUE", "memory://")
    monkeypatch.setattr(utils, "_socketio", None)
    monkeypatch.setattr(utils, "SocketIO", MagicMock(name="utils.SocketIO"))
    _SoftLimitOnFirstFit.fired = False

    mocks = {
        "db": MagicMock(name="db"),
        "socketio": MagicMock(name="socketio"),
        "SocketIO": MagicMock(name="SocketIO"),
        "calculate_training_metrics": MagicMock(return_value={"auc": {}}),
        "compute_predictions": MagicMock(return_value=({}, {})),
        "get_model_path": MagicMock(return_value=str(tmp_path / "m" / "m.joblib")),
        "joblib": MagicMock(name="joblib"),
        "Model": MagicMock(name="Model"),
        "format_model": MagicMock(return_value={"id": 1}),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(tasks, name, mock, raising=False)
    mocks["tasks"] = tasks
    return mocks


def _real_grid_kwargs(n_jobs):
    cv = StratifiedKFold(n_splits=2)
    cv.cvargs = {"n_splits": 2}
    cv.n_repeats = 1
    label_category = MagicMock(label_type="Classification", id=5)
    X_train = pd.DataFrame(
        {"f1": np.arange(8, dtype=float), "f2": np.arange(8, dtype=float) % 3}
    )
    return dict(
        feature_extraction_id=1,
        collection_id=None,
        album={"album_id": "album-1", "name": "Album"},
        feature_selection=None,
        feature_names=["f1", "f2"],
        pipeline=Pipeline(
            [
                ("preprocessor", StandardScaler()),
                ("classifier", _SoftLimitOnFirstFit()),
            ]
        ),
        parameter_grid=[
            {"preprocessor": [StandardScaler()], "classifier": [_SoftLimitOnFirstFit()]}
        ],
        estimator_step="classifier",
        scoring={"auc": "roc_auc"},
        refit_metric="auc",
        cv=cv,
        n_jobs=n_jobs,
        X_train=X_train,
        X_test=None,
        label_category=label_category,
        data_splitting_type="fullDataset",
        train_test_splitting_type=None,
        training_patients=[f"p{i}" for i in range(8)],
        test_patients=None,
        y_train_encoded=np.array([0, 1] * 4),
        y_test_encoded=None,
        is_train_test=False,
        random_seed=42,
        training_id=888,
        user_id="user-1",
    )


@requires_okapy
class TestSoftTimeLimitInProcessGridSearch:
    def test_soft_limit_during_sequential_fit_reports_failure(self, real_grid_training):
        """With n_jobs=1 the CV fits run in the Celery pool process itself,
        where the soft-limit signal is raised. GridSearchCV's default
        ``error_score=np.nan`` catches any Exception from fit, so the soft
        limit must not be swallowed: the training has to stop and report
        ``failed: True`` (otherwise it runs on until the hard SIGKILL, which
        emits nothing and leaves the UI waiting forever)."""
        tasks = real_grid_training["tasks"]

        tasks.train_model.run(**_real_grid_kwargs(n_jobs=1))

        assert _SoftLimitOnFirstFit.fired
        bodies = _training_status_bodies(real_grid_training["socketio"])
        assert bodies, "no training-status emitted"
        final = bodies[-1]
        assert final["training-id"] == 888
        assert final.get("failed") is True, "soft time limit was swallowed"
        assert "SoftTimeLimitExceeded" in final.get("error", "")
        real_grid_training["Model"].assert_not_called()
        real_grid_training["joblib"].dump.assert_not_called()


# ---------------------------------------------------------------------------
# Reserved (prefetched) training messages vs. the visibility timeout
# ---------------------------------------------------------------------------


@requires_okapy
class TestReservedTrainingMessages:
    def test_visibility_timeout_exceeds_train_hard_limit(self):
        """acks_late=False acknowledges a training when it *starts*. With
        prefetch_multiplier=1 the worker still reserves one extra training per
        slot, unacknowledged, until a slot frees up, i.e. for up to the hard
        time_limit of the trainings already running. If that wait exceeds the
        visibility timeout, Redis restores the reserved message and the same
        training runs twice."""
        import tasks

        if os.environ.get("CELERY_CONFIG_MODULE") != "celeryconfig":
            pytest.skip("celeryconfig not wired via CELERY_CONFIG_MODULE here")

        conf = tasks.celery.conf
        assert conf.worker_prefetch_multiplier >= 1
        assert tasks.train_model.acks_late is False
        visibility_timeout = conf.broker_transport_options["visibility_timeout"]

        assert visibility_timeout > tasks.train_model.time_limit


# ---------------------------------------------------------------------------
# CPU budget that cannot be resolved
# ---------------------------------------------------------------------------


class TestResolveNJobsEdgeCases:
    def test_zero_worker_concurrency_still_gives_a_budget(self, monkeypatch):
        """Celery treats worker_concurrency=0 as "one slot per CPU"; resolving
        the budget must not divide by zero."""
        from utils import resolve_n_jobs

        _set_machine(monkeypatch, cores=8, concurrency=0)

        result = resolve_n_jobs(-1)

        assert isinstance(result, int)
        assert result >= 1

    def test_numpy_int_request_is_an_explicit_override(self, monkeypatch):
        from utils import resolve_n_jobs

        _set_machine(monkeypatch, cores=40, concurrency=4)

        assert resolve_n_jobs(np.int64(3)) == 3
        assert resolve_n_jobs(np.int64(-1)) == 10


@requires_okapy
class TestTrainingReportsBudgetErrors:
    def test_unresolvable_budget_is_reported_as_failed_training(
        self, real_grid_training, monkeypatch
    ):
        """Anything that goes wrong before the grid search (here the CPU budget
        with CELERY_WORKER_CONCURRENCY=0) must still reach the user as a failed
        training instead of an exception with no training-status event."""
        tasks = real_grid_training["tasks"]
        _set_machine(monkeypatch, cores=8, concurrency=0)

        try:
            tasks.train_model.run(**_real_grid_kwargs(n_jobs=-1))
        except Exception:
            pass

        bodies = _training_status_bodies(real_grid_training["socketio"])
        assert bodies, "training ended without any training-status event"
        assert bodies[-1]["training-id"] == 888

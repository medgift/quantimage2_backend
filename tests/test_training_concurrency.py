"""
Tests for training concurrency and redelivery safety (issue #54).

Covers:
- ``utils.resolve_n_jobs``: CPU budget split across worker slots
- ``utils.fake_score``: module-level, picklable progress scorer with a cached
  Socket.IO client (no Flask app per scoring call)
- ``tasks.train_model``: time limits (env-configurable), CPU budget passed to
  grid search and feature importance, soft time limit handling
- env files shipping the new training worker concurrency
"""

import importlib
import importlib.util
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

from quantimage2_backend_common.const import FAKE_SCORER_KEY, TRAINING_PHASES
from quantimage2_backend_common.utils import MessageType

requires_okapy = pytest.mark.skipif(
    not importlib.util.find_spec("okapy"),
    reason="okapy not available (workers-only dependency)",
)

TRAINING_STATUS = MessageType.TRAINING_STATUS.value


def _emit_event_and_body(call):
    """Extract (event, body) from a SocketIO.emit mock call."""
    args, kwargs = call.args, call.kwargs
    event = args[0] if args else kwargs.get("event")
    body = args[1] if len(args) > 1 else kwargs.get("data")
    return event, body


# ---------------------------------------------------------------------------
# resolve_n_jobs
# ---------------------------------------------------------------------------


@pytest.fixture
def machine(monkeypatch):
    """Configure the visible CPU count and worker concurrency.

    ``affinity=None`` removes ``os.sched_getaffinity`` (non-Linux platforms)
    so the ``os.cpu_count`` fallback is exercised.
    """

    def _configure(*, affinity=None, cpu_count=None, concurrency=None):
        if affinity is None:
            monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        else:
            cores = set(range(affinity))
            monkeypatch.setattr(
                os, "sched_getaffinity", lambda pid: set(cores), raising=False
            )
        # A distinct cpu_count proves which source is used.
        monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
        if concurrency is None:
            monkeypatch.delenv("CELERY_WORKER_CONCURRENCY", raising=False)
        else:
            monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", str(concurrency))

    return _configure


class TestResolveNJobs:
    @pytest.mark.parametrize("requested", [-1, None, 0, -2])
    def test_unset_or_non_positive_request_gets_per_slot_budget(
        self, machine, requested
    ):
        from utils import resolve_n_jobs

        machine(affinity=40, cpu_count=999, concurrency=4)
        result = resolve_n_jobs(requested)

        assert result == 10
        assert isinstance(result, int)

    @pytest.mark.parametrize("requested", [1, 3, 10, 64])
    def test_positive_request_returned_unchanged(self, machine, requested):
        from utils import resolve_n_jobs

        machine(affinity=40, cpu_count=999, concurrency=4)

        assert resolve_n_jobs(requested) == requested

    def test_more_slots_than_cores_gives_at_least_one_job(self, machine):
        from utils import resolve_n_jobs

        machine(affinity=4, cpu_count=999, concurrency=8)

        assert resolve_n_jobs(-1) == 1

    def test_budget_rounds_down(self, machine):
        from utils import resolve_n_jobs

        machine(affinity=10, cpu_count=999, concurrency=3)

        assert resolve_n_jobs(-1) == 3

    def test_affinity_preferred_over_cpu_count(self, machine):
        """In a container limited by cpuset, the affinity mask is the truth."""
        from utils import resolve_n_jobs

        machine(affinity=8, cpu_count=64, concurrency=2)

        assert resolve_n_jobs(-1) == 4

    def test_falls_back_to_cpu_count_without_sched_getaffinity(self, machine):
        from utils import resolve_n_jobs

        machine(affinity=None, cpu_count=16, concurrency=4)
        assert not hasattr(os, "sched_getaffinity")

        assert resolve_n_jobs(-1) == 4

    def test_concurrency_unset_means_single_slot_gets_all_cores(self, machine):
        from utils import resolve_n_jobs

        machine(affinity=12, cpu_count=999, concurrency=None)

        assert resolve_n_jobs(-1) == 12
        assert resolve_n_jobs(None) == 12

    def test_concurrency_read_at_call_time(self, machine, monkeypatch):
        from utils import resolve_n_jobs

        machine(affinity=40, cpu_count=999, concurrency=4)
        assert resolve_n_jobs(-1) == 10

        monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "8")
        assert resolve_n_jobs(-1) == 5


# ---------------------------------------------------------------------------
# fake_score (progress scorer)
# ---------------------------------------------------------------------------


@pytest.fixture
def progress_socketio(monkeypatch):
    """Patch the Socket.IO client class used by ``utils.fake_score``.

    Resets the per-process client cache so each test starts cold, and makes
    sure no Flask app can be created behind our back.
    """
    import utils
    import quantimage2_backend_common.flask_init as flask_init

    monkeypatch.setenv("SOCKET_MESSAGE_QUEUE", "redis://socket-queue-test:6379/")
    monkeypatch.setattr(utils, "_socketio", None, raising=False)
    socketio_cls = MagicMock(name="SocketIO")
    # raising=True: utils must expose SocketIO in its namespace (contract)
    monkeypatch.setattr(utils, "SocketIO", socketio_cls)

    create_app_mock = MagicMock(name="create_app")
    monkeypatch.setattr(flask_init, "create_app", create_app_mock)
    utils_create_app = None
    if hasattr(utils, "create_app"):
        utils_create_app = MagicMock(name="utils.create_app")
        monkeypatch.setattr(utils, "create_app", utils_create_app)

    yield {
        "cls": socketio_cls,
        "client": socketio_cls.return_value,
        "create_app": create_app_mock,
        "utils_create_app": utils_create_app,
    }


class TestFakeScore:
    def test_fake_score_is_module_level_function(self):
        import utils

        assert callable(utils.fake_score)
        assert utils.fake_score.__qualname__ == "fake_score"
        assert utils.fake_score.__module__ == utils.__name__

    def test_emits_training_progress_and_returns_zero(self, progress_socketio):
        from utils import fake_score

        result = fake_score([0, 1], [0, 1], training_id=42)

        assert result == 0
        client = progress_socketio["client"]
        assert client.emit.call_count == 1
        event, body = _emit_event_and_body(client.emit.call_args)
        assert event == TRAINING_STATUS
        assert body == {
            "training-id": 42,
            "phase": TRAINING_PHASES.TRAINING.value,
        }

    def test_client_uses_socket_message_queue(self, progress_socketio):
        from utils import fake_score

        fake_score([0], [0], training_id=1)

        socketio_cls = progress_socketio["cls"]
        assert socketio_cls.call_count == 1
        assert socketio_cls.call_args.kwargs == {
            "message_queue": "redis://socket-queue-test:6379/"
        }

    def test_client_created_once_across_many_calls(self, progress_socketio):
        """Regression #54: one Socket.IO client per process, not per score."""
        from utils import fake_score

        for i in range(25):
            assert fake_score([0, 1], [1, 0], training_id=i) == 0

        assert progress_socketio["cls"].call_count == 1
        client = progress_socketio["client"]
        assert client.emit.call_count == 25
        bodies = [_emit_event_and_body(c)[1] for c in client.emit.call_args_list]
        assert [b["training-id"] for b in bodies] == list(range(25))

    def test_cache_is_module_level_and_resettable(self, progress_socketio):
        import utils

        utils.fake_score([0], [0], training_id=1)
        assert utils._socketio is progress_socketio["client"]

        # A fresh process (cache reset) builds its own client
        utils._socketio = None
        utils.fake_score([0], [0], training_id=1)
        assert progress_socketio["cls"].call_count == 2

    def test_never_creates_flask_app(self, progress_socketio):
        """Regression #54: the old scorer built a Flask app on every call."""
        from utils import fake_score

        for _ in range(5):
            fake_score([0, 1], [0, 1], training_id=3)

        progress_socketio["create_app"].assert_not_called()
        if progress_socketio["utils_create_app"] is not None:
            progress_socketio["utils_create_app"].assert_not_called()

    def test_default_training_id_is_none(self, progress_socketio):
        from utils import fake_score

        fake_score([0], [0])

        _, body = _emit_event_and_body(progress_socketio["client"].emit.call_args)
        assert body == {"training-id": None, "phase": TRAINING_PHASES.TRAINING.value}

    def test_scorer_wrapping_fake_score_is_picklable(self):
        """GridSearchCV workers (loky) must be able to pickle the scorer."""
        import utils

        scorer = make_scorer(utils.fake_score, training_id=1234)

        restored = pickle.loads(pickle.dumps(scorer))

        assert restored._score_func is utils.fake_score
        assert restored._kwargs == {"training_id": 1234}

    def test_scorer_wrapping_fake_score_reports_progress(self, progress_socketio):
        """Used as a sklearn scorer, it emits progress and scores 0."""
        import utils

        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0, 1])
        estimator = DummyClassifier(strategy="most_frequent").fit(X, y)
        scorer = pickle.loads(
            pickle.dumps(make_scorer(utils.fake_score, training_id=99))
        )

        assert scorer(estimator, X, y) == 0

        event, body = _emit_event_and_body(progress_socketio["client"].emit.call_args)
        assert event == TRAINING_STATUS
        assert body == {"training-id": 99, "phase": TRAINING_PHASES.TRAINING.value}


# ---------------------------------------------------------------------------
# train_model: time limits
# ---------------------------------------------------------------------------

_LIMIT_ENV = ("TRAINING_SOFT_TIME_LIMIT", "TRAINING_TIME_LIMIT")


@pytest.fixture
def reload_tasks(monkeypatch):
    """Re-import ``tasks`` under a given environment, then restore it."""
    import tasks

    def _reload(**env):
        for key in _LIMIT_ENV:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(tasks)

    try:
        yield _reload
    finally:
        monkeypatch.undo()
        importlib.reload(tasks)


@requires_okapy
class TestTrainTimeLimits:
    def test_default_time_limits(self, reload_tasks):
        tasks = reload_tasks()

        assert tasks.train_model.soft_time_limit == 21600
        assert tasks.train_model.time_limit == 22200

    def test_soft_limit_from_env_hard_defaults_to_soft_plus_600(self, reload_tasks):
        tasks = reload_tasks(TRAINING_SOFT_TIME_LIMIT="3600")

        assert tasks.train_model.soft_time_limit == 3600
        assert tasks.train_model.time_limit == 4200

    def test_both_limits_from_env(self, reload_tasks):
        tasks = reload_tasks(
            TRAINING_SOFT_TIME_LIMIT="3600", TRAINING_TIME_LIMIT="5000"
        )

        assert tasks.train_model.soft_time_limit == 3600
        assert tasks.train_model.time_limit == 5000

    def test_env_limits_do_not_affect_extraction(self, reload_tasks):
        tasks = reload_tasks(TRAINING_SOFT_TIME_LIMIT="100", TRAINING_TIME_LIMIT="200")

        assert tasks.run_extraction.time_limit == 7200
        assert tasks.run_extraction.soft_time_limit == 6600

    def test_train_does_not_ack_late_whatever_the_limits(self, reload_tasks):
        tasks = reload_tasks(TRAINING_SOFT_TIME_LIMIT="3600")

        assert tasks.train_model.acks_late is False


# ---------------------------------------------------------------------------
# train_model: execution (heavy work and I/O mocked)
# ---------------------------------------------------------------------------


class _GridSearchRecorder:
    """Stand-in for GridSearchCV that records how it was configured."""

    def __init__(self, fit_error=None):
        self.calls = []
        self.fake_scorers = []
        self.fit_error = fit_error

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        scoring = kwargs.get("scoring")
        self.fake_scorers.append(
            scoring.get(FAKE_SCORER_KEY) if isinstance(scoring, dict) else None
        )

        fitted = MagicMock(name="fitted_grid")
        fitted.cv_results_ = {
            "mean_test_auc": np.array([0.8]),
            f"mean_test_{FAKE_SCORER_KEY}": np.array([0.0]),
        }
        # Like sklearn: the fitted search keeps the very dict it was given
        fitted.scoring = scoring
        fitted.scorer_ = dict(scoring)
        fitted.best_index_ = 0
        fitted.best_params_ = {
            "classifier": LogisticRegression(),
            "preprocessor": StandardScaler(),
        }
        fitted.best_estimator_ = MagicMock(name="best_estimator")

        grid = MagicMock(name="grid")
        if self.fit_error is not None:
            grid.fit.side_effect = self.fit_error
        else:
            grid.fit.return_value = fitted
        return grid


def _train_kwargs(**overrides):
    cv = MagicMock(name="cv")
    cv.cvargs = {"n_splits": 5}
    cv.n_repeats = 2
    label_category = MagicMock(name="label_category")
    label_category.label_type = "Classification"
    label_category.id = 5

    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.0, 1.0, 0.0, 1.0]})
    X_test = pd.DataFrame({"f1": [5.0, 6.0], "f2": [1.0, 0.0]})
    kwargs = dict(
        feature_extraction_id=1,
        collection_id=None,
        album={"album_id": "album-1", "name": "Album"},
        feature_selection=None,
        feature_names=["f1", "f2"],
        pipeline=MagicMock(name="pipeline"),
        parameter_grid=[{}],
        estimator_step="classifier",
        scoring={"auc": "roc_auc"},
        refit_metric="auc",
        cv=cv,
        n_jobs=-1,
        X_train=X_train,
        X_test=X_test,
        label_category=label_category,
        data_splitting_type="trainTestSplit",
        train_test_splitting_type="automatic",
        training_patients=["p1", "p2", "p3", "p4"],
        test_patients=["p5", "p6"],
        y_train_encoded=np.array([0, 1, 0, 1]),
        y_test_encoded=np.array([1, 0]),
        is_train_test=True,
        random_seed=42,
        training_id=777,
        user_id="user-1",
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def training_env(monkeypatch, tmp_path, machine):
    """Mock every external boundary of ``train_model``.

    40 visible cores, 4 worker slots -> per-slot budget of 10 jobs.
    """
    import tasks

    machine(affinity=40, cpu_count=999, concurrency=4)

    grid = _GridSearchRecorder()
    mocks = {"grid": grid}
    targets = {
        "db": MagicMock(name="db"),
        "socketio": MagicMock(name="socketio"),
        "SocketIO": MagicMock(name="SocketIO"),
        "run_bootstrap": MagicMock(
            name="run_bootstrap", return_value=({"auc": [0.8]}, 100)
        ),
        "calculate_training_metrics": MagicMock(
            name="calculate_training_metrics", return_value={"auc": {"mean": 0.8}}
        ),
        "calculate_test_metrics": MagicMock(
            name="calculate_test_metrics",
            return_value=({"auc": {"mean": 0.7}}, {"auc": [0.7]}),
        ),
        "compute_feature_importance": MagicMock(
            name="compute_feature_importance", return_value={"f1": 0.1, "f2": 0.2}
        ),
        "compute_predictions": MagicMock(
            name="compute_predictions",
            return_value=({"p5": 1, "p6": 0}, {"p5": [0.2, 0.8], "p6": [0.9, 0.1]}),
        ),
        "validate_metrics_no_nan": MagicMock(name="validate_metrics_no_nan"),
        "get_model_path": MagicMock(
            name="get_model_path", return_value=str(tmp_path / "models" / "m.joblib")
        ),
        "joblib": MagicMock(name="joblib"),
        "Model": MagicMock(name="Model"),
        "format_model": MagicMock(name="format_model", return_value={"id": 1}),
    }
    monkeypatch.setattr(tasks, "GridSearchCV", grid)
    for name, mock in targets.items():
        monkeypatch.setattr(tasks, name, mock, raising=False)
        mocks[name] = mock
    mocks["tasks"] = tasks
    return mocks


def _run_training(tasks, **overrides):
    return tasks.train_model.run(**_train_kwargs(**overrides))


def _training_status_bodies(socketio_mock):
    bodies = []
    for call in socketio_mock.emit.call_args_list:
        event, body = _emit_event_and_body(call)
        if event == TRAINING_STATUS:
            bodies.append(body)
    return bodies


@requires_okapy
class TestTrainModelExecution:
    def test_grid_search_gets_per_slot_budget_not_all_cores(self, training_env):
        """Regression #54: n_jobs=-1 made every concurrent training use all
        cores. The grid search must get cores // worker slots."""
        _run_training(training_env["tasks"], n_jobs=-1)

        grid = training_env["grid"]
        assert len(grid.calls) == 1
        assert grid.calls[0]["n_jobs"] == 10

    def test_feature_importance_gets_per_slot_budget(self, training_env):
        """Regression #54: feature importance ran on a single core."""
        _run_training(training_env["tasks"], n_jobs=-1)

        cfi = training_env["compute_feature_importance"]
        assert cfi.call_count == 1
        assert cfi.call_args.kwargs.get("n_jobs") == 10

    def test_explicit_positive_n_jobs_is_used_as_is(self, training_env):
        _run_training(training_env["tasks"], n_jobs=3)

        assert training_env["grid"].calls[0]["n_jobs"] == 3
        assert (
            training_env["compute_feature_importance"].call_args.kwargs.get("n_jobs")
            == 3
        )

    def test_none_n_jobs_gets_per_slot_budget(self, training_env):
        _run_training(training_env["tasks"], n_jobs=None)

        assert training_env["grid"].calls[0]["n_jobs"] == 10
        assert (
            training_env["compute_feature_importance"].call_args.kwargs.get("n_jobs")
            == 10
        )

    def test_progress_scorer_wraps_module_level_fake_score(self, training_env):
        import utils

        _run_training(training_env["tasks"], training_id=4242)

        scorer = training_env["grid"].fake_scorers[0]
        assert scorer is not None, "progress scorer missing from scoring dict"
        assert scorer._score_func is utils.fake_score
        assert scorer._kwargs == {"training_id": 4242}
        # And it can travel to joblib/loky workers
        restored = pickle.loads(pickle.dumps(scorer))
        assert restored._score_func is utils.fake_score

    def test_user_scoring_kept_alongside_progress_scorer(self, training_env):
        _run_training(training_env["tasks"], scoring={"auc": "roc_auc"})

        scoring = training_env["grid"].calls[0]["scoring"]
        assert scoring.get("auc") == "roc_auc"
        assert training_env["grid"].fake_scorers[0] is not None

    def test_success_emits_complete_and_saves_model(self, training_env):
        _run_training(training_env["tasks"], training_id=777)

        bodies = _training_status_bodies(training_env["socketio"])
        assert bodies, "no training-status emitted on tasks.socketio"
        final = bodies[-1]
        assert final.get("complete") is True
        assert final["training-id"] == 777
        assert final.get("model") == {"id": 1}
        assert "failed" not in final

        training_env["Model"].assert_called_once()
        training_env["Model"].return_value.save_to_db.assert_called_once()
        training_env["joblib"].dump.assert_called_once()

    def test_soft_time_limit_in_grid_search_reports_failure(self, training_env):
        from celery.exceptions import SoftTimeLimitExceeded

        tasks = training_env["tasks"]
        grid = _GridSearchRecorder(fit_error=SoftTimeLimitExceeded())
        with patch.object(tasks, "GridSearchCV", grid):
            # Must not raise
            _run_training(tasks, training_id=555)

        bodies = _training_status_bodies(training_env["socketio"])
        assert bodies, "no training-status emitted on tasks.socketio"
        final = bodies[-1]
        assert final["training-id"] == 555
        assert final.get("failed") is True
        assert not any(b.get("complete") for b in bodies)

        training_env["Model"].assert_not_called()
        training_env["Model"].return_value.save_to_db.assert_not_called()
        training_env["joblib"].dump.assert_not_called()

    def test_soft_time_limit_in_feature_importance_saves_nothing(self, training_env):
        from celery.exceptions import SoftTimeLimitExceeded

        training_env["compute_feature_importance"].side_effect = SoftTimeLimitExceeded()

        _run_training(training_env["tasks"], training_id=556)

        final = _training_status_bodies(training_env["socketio"])[-1]
        assert final["training-id"] == 556
        assert final.get("failed") is True
        training_env["Model"].assert_not_called()
        training_env["joblib"].dump.assert_not_called()


# ---------------------------------------------------------------------------
# Deployment env files
# ---------------------------------------------------------------------------

_ENV_FILES_DIR = Path(__file__).resolve().parent.parent / "env_files"


def _read_env_file(name):
    path = _ENV_FILES_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} not available in this environment")
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class TestEnvFiles:
    def test_training_worker_runs_two_slots(self):
        values = _read_env_file("workers-training.env")
        assert values.get("CELERY_WORKER_CONCURRENCY") == "2"

    def test_webapp_still_requests_all_cores_per_grid_search(self):
        """The webapp keeps -1; the worker turns it into a per-slot budget."""
        values = _read_env_file("webapp.env")
        assert values.get("GRID_SEARCH_CONCURRENCY") == "-1"

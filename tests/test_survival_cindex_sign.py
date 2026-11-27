"""
Tests for issue #57: the survival c-index must be computed on *risk scores*.

sksurv's ``IPCRidge`` predicts (log) survival time -- higher means longer
survival, i.e. lower risk -- and marks itself with ``_predict_risk_score =
False``. Our c-index scorer used the raw prediction as a risk score, so every
IPCRidge c-index came out as ~(1 - real value). sksurv's own ``score()``
negates the prediction for such estimators; our scorer must agree with it.

Covers:
- ``modeling_utils.predicts_risk_score`` / ``to_risk_scores`` / ``CIndexScorer``
- ``Survival.get_scoring()`` (grid search, incl. pickle round-trip and the fake
  progress scorer used by ``tasks.train_model``)
- ``utils.run_bootstrap`` (test-set metrics)
- ``utils.compute_feature_importance`` (permutation importance)
- ``utils.compute_risk_scores`` (output must not change)
- Regressions: ``NoComparablePairException`` propagation, ``calculate_scores``
  on already-converted predictions, old pickled ``make_scorer(c_index_score)``

``modeling.survival`` is a webapp module (not importable in the
celery_training image); tests that need it skip there. ``utils`` is the
workers module; it is importable in both containers under pytest.
"""

import importlib.util
import pickle

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sksurv.exceptions import NoComparablePairException
from sksurv.linear_model import (
    CoxnetSurvivalAnalysis,
    CoxPHSurvivalAnalysis,
    IPCRidge,
)
from sksurv.util import Surv

from quantimage2_backend_common.const import FAKE_SCORER_KEY
from quantimage2_backend_common.modeling_utils import (
    SurvivalRepeatedStratifiedKFold,
    c_index_score,
)

pytestmark = pytest.mark.ml

_HAS_SURVIVAL_MODULE = importlib.util.find_spec("modeling") is not None and (
    importlib.util.find_spec("modeling.survival") is not None
)

requires_survival_module = pytest.mark.skipif(
    not _HAS_SURVIVAL_MODULE,
    reason="webapp modeling.survival not importable here (webapp-only module)",
)

ANALYZERS = {
    "IPCRidge": IPCRidge,
    "CoxnetSurvivalAnalysis": CoxnetSurvivalAnalysis,
    "CoxPHSurvivalAnalysis": CoxPHSurvivalAnalysis,
}
COX_ANALYZERS = ["CoxnetSurvivalAnalysis", "CoxPHSurvivalAnalysis"]


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _signal_data(n=200, n_features=5, seed=0):
    """Data from the issue: feature 0 strongly increases the hazard."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    t = rng.exponential(np.exp(-X[:, 0]))
    c = rng.uniform(0, 3, n)
    y = Surv.from_arrays(t <= c, np.minimum(t, c), name_event="Event", name_time="Time")
    return X, y


def _pipeline(analyzer_name):
    return Pipeline(
        [("preprocessor", StandardScaler()), ("analyzer", ANALYZERS[analyzer_name]())]
    )


def _fitted_pipeline(analyzer_name, X, y):
    return _pipeline(analyzer_name).fit(X, y)


def _survival_scoring():
    """The production scoring dict, from ``Survival.get_scoring()``."""
    from modeling.survival import Survival

    # get_scoring() doesn't depend on instance state; avoid the heavy __init__
    return Survival.__new__(Survival).get_scoring()


def _c_index_scorer(analyzer_name):
    """Production scorer when the webapp module is available, else the new
    shared scorer (worker image). For Cox models (whose behaviour must not
    change) the legacy scorer is an acceptable stand-in if the new one is not
    importable."""
    if _HAS_SURVIVAL_MODULE:
        return _survival_scoring()["c-index"]
    try:
        from quantimage2_backend_common.modeling_utils import CIndexScorer
    except ImportError:
        if analyzer_name == "IPCRidge":
            raise
        return make_scorer(c_index_score)

    return CIndexScorer()


def _fitted_grid(analyzer_name, X, y, scoring=None):
    """Fitted GridSearchCV over a single analyzer, as tasks.train_model builds
    it (the fitted search is what run_bootstrap & co. receive)."""
    if scoring is None:
        scoring = {"c-index": make_scorer(c_index_score)}
    grid = GridSearchCV(
        Pipeline([("preprocessor", None), ("analyzer", None)]),
        [
            {
                "preprocessor": [StandardScaler()],
                "analyzer": [ANALYZERS[analyzer_name]()],
            }
        ],
        scoring=scoring,
        refit="c-index",
        cv=SurvivalRepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0),
        n_jobs=1,
    )
    return grid.fit(X, y)


@pytest.fixture(scope="module")
def data():
    return _signal_data()


@pytest.fixture(scope="module")
def train_test():
    X, y = _signal_data(n=240)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X_df.iloc[:160], y[:160], X_df.iloc[160:], y[160:]


def _no_comparable_pairs_test_set(n_features=5, n=10):
    """n patients, a single event at the latest time: zero comparable pairs."""
    events = np.zeros(n, dtype=bool)
    events[-1] = True
    times = np.arange(1, n + 1, dtype=float) * 10.0
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, n_features))
    return X, Surv.from_arrays(events, times, name_event="Event", name_time="Time")


# ---------------------------------------------------------------------------
# Sanity: the premise of the bug in the installed sksurv
# ---------------------------------------------------------------------------


def test_sksurv_marks_ipcridge_as_not_predicting_risk_scores():
    # In sksurv 0.24 it is a property, so check instances
    assert IPCRidge()._predict_risk_score is False
    assert getattr(CoxPHSurvivalAnalysis(), "_predict_risk_score", True) is True
    assert getattr(CoxnetSurvivalAnalysis(), "_predict_risk_score", True) is True


# ---------------------------------------------------------------------------
# predicts_risk_score
# ---------------------------------------------------------------------------


class TestPredictsRiskScore:
    def test_false_for_ipcridge_estimator(self):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        assert predicts_risk_score(IPCRidge()) is False

    def test_false_for_fitted_ipcridge_estimator(self, data):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        X, y = data
        assert predicts_risk_score(IPCRidge().fit(X, y)) is False

    def test_false_for_pipeline_ending_in_ipcridge(self, data):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        X, y = data
        assert predicts_risk_score(_pipeline("IPCRidge")) is False
        assert predicts_risk_score(_fitted_pipeline("IPCRidge", X, y)) is False

    def test_false_for_fitted_grid_search_with_ipcridge_best_estimator(self, data):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        X, y = data
        assert predicts_risk_score(_fitted_grid("IPCRidge", X, y)) is False

    @pytest.mark.parametrize("analyzer_name", COX_ANALYZERS)
    def test_true_for_cox_models(self, data, analyzer_name):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        X, y = data
        assert predicts_risk_score(ANALYZERS[analyzer_name]()) is True
        assert predicts_risk_score(_fitted_pipeline(analyzer_name, X, y)) is True
        assert predicts_risk_score(_fitted_grid(analyzer_name, X, y)) is True

    def test_true_for_classifier(self):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 3))
        y = (X[:, 0] > 0).astype(int)
        clf = Pipeline(
            [("preprocessor", StandardScaler()), ("classifier", LogisticRegression())]
        ).fit(X, y)

        assert predicts_risk_score(LogisticRegression()) is True
        assert predicts_risk_score(clf) is True

    def test_returns_real_bool(self):
        from quantimage2_backend_common.modeling_utils import predicts_risk_score

        assert type(predicts_risk_score(IPCRidge())) is bool
        assert type(predicts_risk_score(CoxPHSurvivalAnalysis())) is bool


# ---------------------------------------------------------------------------
# to_risk_scores
# ---------------------------------------------------------------------------


class TestToRiskScores:
    def test_negates_ipcridge_predictions(self, data):
        from quantimage2_backend_common.modeling_utils import to_risk_scores

        X, y = data
        model = _fitted_grid("IPCRidge", X, y)
        preds = model.predict(X)

        result = to_risk_scores(model, preds)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, -preds)

    @pytest.mark.parametrize("analyzer_name", COX_ANALYZERS)
    def test_keeps_cox_predictions_unchanged(self, data, analyzer_name):
        from quantimage2_backend_common.modeling_utils import to_risk_scores

        X, y = data
        model = _fitted_grid(analyzer_name, X, y)
        preds = model.predict(X)

        result = to_risk_scores(model, preds)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, preds)

    def test_does_not_mutate_input(self, data):
        from quantimage2_backend_common.modeling_utils import to_risk_scores

        X, y = data
        model = _fitted_pipeline("IPCRidge", X, y)
        preds = model.predict(X)
        original = preds.copy()

        to_risk_scores(model, preds)

        np.testing.assert_array_equal(preds, original)


# ---------------------------------------------------------------------------
# CIndexScorer
# ---------------------------------------------------------------------------


class TestCIndexScorer:
    def test_issue_reproduction_ipcridge_matches_sksurv_score(self, data):
        """Exact reproduction from issue #57 (scorer gave 0.269 vs 0.731)."""
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        X, y = data
        p = _fitted_pipeline("IPCRidge", X, y)

        ours = CIndexScorer()(p, X, y)

        assert ours == pytest.approx(p.score(X, y), abs=1e-12)
        assert ours > 0.5

    @pytest.mark.parametrize("analyzer_name", list(ANALYZERS))
    def test_matches_sksurv_score_for_every_survival_model(self, data, analyzer_name):
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        X, y = data
        p = _fitted_pipeline(analyzer_name, X, y)

        assert CIndexScorer()(p, X, y) == pytest.approx(p.score(X, y), abs=1e-12)

    @pytest.mark.parametrize("analyzer_name", list(ANALYZERS))
    def test_matches_sksurv_score_through_fitted_grid_search(self, data, analyzer_name):
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        X, y = data
        grid = _fitted_grid(analyzer_name, X, y)

        assert CIndexScorer()(grid, X, y) == pytest.approx(
            grid.best_estimator_.score(X, y), abs=1e-12
        )

    def test_ipcridge_score_is_complement_of_raw_prediction_c_index(self, data):
        """Neighbouring case: the old (buggy) value is exactly 1 - new value."""
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        X, y = data
        p = _fitted_pipeline("IPCRidge", X, y)
        raw = c_index_score(y, p.predict(X))

        assert CIndexScorer()(p, X, y) == pytest.approx(1 - raw, abs=1e-12)

    def test_exposes_score_func_and_empty_kwargs(self):
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        scorer = CIndexScorer()

        assert scorer._score_func is c_index_score
        assert scorer._kwargs == {}

    def test_is_picklable_and_still_correct_after_round_trip(self, data):
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        X, y = data
        p = _fitted_pipeline("IPCRidge", X, y)
        scorer = pickle.loads(pickle.dumps(CIndexScorer()))

        assert scorer(p, X, y) == pytest.approx(p.score(X, y), abs=1e-12)

    def test_no_comparable_pair_exception_propagates(self, data):
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        X, y = data
        X_bad, y_bad = _no_comparable_pairs_test_set()
        for name in ("IPCRidge", "CoxPHSurvivalAnalysis"):
            p = _fitted_pipeline(name, X, y)
            with pytest.raises(NoComparablePairException):
                CIndexScorer()(p, X_bad, y_bad)


# ---------------------------------------------------------------------------
# Survival.get_scoring()
# ---------------------------------------------------------------------------


@requires_survival_module
class TestSurvivalGetScoring:
    def test_c_index_is_a_cindex_scorer(self):
        from quantimage2_backend_common.modeling_utils import CIndexScorer

        scoring = _survival_scoring()

        assert set(scoring) == {"c-index"}
        assert isinstance(scoring["c-index"], CIndexScorer)

    @pytest.mark.parametrize("analyzer_name", list(ANALYZERS))
    def test_c_index_matches_sksurv_score(self, data, analyzer_name):
        X, y = data
        p = _fitted_pipeline(analyzer_name, X, y)

        assert _survival_scoring()["c-index"](p, X, y) == pytest.approx(
            p.score(X, y), abs=1e-12
        )

    def test_survives_pickle_round_trip(self, data):
        X, y = data
        scoring = pickle.loads(pickle.dumps(_survival_scoring()))
        p = _fitted_pipeline("IPCRidge", X, y)

        assert set(scoring) == {"c-index"}
        assert scoring["c-index"](p, X, y) == pytest.approx(p.score(X, y), abs=1e-12)

    def test_grid_search_with_fake_progress_scorer_matches_sksurv_cv_scores(
        self, data, monkeypatch
    ):
        """Grid search as tasks.train_model runs it: pickled scoring dict +
        fake progress scorer, refit on c-index. Each candidate's CV c-index
        must equal sksurv's own score() over the same folds."""
        import utils

        socket = MagicMock()
        monkeypatch.setattr(utils, "_socketio", socket)

        X, y = data
        scoring = pickle.loads(pickle.dumps(_survival_scoring()))
        scoring = {
            **scoring,
            FAKE_SCORER_KEY: make_scorer(utils.fake_score, training_id=42),
        }
        cv = SurvivalRepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
        grid = GridSearchCV(
            Pipeline([("preprocessor", None), ("analyzer", None)]),
            [
                {"preprocessor": [StandardScaler()], "analyzer": [IPCRidge()]},
                {
                    "preprocessor": [StandardScaler()],
                    "analyzer": [CoxPHSurvivalAnalysis()],
                },
            ],
            scoring=scoring,
            refit="c-index",
            cv=cv,
            n_jobs=1,
        ).fit(X, y)

        # The progress scorer still runs (once per fold and candidate)
        assert socket.emit.call_count == 6

        means = grid.cv_results_["mean_test_c-index"]
        for i, name in enumerate(["IPCRidge", "CoxPHSurvivalAnalysis"]):
            expected = cross_val_score(_pipeline(name), X, y, cv=cv).mean()
            assert means[i] == pytest.approx(expected, abs=1e-12), name
            assert means[i] > 0.5, name

        assert grid.best_score_ == pytest.approx(max(means), abs=1e-12)


# ---------------------------------------------------------------------------
# run_bootstrap
# ---------------------------------------------------------------------------


def _expected_bootstrap_c_index(y_test, risk_scores, random_seed, n_bootstrap):
    """Bootstrap c-index computed directly on risk scores, using the same
    stratified resampling run_bootstrap uses for survival models."""
    expected = []
    for i in range(n_bootstrap):
        preds_res, y_res = resample(
            risk_scores,
            y_test,
            replace=True,
            n_samples=len(y_test),
            random_state=random_seed + i,
            stratify=[o[0] for o in y_test],
        )
        expected.append(c_index_score(y_res, preds_res))
    return expected


def _bootstrap_scorings():
    """Both the legacy make_scorer and (where available) production scoring:
    run_bootstrap must convert predictions whichever scorer is passed."""
    params = [
        pytest.param(lambda: {"c-index": make_scorer(c_index_score)}, id="make_scorer")
    ]
    params.append(
        pytest.param(
            _survival_scoring, id="survival_get_scoring", marks=requires_survival_module
        )
    )
    return params


class TestRunBootstrap:
    N_BOOTSTRAP = 10
    SEED = 3

    @pytest.mark.parametrize("make_scoring", _bootstrap_scorings())
    def test_ipcridge_bootstrap_c_index_reflects_real_discrimination(
        self, train_test, make_scoring
    ):
        from utils import run_bootstrap

        X_train, y_train, X_test, y_test = train_test
        model = _fitted_grid("IPCRidge", X_train, y_train)

        scores, n = run_bootstrap(
            X_test,
            y_test,
            model,
            self.SEED,
            make_scoring(),
            n_bootstrap=self.N_BOOTSTRAP,
        )

        values = [s["c-index"] for s in scores]
        assert n == self.N_BOOTSTRAP
        assert len(values) == self.N_BOOTSTRAP
        assert all(v > 0.5 for v in values), values
        assert np.mean(values) == pytest.approx(
            model.best_estimator_.score(X_test, y_test), abs=0.05
        )
        expected = _expected_bootstrap_c_index(
            y_test, -model.predict(X_test), self.SEED, self.N_BOOTSTRAP
        )
        assert values == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("make_scoring", _bootstrap_scorings())
    @pytest.mark.parametrize("analyzer_name", COX_ANALYZERS)
    def test_cox_bootstrap_values_unchanged(
        self, train_test, make_scoring, analyzer_name
    ):
        from utils import run_bootstrap

        X_train, y_train, X_test, y_test = train_test
        model = _fitted_grid(analyzer_name, X_train, y_train)

        scores, _ = run_bootstrap(
            X_test,
            y_test,
            model,
            self.SEED,
            make_scoring(),
            n_bootstrap=self.N_BOOTSTRAP,
        )

        values = [s["c-index"] for s in scores]
        expected = _expected_bootstrap_c_index(
            y_test, model.predict(X_test), self.SEED, self.N_BOOTSTRAP
        )
        assert values == pytest.approx(expected, abs=1e-12)
        assert all(v > 0.5 for v in values), values


# ---------------------------------------------------------------------------
# compute_feature_importance
# ---------------------------------------------------------------------------


class TestComputeFeatureImportance:
    @pytest.mark.parametrize("analyzer_name", list(ANALYZERS))
    def test_informative_feature_has_positive_and_largest_importance(
        self, train_test, analyzer_name
    ):
        from utils import compute_feature_importance

        X_train, y_train, X_test, y_test = train_test
        scoring = {"c-index": _c_index_scorer(analyzer_name)}
        model = _fitted_grid(analyzer_name, X_train, y_train, scoring=scoring)

        importances = compute_feature_importance(
            X_test, y_test, model, scoring, "Survival", random_seed=0
        )

        assert set(importances) == set(X_test.columns)
        assert importances["f0"] > 0.05, importances
        assert importances["f0"] == max(importances.values()), importances


# ---------------------------------------------------------------------------
# compute_risk_scores (output must stay as before)
# ---------------------------------------------------------------------------


class TestComputeRiskScoresUnchanged:
    def test_ipcridge_risk_scores_are_negated_predictions(self, train_test):
        from utils import compute_risk_scores

        X_train, y_train, X_test, _ = train_test
        model = _fitted_grid("IPCRidge", X_train, y_train)
        patients = [f"P{i}" for i in range(len(X_test))]

        result = compute_risk_scores(X_test, model, patients)

        preds = model.predict(X_test)
        assert list(result) == patients
        for patient, pred in zip(patients, preds):
            assert result[patient] == {"risk_score": pytest.approx(-float(pred))}
            assert isinstance(result[patient]["risk_score"], float)

    @pytest.mark.parametrize("analyzer_name", COX_ANALYZERS)
    def test_cox_risk_scores_are_raw_predictions(self, train_test, analyzer_name):
        from utils import compute_risk_scores

        X_train, y_train, X_test, _ = train_test
        model = _fitted_grid(analyzer_name, X_train, y_train)
        patients = [f"P{i}" for i in range(len(X_test))]

        result = compute_risk_scores(X_test, model, patients)

        preds = model.predict(X_test)
        assert list(result) == patients
        for patient, pred in zip(patients, preds):
            assert result[patient] == {"risk_score": pytest.approx(float(pred))}
            assert isinstance(result[patient]["risk_score"], float)


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_c_index_score_still_takes_risk_scores(self, data):
        X, y = data
        risk = X[:, 0]  # true hazard driver

        assert c_index_score(y, risk) > 0.7
        assert c_index_score(y, -risk) == pytest.approx(
            1 - c_index_score(y, risk), abs=1e-12
        )

    def test_old_pickled_make_scorer_still_unpickles(self, data):
        X, y = data
        blob = pickle.dumps({"c-index": make_scorer(c_index_score)})

        scoring = pickle.loads(blob)

        p = _fitted_pipeline("CoxPHSurvivalAnalysis", X, y)
        assert scoring["c-index"](p, X, y) == pytest.approx(p.score(X, y), abs=1e-12)

    def test_calculate_scores_uses_score_func_on_risk_scores(self, data):
        from quantimage2_backend_common.modeling_utils import CIndexScorer
        from utils import calculate_scores

        X, y = data
        risk = X[:, 0]

        scores = calculate_scores(y, risk, None, {"c-index": CIndexScorer()})

        assert scores == {"c-index": pytest.approx(c_index_score(y, risk), abs=1e-12)}

    @requires_survival_module
    def test_no_comparable_pair_exception_propagates_from_production_scorer(self, data):
        X, y = data
        X_bad, y_bad = _no_comparable_pairs_test_set()
        p = _fitted_pipeline("IPCRidge", X, y)

        with pytest.raises(NoComparablePairException):
            _survival_scoring()["c-index"](p, X_bad, y_bad)

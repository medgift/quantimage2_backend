"""fit_grid_search stops at once on a real soft-time-limit signal (#54).

The prefork pool sends SIGUSR1 to the process running the task; billiard's
handler raises SoftTimeLimitExceeded, which GridSearchCV would otherwise record
as a failed fit or score (error_score=nan) and carry on.
"""

import os
import signal

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1 on this platform"
)

X = np.arange(40, dtype=float).reshape(20, 2)
y = np.array([0, 1] * 10)


@pytest.fixture
def billiard_soft_timeout_handler():
    """Install the handler the prefork pool process has, restore afterwards."""
    from celery.exceptions import SoftTimeLimitExceeded

    def soft_timeout_sighandler(signum, frame):
        raise SoftTimeLimitExceeded()

    previous = signal.signal(signal.SIGUSR1, soft_timeout_sighandler)
    yield soft_timeout_sighandler
    signal.signal(signal.SIGUSR1, previous)


class _Counting(BaseEstimator, ClassifierMixin):
    """Sends the soft-limit signal on the n-th fit (or n-th score) call."""

    fits = 0
    signal_on_fit = None

    def __init__(self, c=1):
        self.c = c

    def fit(self, X, y):
        type(self).fits += 1
        if type(self).fits == type(self).signal_on_fit:
            os.kill(os.getpid(), signal.SIGUSR1)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _grid(scoring="accuracy"):
    return GridSearchCV(
        _Counting(), {"c": [1, 2, 3, 4]}, cv=3, n_jobs=1, scoring=scoring
    )


def test_signal_during_fit_aborts_search_immediately(billiard_soft_timeout_handler):
    from celery.exceptions import SoftTimeLimitExceeded
    from utils import fit_grid_search

    _Counting.fits, _Counting.signal_on_fit = 0, 2

    with pytest.raises(SoftTimeLimitExceeded):
        fit_grid_search(_grid(), X, y)

    # 12 candidate fits + refit would follow; the search stopped at the 2nd
    assert _Counting.fits == 2
    assert signal.getsignal(signal.SIGUSR1) is billiard_soft_timeout_handler


def test_signal_during_scoring_aborts_search(billiard_soft_timeout_handler):
    from celery.exceptions import SoftTimeLimitExceeded
    from utils import fit_grid_search

    calls = {"n": 0}

    def scorer(y_true, y_pred):
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGUSR1)
        return 0.0

    _Counting.fits, _Counting.signal_on_fit = 0, None

    with pytest.raises(SoftTimeLimitExceeded):
        fit_grid_search(_grid(make_scorer(scorer)), X, y)

    assert calls["n"] == 1
    assert _Counting.fits == 1


def test_no_signal_returns_fitted_search_and_restores_handler(
    billiard_soft_timeout_handler,
):
    from utils import fit_grid_search

    _Counting.fits, _Counting.signal_on_fit = 0, None

    fitted = fit_grid_search(_grid(), X, y)

    assert fitted.best_params_ in [{"c": c} for c in (1, 2, 3, 4)]
    assert _Counting.fits == 13  # 4 candidates x 3 folds + refit
    assert signal.getsignal(signal.SIGUSR1) is billiard_soft_timeout_handler


def test_other_fit_failures_still_only_warn(billiard_soft_timeout_handler):
    from sklearn.exceptions import FitFailedWarning
    from utils import fit_grid_search

    class _FailsOnce(_Counting):
        def fit(self, X, y):
            if self.c == 1:
                raise ValueError("bad candidate")
            return super().fit(X, y)

    with pytest.warns(FitFailedWarning, match="bad candidate"):
        fitted = fit_grid_search(
            GridSearchCV(_FailsOnce(), {"c": [1, 2]}, cv=3, n_jobs=1), X, y
        )
    assert fitted.best_params_ == {"c": 2}

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.metrics import concordance_index_censored


class SurvivalRepeatedStratifiedKFold(RepeatedStratifiedKFold):
    def split(self, X, y, groups=None):
        # Keep only the [0] element of each label, which corresponds to the Event (true/false)
        for train_index, test_index in super().split(X, [o[0] for o in y]):
            yield train_index, test_index


def c_index_score(y_true, y_pred):
    name_event, name_time = y_true.dtype.names

    c_index, _, _, _, _ = concordance_index_censored(
        y_true[name_event], y_true[name_time], y_pred
    )

    return c_index


def predicts_risk_score(estimator):
    """Whether ``estimator.predict`` returns risk scores (higher = earlier event).

    Unwraps a fitted search (``best_estimator_``) and a ``Pipeline`` (last step),
    then reads sksurv's ``_predict_risk_score`` flag: ``False`` for estimators that
    predict survival time (e.g. IPCRidge). Estimators without it count as risk scores.
    """
    final = getattr(estimator, "best_estimator_", estimator)
    if isinstance(final, Pipeline):
        final = final[-1]
    return bool(getattr(final, "_predict_risk_score", True))


def to_risk_scores(estimator, predictions):
    """Convert ``estimator``'s predictions to risk scores (negate survival times)."""
    predictions = np.asarray(predictions)
    return predictions if predicts_risk_score(estimator) else -predictions


class CIndexScorer:
    """Picklable c-index scorer that converts predictions to risk scores first.

    ``_score_func`` / ``_kwargs`` mirror sklearn scorers so code that calls the
    metric directly on (already converted) risk scores keeps working.
    """

    _score_func = staticmethod(c_index_score)

    @property
    def _kwargs(self):
        # A fresh dict each time: no state shared between instances
        return {}

    def __call__(self, estimator, X, y):
        return c_index_score(y, to_risk_scores(estimator, estimator.predict(X)))

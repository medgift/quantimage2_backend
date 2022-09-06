from sklearn.model_selection import RepeatedStratifiedKFold
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

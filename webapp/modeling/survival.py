from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge

from quantimage2_backend_common.modeling_utils import (
    c_index_score,
    SurvivalRepeatedStratifiedKFold,
)
from quantimage2_backend_common.utils import CV_SPLITS, CV_REPEATS
from modeling.modeling import Modeling

from sksurv.util import Surv

from service.feature_transformation import (
    OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME,
)

SURVIVAL_METHODS = ["coxnet", "coxnet_elastic", "ipc_ridge"]
SURVIVAL_PARAMS = {
    "coxnet": {"alpha": [0.1], "n_iter": [100]},
    "coxnet_elastic": {"n_alphas": [100], "l1_ratio": [0.5]},
    "ipc_ridge": {"alpha": [1]},
}


class Survival(Modeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_survival_analyzer(self, analyzer_name):
        if analyzer_name == "coxnet":
            options = {"analyzer": [CoxPHSurvivalAnalysis()]}
            for key, value in SURVIVAL_PARAMS[analyzer_name].items():
                options[f"analyzer__{key}"] = value
            return options
        elif analyzer_name == "coxnet_elastic":
            return {"analyzer": [CoxnetSurvivalAnalysis()]}
        elif analyzer_name == "ipc_ridge":
            return {"analyzer": [IPCRidge()]}

    def encode_labels(self, labels):
        # Transform Event column to boolean
        labels_copy = labels.copy()
        labels_copy["Event"] = labels_copy["Event"].astype(bool)

        # Transform to structured array
        encoded_labels = Surv.from_dataframe(
            OUTCOME_FIELD_SURVIVAL_EVENT, OUTCOME_FIELD_SURVIVAL_TIME, labels_copy
        )

        return encoded_labels

    def get_cv(self, n_splits=CV_SPLITS, n_repeats=CV_REPEATS):
        return SurvivalRepeatedStratifiedKFold(
            random_state=self.random_seed, n_splits=n_splits, n_repeats=n_repeats
        )

    def get_scoring(self):
        # return {"c-index": as_concordance_index_ipcw_scorer}
        return {"c-index": make_scorer(c_index_score)}

    def get_pipeline(self):
        return Pipeline([("preprocessor", None), ("analyzer", None)])

    def get_parameter_grid(self):
        methods = []
        for survival_method in SURVIVAL_METHODS:
            methods.append(
                {**self.preprocessor, **self.select_survival_analyzer(survival_method)}
            )

        return methods

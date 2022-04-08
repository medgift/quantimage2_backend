from modeling.survival import Survival
from service.feature_transformation import (
    OUTCOME_FIELD_SURVIVAL_TIME,
    OUTCOME_FIELD_SURVIVAL_EVENT,
)
from modeling.utils import get_random_seed
from service.machine_learning import get_features_labels
from sksurv.util import Surv
import pandas


def train_survival_model(
    extraction_id,
    collection_id,
    studies,
    data_splitting_type,
    training_patients,
    test_patients,
    gt,
):
    features_df, labels_df_indexed = get_features_labels(
        extraction_id,
        collection_id,
        studies,
        gt,
        outcome_columns=[OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT],
    )

    # Convert to numeric values
    labels_df_indexed = labels_df_indexed.apply(pandas.to_numeric)

    training_validation = None
    test_validation = None
    test_validation_params = None

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    survival = Survival(
        features_df,
        labels_df_indexed,
        data_splitting_type,
        training_patients,
        test_patients,
        random_seed,
        "c-index",
    )

    # Run modeling pipeline depending on the type of validation (full CV, train/test)
    return survival.analyze()


def refit_survival(results):
    print(results)

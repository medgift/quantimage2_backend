import os

from quantimage2_backend_common.const import ESTIMATOR_STEP
from quantimage2_backend_common.utils import get_training_id
from modeling.survival import Survival
from service.feature_transformation import (
    OUTCOME_FIELD_SURVIVAL_TIME,
    OUTCOME_FIELD_SURVIVAL_EVENT,
)
from modeling.utils import get_random_seed
from service.machine_learning import get_features_labels
import pandas


def train_survival_model(
    extraction_id,
    collection_id,
    album,
    studies,
    feature_selection,
    label_category,
    data_splitting_type,
    train_test_splitting_type,
    training_patients,
    test_patients,
    gt,
    user_id,
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

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    training_id = get_training_id(extraction_id, collection_id)

    survival = Survival(
        feature_extraction_id=extraction_id,
        collection_id=collection_id,
        album=album,
        feature_selection=feature_selection,
        feature_names=features_df.columns,  # TODO - This might change with feature selection
        estimator_step=ESTIMATOR_STEP.SURVIVAL.value,
        label_category=label_category,
        features_df=features_df,
        labels_df=labels_df_indexed,
        data_splitting_type=data_splitting_type,
        train_test_splitting_type=train_test_splitting_type,
        training_patients=training_patients,
        test_patients=test_patients,
        random_seed=random_seed,
        refit_metric="c-index",
        n_jobs=int(os.environ["GRID_SEARCH_CONCURRENCY"]),
        training_id=training_id,
    )

    # Run modeling pipeline depending on the type of validation (full CV, train/test)
    return survival.analyze()

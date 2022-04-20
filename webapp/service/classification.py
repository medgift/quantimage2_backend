import os

from imaginebackend_common.const import ESTIMATOR_STEP
from imaginebackend_common.models import FeatureExtraction, FeatureCollection
from imaginebackend_common.utils import get_training_id
from modeling.classification import Classification
from service.feature_transformation import (
    transform_studies_features_to_df,
    transform_studies_collection_features_to_df,
    OUTCOME_FIELD_CLASSIFICATION,
)
from modeling.utils import get_random_seed
from service.machine_learning import concatenate_modalities_rois, get_features_labels
import pandas


def train_classification_model(
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
):
    features_df, labels_df_indexed = get_features_labels(
        extraction_id,
        collection_id,
        studies,
        gt,
        outcome_columns=[OUTCOME_FIELD_CLASSIFICATION],
    )

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    training_id = get_training_id(extraction_id, collection_id)

    classifier = Classification(
        feature_extraction_id=extraction_id,
        collection_id=collection_id,
        album=album,
        feature_selection=feature_selection,
        feature_names=features_df.columns,  # TODO - This might change with feature selection
        estimator_step=ESTIMATOR_STEP.CLASSIFICATION.value,
        label_category=label_category,
        features_df=features_df,
        labels_df=labels_df_indexed,
        data_splitting_type=data_splitting_type,
        train_test_splitting_type=train_test_splitting_type,
        training_patients=training_patients,
        test_patients=test_patients,
        random_seed=random_seed,
        refit_metric="auc",
        n_jobs=int(os.environ["GRID_SEARCH_CONCURRENCY"]),
        training_id=training_id,
    )

    # Run modeling pipeline depending on the type of validation (full CV, train/test)
    return classifier.classify()

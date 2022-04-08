from imaginebackend_common.models import FeatureExtraction, FeatureCollection
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
        outcome_columns=[OUTCOME_FIELD_CLASSIFICATION],
    )

    training_validation = None
    test_validation = None
    test_validation_params = None

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    classifier = Classification(
        features_df,
        labels_df_indexed,
        data_splitting_type,
        training_patients,
        test_patients,
        random_seed,
        "auc",
    )

    # Run modeling pipeline depending on the type of validation (full CV, train/test)
    return classifier.classify()

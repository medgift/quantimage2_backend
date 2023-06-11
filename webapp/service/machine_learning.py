import traceback
import os

import pandas
from flask import jsonify, make_response
from modeling.classification import Classification
from modeling.survival import Survival
from modeling.utils import get_random_seed
from quantimage2_backend_common.const import FEATURE_ID_SEPARATOR, MODEL_TYPES, ESTIMATOR_STEP
from quantimage2_backend_common.models import (FeatureCollection,
                                               FeatureExtraction)
from quantimage2_backend_common.utils import get_training_id
from service.feature_transformation import (
    OUTCOME_FIELD_CLASSIFICATION, OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME, transform_studies_collection_features_to_df,
    transform_studies_features_to_df)


def get_features_labels(
    extraction_id,
    collection_id,
    studies,
    gt,
    outcome_columns=[OUTCOME_FIELD_CLASSIFICATION],
):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        header, features_df = transform_studies_collection_features_to_df(
            collection, studies
        )
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

    # Get Labels DataFrame
    # TODO - Allow choosing a mode (Patient only or Patient + ROI)
    print(gt)
    labels_df = pandas.DataFrame(gt, columns=["PatientID", *outcome_columns])

    labels_df_indexed = labels_df.set_index("PatientID", drop=True)

    # TODO - Check how to best deal with this, so far we ignore unlabelled patients
    label_conditions = None
    for column in outcome_columns:
        if label_conditions is None:
            label_conditions = (labels_df[column] != "") & (labels_df[column].notnull())
        else:
            label_conditions = (
                label_conditions
                & (labels_df[column] != "")
                & (labels_df[column].notnull())
            )

    labelled_patients = list(labels_df[label_conditions]["PatientID"])

    # Filter out unlabelled patients
    features_df = features_df[features_df.PatientID.isin(labelled_patients)]
    labels_df_indexed = labels_df_indexed.filter(items=labelled_patients, axis=0)

    # Concatenate features by modality & ROI
    features_df = concatenate_modalities_rois(features_df)

    # Sort both dataframe by index to ensure same order
    labels_df_indexed.sort_index(inplace=True)
    features_df.sort_index(inplace=True)

    # TODO - This will be done in Melampus also in the future
    # Impute mean for NaNs
    features_df = features_df.fillna(features_df.mean())

    return features_df, labels_df_indexed


def concatenate_modalities_rois(features_df):
    # Concatenate features from the various modalities & ROIs (if necessary)

    # Keep PatientID
    patient_id_df = features_df["PatientID"].to_frame()
    unique_pid_df = patient_id_df.drop_duplicates(subset="PatientID")
    unique_pid_df = unique_pid_df.set_index("PatientID", drop=False)

    separator = FEATURE_ID_SEPARATOR

    to_concat = [unique_pid_df]
    # Groupe dataframes by Modality & ROI
    for group, groupDf in features_df.groupby(["Modality", "ROI"]):
        # Only keep selected modalities & ROIs
        without_modality_and_roi_df = groupDf.drop(["Modality", "ROI"], axis=1)
        without_modality_and_roi_df = without_modality_and_roi_df.set_index(
            "PatientID", drop=True
        )
        prefix = separator.join(group)
        without_modality_and_roi_df = without_modality_and_roi_df.add_prefix(
            prefix + separator
        )
        # Drop columns with only NaNs
        without_modality_and_roi_df.dropna(axis=1, inplace=True, how="all")
        to_concat.append(without_modality_and_roi_df)

    # Add back the Patient ID at the end
    concatenated_df = pandas.concat(to_concat, axis=1)

    return concatenated_df


def train_model(
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

    if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION:
        outcome_columns = [OUTCOME_FIELD_CLASSIFICATION]
        ModelClass = Classification
    elif MODEL_TYPES(label_category.label_type) == MODEL_TYPES.SURVIVAL:
        outcome_columns = [OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT]
        ModelClass = Survival
    else:
        raise NotImplementedError()

    features_df, labels_df_indexed = get_features_labels(
        extraction_id,
        collection_id,
        studies,
        gt,
        outcome_columns=outcome_columns,
    )

    # Convert to numeric values
    labels_df_indexed = labels_df_indexed.apply(pandas.to_numeric)

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    training_id = get_training_id(extraction_id, collection_id)

    model = ModelClass(
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

    return model.classify()
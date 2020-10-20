import os

import pandas
from flask import Blueprint, jsonify, request, g, current_app, Response

# Define blueprint
from sklearn.preprocessing import StandardScaler

from imaginebackend_common.const import MODEL_TYPES
from imaginebackend_common.models import FeatureExtraction, Label
from service.feature_extraction import get_studies_from_album
from service.feature_transformation import (
    transform_studies_features_to_df,
    PATIENT_ID_FIELD,
    MODALITY_FIELD,
    ROI_FIELD,
    OUTCOME_FIELD_CLASSIFICATION,
)

bp = Blueprint(__name__, "charts")


@bp.route("/charts/<extraction_id>/lasagna")
def lasagna_chart(extraction_id):

    # To simplify the access, use album token (for fixed album so far)

    album_token = os.environ["KHEOPS_ALBUM_TOKEN"]

    extraction = FeatureExtraction.find_by_id(extraction_id)
    studies = get_studies_from_album(extraction.album_id, album_token)

    header, features_df = transform_studies_features_to_df(extraction, studies)

    # TODO - Determine which labels we want to get???
    labels = Label.find_by_album(
        extraction.album_id, extraction.user_id, MODEL_TYPES.CLASSIFICATION.value
    )

    formatted_lasagna_data = format_lasagna_data(features_df, labels)

    return jsonify(formatted_lasagna_data)


def format_chart_labels(labels):
    formatted_labels = []

    for label in labels:
        formatted_labels.append(
            {
                PATIENT_ID_FIELD: label.patient_id,
                OUTCOME_FIELD_CLASSIFICATION: label.label_content[
                    OUTCOME_FIELD_CLASSIFICATION
                ],
            }
        )

    return formatted_labels


def format_lasagna_data(features_df, labels):

    features_df_reindexed = features_df.reset_index(drop=True)

    # Standardize features (by column)
    features_df_standardized = pandas.DataFrame(
        StandardScaler().fit_transform(
            features_df_reindexed.loc[
                :, ~features_df_reindexed.columns.isin(["PatientID", "Modality", "ROI"])
            ]
        )
    )
    features_df_standardized.columns = features_df_reindexed.columns[3:]

    full_df = pandas.concat(
        [
            features_df_reindexed.loc[
                :, features_df_reindexed.columns.isin(["PatientID", "Modality", "ROI"])
            ],
            features_df_standardized,
        ],
        axis=1,
    )

    # Features
    features_list = full_df.to_dict(orient="records")

    formatted_features = []

    for patient_record in features_list:
        always_present_fields = [PATIENT_ID_FIELD, MODALITY_FIELD, ROI_FIELD]

        patient_id = patient_record[PATIENT_ID_FIELD]
        modality = patient_record[MODALITY_FIELD]
        roi = patient_record[ROI_FIELD]

        for feature_name, feature_value in patient_record.items():
            # Don't add the Patient ID as another feature
            if feature_name not in always_present_fields:
                formatted_features.append(
                    {
                        PATIENT_ID_FIELD: patient_id,
                        MODALITY_FIELD: modality,
                        ROI_FIELD: roi,
                        "feature_id": "-".join([modality, roi]) + "-" + feature_name,
                        "feature_name": feature_name,
                        "feature_value": feature_value,
                    }
                )

    # Labels
    formatted_labels = format_chart_labels(labels)

    return {"features": formatted_features, "outcomes": formatted_labels}


@bp.route("/charts/<extraction_id>/pca")
def pca_chart(extraction_id):
    return None

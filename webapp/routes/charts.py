import json
import os

import pandas
from flask import Blueprint, jsonify, request, g, current_app, Response

# Define blueprint
from sklearn.preprocessing import StandardScaler

from imaginebackend_common.const import MODEL_TYPES
from imaginebackend_common.models import FeatureExtraction, Label, FeatureCollection
from routes.utils import decorate_if_possible
from service.feature_analysis import concatenate_modalities_rois
from service.feature_extraction import get_studies_from_album
from service.feature_transformation import (
    transform_studies_features_to_df,
    PATIENT_ID_FIELD,
    MODALITY_FIELD,
    ROI_FIELD,
    OUTCOME_FIELD_CLASSIFICATION,
    transform_studies_collection_features_to_df,
)

from melampus.feature_ranking import MelampusFeatureRank

bp = Blueprint(__name__, "charts")


@bp.before_request
def before_request():
    if not request.path.endswith("download"):
        decorate_if_possible(request)


@bp.route("/charts/<album_id>/lasagna", defaults={"collection_id": None})
@bp.route("/charts/<album_id>/<collection_id>/lasagna")
def lasagna_chart(album_id, collection_id):

    print("collection_id", collection_id)

    # TODO - Remove this hard-coded test route that's used by Julien
    if album_id.isnumeric():
        # To simplify the access, use album token (for fixed album so far)
        token = os.environ["KHEOPS_ALBUM_TOKEN"]
        # Album ID is actually an extraction ID in this setting
        extraction_id = int(album_id)
    else:
        token = g.token
        user_id = g.user

        # Find latest feature extraction for this album
        latest_extraction_of_album = FeatureExtraction.find_latest_by_user_and_album_id(
            user_id, album_id
        )

        extraction_id = latest_extraction_of_album.id

    extraction = FeatureExtraction.find_by_id(extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    # Whole extraction or sub-collection?
    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)

        header, features_df = transform_studies_collection_features_to_df(
            extraction, studies, collection
        )
    else:
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

    # Flatten features by Modality & ROI to calculate ranks
    concatenated_features_df = concatenate_modalities_rois(
        features_df,
        list(features_df[MODALITY_FIELD].unique()),
        list(features_df[ROI_FIELD].unique()),
    )

    # Reset the index to avoid problems with Melampus
    concatenated_features_df.reset_index(drop=True, inplace=True)

    # Get the outcomes in the same order as they appear in the DataFrame
    outcomes = []
    for index, row in concatenated_features_df.iterrows():
        label_to_add = next(
            label.label_content[OUTCOME_FIELD_CLASSIFICATION]
            for label in labels
            if (label.patient_id == row[PATIENT_ID_FIELD])
        )
        outcomes.append(label_to_add)

    # Feature Ranking
    feature_ranking = MelampusFeatureRank(
        None,
        concatenated_features_df,
        None,
        outcomes,
        id_names_map={"patient_id": PATIENT_ID_FIELD},
    )

    ranked_features = feature_ranking.rank_by_univariate_f(
        return_type="names", ascending=False
    )

    feature_rank_map = {k: v for v, k in enumerate(list(ranked_features))}

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
    # features_list = full_df.to_dict(orient="records")

    features_list = json.loads(full_df.to_json(orient="records"))

    formatted_features = []

    for patient_record in features_list:
        always_present_fields = [PATIENT_ID_FIELD, MODALITY_FIELD, ROI_FIELD]

        patient_id = patient_record[PATIENT_ID_FIELD]
        modality = patient_record[MODALITY_FIELD]
        roi = patient_record[ROI_FIELD]

        for feature_name, feature_value in patient_record.items():
            feature_id = "-".join([modality, roi]) + "-" + feature_name

            # Don't add the Patient ID as another feature
            if feature_name not in always_present_fields:
                formatted_features.append(
                    {
                        PATIENT_ID_FIELD: patient_id,
                        MODALITY_FIELD: modality,
                        ROI_FIELD: roi,
                        "feature_rank": feature_rank_map[feature_id]
                        if feature_value is not None
                        else None,
                        "feature_id": feature_id,
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

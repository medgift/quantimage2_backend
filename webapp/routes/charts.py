import json
import os
import re

import pandas
from flask import Blueprint, jsonify, request, g, current_app, Response

# Define blueprint
from sklearn.preprocessing import StandardScaler

from imaginebackend_common.const import MODEL_TYPES, featureIDMatcher
from imaginebackend_common.models import (
    FeatureExtraction,
    Label,
    FeatureCollection,
    Album,
    LabelCategory,
    AlbumOutcome,
)
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
    OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME,
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
        # User is taken from the environment too
        user_id = os.environ["KHEOPS_OUTCOME_USER_ID"]
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
            collection, studies
        )
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

    album_outcome = AlbumOutcome.find_by_album_user_id(extraction.album_id, user_id)

    # Get labels (if current outcome is defined for this user)
    labels = []
    label_category = None

    if album_outcome:
        label_category = LabelCategory.find_by_id(album_outcome.outcome_id)
        labels = Label.find_by_label_category(label_category.id)

    formatted_lasagna_data = format_lasagna_data(features_df, label_category, labels)

    return jsonify(formatted_lasagna_data)


def format_chart_labels(labels):
    formatted_labels = []

    # Spread out the whole label content into the labels
    for label in labels:
        formatted_labels.append(
            {PATIENT_ID_FIELD: label.patient_id, **label.label_content}
        )

    return formatted_labels


def format_lasagna_data(features_df, label_category, labels):

    # Flatten features by Modality & ROI to calculate ranks
    concatenated_features_df = concatenate_modalities_rois(features_df)

    # Reset the index to avoid problems with Melampus
    concatenated_features_df.reset_index(drop=True, inplace=True)

    # Define which field to use for visualization of the data
    if label_category:
        visualization_field_names = (
            [OUTCOME_FIELD_CLASSIFICATION]
            if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION
            else [OUTCOME_FIELD_SURVIVAL_EVENT, OUTCOME_FIELD_SURVIVAL_TIME]
        )
        ranking_field_name = (
            OUTCOME_FIELD_CLASSIFICATION
            if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION
            else OUTCOME_FIELD_SURVIVAL_EVENT
        )

    else:
        visualization_field_names = [OUTCOME_FIELD_CLASSIFICATION]
        ranking_field_name = OUTCOME_FIELD_CLASSIFICATION

    # Get the outcomes in the same order as they appear in the DataFrame
    outcomes = []
    for index, row in concatenated_features_df.iterrows():
        label_to_add = next(
            (
                label.label_content
                for label in labels
                if label.patient_id == row[PATIENT_ID_FIELD]
                and list(label.label_content.values())[0] != ""
            ),
            {
                visualization_field_name: "UNKNOWN"
                for (visualization_field_name) in visualization_field_names
            },
        )
        outcomes.append(label_to_add)

    # TODO - This will be done in Melampus also in the future
    # Imput mean values for NaNs to avoid problems for feature ranking
    no_nan_concatenated_features_df = concatenated_features_df.fillna(
        concatenated_features_df.mean(numeric_only=True)
    )

    # Feature Ranking
    # TODO - Feature Ranking should be done differently for survival!
    outcomes_list_ranking = [outcome[ranking_field_name] for outcome in outcomes]

    feature_ranking = MelampusFeatureRank(
        None,
        no_nan_concatenated_features_df,
        None,
        outcomes_list_ranking,
        id_names_map={"patient_id": PATIENT_ID_FIELD},
    )

    ranked_features = feature_ranking.rank_by_univariate_f(
        return_type="names", ascending=False
    )

    feature_rank_map = {k: v for v, k in enumerate(list(ranked_features))}

    # Standardize features (by CONCATENATED columns!)
    concatenated_features_df_standardized = pandas.DataFrame(
        StandardScaler().fit_transform(
            concatenated_features_df.loc[
                :, ~concatenated_features_df.columns.isin(["PatientID"])
            ]
        )
    )

    full_df = pandas.concat(
        [
            concatenated_features_df.loc[
                :, concatenated_features_df.columns.isin(["PatientID"])
            ],
            concatenated_features_df_standardized,
        ],
        axis=1,
    )

    # Put back columns names from concatenated dataframe
    full_df.columns = concatenated_features_df.columns

    # Features
    features_list = json.loads(full_df.to_json(orient="records"))

    formatted_features = []

    formatted_labels = []
    patientIdx = 0
    for patient_record in features_list:
        patient_id = patient_record[PATIENT_ID_FIELD]

        # Add outcome on the backend already to avoid doing this in React
        patient_outcome = outcomes[patientIdx]

        # Add formatted label to list
        formatted_labels.append(
            {
                PATIENT_ID_FIELD: patient_id,
                **patient_outcome,
            }
        )

        for feature_id, feature_value in patient_record.items():
            # Don't add the Patient ID as another feature
            if feature_id != PATIENT_ID_FIELD:
                # Get modality, ROI & feature based on the feature name
                matches = featureIDMatcher.match(feature_id)

                modality = matches.group("modality")
                roi = matches.group("roi")
                feature_name = matches.group("feature")

                formatted_features.append(
                    {
                        PATIENT_ID_FIELD: patient_id,
                        MODALITY_FIELD: modality,
                        ROI_FIELD: roi,
                        **patient_outcome,
                        "feature_rank": feature_rank_map[feature_id]
                        if feature_value is not None
                        else None,
                        "feature_id": feature_id,
                        "feature_name": feature_name,
                        "feature_value": feature_value,
                    }
                )

        patientIdx += 1

    # Labels
    # formatted_labels = format_chart_labels(labels)

    return {"features": formatted_features, "outcomes": formatted_labels}


@bp.route("/charts/<extraction_id>/pca")
def pca_chart(extraction_id):
    return None

import json
import os
import re

import pandas
from flask import Blueprint, jsonify, request, g, current_app, Response

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

# Define blueprint
bp = Blueprint(__name__, "charts")


@bp.before_request
def before_request():
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

    formatted_chart_data = format_chart_data(features_df, label_category, labels)

    return jsonify(formatted_chart_data)


def format_chart_labels(labels):
    return [
        {PATIENT_ID_FIELD: label.patient_id, **label.label_content} for label in labels
    ]


def format_chart_data(features_df, label_category, labels):

    # Flatten features by Modality & ROI to calculate ranks
    concatenated_features_df = concatenate_modalities_rois(features_df)

    # No ranking by default
    feature_ranks_df = None

    # If there are any active labels, we can do feature ranking
    if label_category and len(labels) > 0:
        feature_ranks_df = calculate_feature_rankings(
            label_category, labels, concatenated_features_df
        )

    # Drop PatientID column (it is not needed in the output)
    concatenated_features_df.drop("PatientID", axis=1, inplace=True)

    standardized_features_df = pandas.DataFrame(
        StandardScaler().fit_transform(concatenated_features_df),
        index=concatenated_features_df.index,
        columns=concatenated_features_df.columns,
    )

    transposed_features_df = standardized_features_df.transpose()

    # Add ranking to the DF (if available)
    if feature_ranks_df is not None:
        transposed_features_df = transposed_features_df.merge(
            feature_ranks_df, left_index=True, right_index=True
        )

    return transposed_features_df


def calculate_feature_rankings(label_category, labels, concatenated_features_df):
    ranking_field_name = (
        OUTCOME_FIELD_CLASSIFICATION
        if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION
        else OUTCOME_FIELD_SURVIVAL_EVENT
    )

    formatted_labels = [
        {PATIENT_ID_FIELD: l.patient_id, **l.label_content} for l in labels
    ]
    # Get the outcomes as a DataFrame & index it by Patient ID
    outcomes_df = pandas.DataFrame(formatted_labels).set_index("PatientID")

    # TODO - This will be done in Melampus also in the future
    # Get the outcomes df with only the column for ranking
    outcomes_df_ranking = pandas.DataFrame(outcomes_df[ranking_field_name])
    outcomes_df_ranking.columns = ["Outcome"]

    # TODO - This will be done in Melampus also in the future
    # Imput mean values for NaNs to avoid problems for feature ranking
    no_nan_concatenated_features_df = concatenated_features_df.fillna(
        concatenated_features_df.mean(numeric_only=True)
    )

    no_nan_concatenated_features_with_outcomes_df = pandas.concat(
        [no_nan_concatenated_features_df, outcomes_df_ranking], axis=1
    )

    no_nan_concatenated_features_with_outcomes_no_nan_df = (
        no_nan_concatenated_features_with_outcomes_df.dropna(subset=["Outcome"])
    )

    # Reset the index to avoid problems with Melampus
    # concatenated_features_df.reset_index(drop=True, inplace=True)

    # Feature Ranking
    # TODO - Feature Ranking should be done differently for survival!
    # outcomes_list_ranking = [outcome[ranking_field_name] for outcome in outcomes]

    feature_ranking = MelampusFeatureRank(
        None,
        no_nan_concatenated_features_with_outcomes_no_nan_df,
        "Outcome",
        outcomes=[],
        id_names_map={"patient_id": PATIENT_ID_FIELD},
    )

    ranked_features = feature_ranking.rank_by_univariate_f(
        return_type="names", ascending=False
    )

    feature_rank_map = {k: v for v, k in enumerate(list(ranked_features))}

    feature_ranks_df = pandas.DataFrame.from_dict(feature_rank_map, orient="index")
    feature_ranks_df.columns = ["Ranking"]

    return feature_ranks_df

import pandas
from flask import Blueprint, jsonify, request, g

from sklearn.preprocessing import StandardScaler

from quantimage2_backend_common.const import MODEL_TYPES
from quantimage2_backend_common.models import (
    FeatureExtraction,
    Label,
    FeatureCollection,
    LabelCategory,
    AlbumOutcome,
)
from routes.utils import decorate_if_possible
from service.feature_extraction import get_studies_from_album
from service.machine_learning import concatenate_modalities_rois
from service.feature_transformation import (
    transform_studies_features_to_df,
    PATIENT_ID_FIELD,
    OUTCOME_FIELD_CLASSIFICATION,
    OUTCOME_FIELD_SURVIVAL_EVENT,
)

from melampus.feature_ranking import MelampusFeatureRank

# Define blueprint
bp = Blueprint(__name__, "charts")


@bp.before_request
def before_request():
    decorate_if_possible(request)


def format_chart_labels(labels):
    return [
        {PATIENT_ID_FIELD: label.patient_id, **label.label_content} for label in labels
    ]


def format_chart_data(features_df, label_category, labels):

    # Flatten features by Modality & ROI to calculate ranks
    concatenated_features_df = concatenate_modalities_rois(features_df)

    # No ranking by default
    feature_ranks_df = None

    # Filter out labels that aren't part of the current DataFrame
    patients_in_df = list(features_df.PatientID.unique())
    filtered_labels = [l for l in labels if l.patient_id in patients_in_df]

    # If there are any active labels, we can do feature ranking
    if label_category and len(labels) > 0:
        feature_ranks_df = calculate_feature_rankings(
            label_category, filtered_labels, concatenated_features_df
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

import pandas as pd

from typing import Dict
from collections import defaultdict
from flask import Blueprint, jsonify, request, g
from quantimage2_backend_common.models import (
    ClinicalFeatureDefinition,
    ClinicalFeatureValue,
)
from routes.utils import validate_decorate
from quantimage2_backend_common.models import ClinicalFeatureTypes

from service.feature_transformation import PATIENT_ID_FIELD

# Define blueprint
bp = Blueprint(__name__, "clinical_features")

NA_VALUE = "N/A"


@bp.before_request
def before_request():
    validate_decorate(request)


def load_df_from_request_dict(request_dict: Dict) -> pd.core.frame.DataFrame:
    clinical_features_list = []

    for patient_id, features in request_dict.items():
        features[PATIENT_ID_FIELD] = patient_id
        clinical_features_list.append(features)

    clinical_features_df = pd.DataFrame.from_dict(clinical_features_list)
    # Replace N/A or empty strings by Nones
    # TODO - Implement smarter way to detect N/A values (e.g. regex)
    clinical_features_df.replace(["", "N/A"], None, inplace=True)
    return clinical_features_df


def get_album_id_from_request(request):
    album_id = request.args.get("album_id")

    if not album_id:
        raise ValueError("album_id is required for this request")

    return album_id


@bp.route("/clinical-features/unique-values", methods=["POST"])
def clinical_features_unique_values():
    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(
            request.json["clinical_feature_map"]
        )
        # Filter out Patient ID column, as it will not be used to compute unique values
        clinical_features_df = clinical_features_df.drop(
            columns=PATIENT_ID_FIELD, errors="ignore"
        )

        feature_definitions = request.json["clinical_features_definitions"]
        feature_types = {
            name: value["feat_type"] for name, value in feature_definitions.items()
        }

        response = {}

        # Computing features with no data at all (using strings because we are not guarantee to get nulls from the request)
        for column in clinical_features_df.columns:
            series = clinical_features_df[column]

            if (
                ClinicalFeatureTypes(feature_types[column])
                == ClinicalFeatureTypes.CATEGORICAL
            ):
                frequency_of_occurrence = (
                    series.value_counts() / series.value_counts().sum()
                ) * 100

                response[column] = [
                    f"{value} ({round(percentage, 2)}%)"
                    if len(value) > 0
                    else f"{NA_VALUE} ({round(percentage, 2)}%)"
                    for value, percentage in frequency_of_occurrence.items()
                ]
            elif (
                ClinicalFeatureTypes(feature_types[column])
                == ClinicalFeatureTypes.NUMBER
            ):
                try:
                    # Filter out empty values when determining the min & max
                    filtered_values = series[series != ""]
                    filtered_values = filtered_values.astype(float)

                    min_value = filtered_values.min()
                    max_value = filtered_values.max()
                    response[column] = [
                        f"min={round(min_value, 2)}",
                        f"max={round(max_value, 2)}",
                    ]
                except Exception as e:
                    print(f"Could not convert {column} to float")
                    response[column] = [
                        "ERROR parsing numerical values for this column"
                    ]
                    continue
            else:
                raise ValueError(
                    f"Unsupported feature type detected for feature {column}"
                )

        return response


@bp.route("/clinical-features/filter", methods=["POST"])
def clinical_features_filter():
    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(
            request.json["clinical_feature_map"]
        )
        nulls_df = pd.DataFrame()

        response = {}
        response["only_one_value"] = []
        date_columns = []

        # Computing features with no data at all (using strings because we are not guarantee to get nulls from the request)
        for column in clinical_features_df.columns:
            if (
                column == "__parsed_extra"
            ):  # This happens when papa parse encounters some funkyness in the CSV
                continue
            nulls_df[column] = (
                clinical_features_df[column].astype(str).apply(lambda x: len(x))
            )  # we first create a dataframe with the same shape as the clinical features - but with the length of the string in each cell - len == 0 -> no data.

            # Number of unique values per feature
            n_unique = clinical_features_df[column].unique()
            if len(n_unique) == 1:
                response["only_one_value"].append(column)

            if "date" in column.lower():
                date_columns.append(column)

        columns_with_only_nulls = (nulls_df == 0).sum() == len(clinical_features_df)
        response["only_nulls"] = columns_with_only_nulls[
            columns_with_only_nulls
        ].index.tolist()

        # Dropping features with too little data
        percent_nulls = ((nulls_df == 0).sum() / len(clinical_features_df)) >= 0.9
        response["too_little_data"] = percent_nulls[percent_nulls].index.tolist()

        # Columns that have date in the name
        response["date_columns"] = date_columns

        return response


@bp.route("/clinical-features", methods=("GET", "POST", "DELETE"))
def clinical_features():

    if request.method == "POST":
        # Differentiate between creating or getting features

        # SAVING FEATURES
        if "clinical_feature_map" in request.json:
            clinical_features_df = load_df_from_request_dict(
                request.json["clinical_feature_map"]
            )
            album_id = get_album_id_from_request(request)

            clinical_feature_definitions = (
                ClinicalFeatureDefinition.find_by_user_id_and_album_id(
                    user_id=g.user, album_id=album_id
                )
            )

            # Save clinical feature values to database
            values_to_insert = []

            for idx, row in clinical_features_df.iterrows():
                for feature in clinical_feature_definitions:
                    values_to_insert.append(
                        {
                            "value": row[feature.name],
                            "clinical_feature_definition_id": feature.id,
                            "patient_id": row[PATIENT_ID_FIELD],
                        }
                    )

            print("Number of feature values to create", len(values_to_insert))

            saved_features = ClinicalFeatureValue.insert_values(values_to_insert)

            return jsonify(saved_features)

        # READING FEATURES
        elif "patient_ids" in request.json:
            patient_ids = request.json["patient_ids"]
            album_id = get_album_id_from_request(request)

            all_features_values = ClinicalFeatureValue.find_by_patient_ids(
                patient_ids, album_id, g.user
            )

            output = defaultdict(lambda: {})
            for feat_value in all_features_values:
                out_dict = feat_value[0].to_dict()
                out_dict.update(feat_value[1].to_dict())

                output[out_dict["patient_id"]][out_dict["name"]] = out_dict["value"]

            return jsonify(output)


@bp.route("/clinical-features-definitions", methods=("GET", "POST", "PATCH", "DELETE"))
def clinical_feature_definitions():

    if request.method == "POST":
        album_id = get_album_id_from_request(request)

        clinical_feature_definitions_to_insert = []

        for feature_name, feature in request.json[
            "clinical_feature_definitions"
        ].items():
            clinical_feature_definitions_to_insert.append(
                {
                    "name": feature_name,
                    "feat_type": feature["feat_type"],
                    "encoding": feature["encoding"],
                    "missing_values": feature["missing_values"],
                    "user_id": g.user,
                    "album_id": album_id,
                }
            )

        print(
            "Number of clinical feature definitions to insrt or update",
            len(clinical_feature_definitions_to_insert),
        )
        saved_definitions = ClinicalFeatureDefinition.insert_values(
            clinical_feature_definitions_to_insert
        )

        return jsonify(saved_definitions)

    if request.method == "PATCH":
        updated_definitions = request.json["clinical_feature_definitions"]

        ClinicalFeatureDefinition.update_values(updated_definitions)

        return jsonify(updated_definitions)

    if request.method == "GET":
        album_id = get_album_id_from_request(request)
        clinical_feature_definitions = (
            ClinicalFeatureDefinition.find_by_user_id_and_album_id(
                user_id=g.user, album_id=album_id
            )
        )

        return jsonify([d.to_dict() for d in clinical_feature_definitions])

    if request.method == "DELETE":
        album_id = get_album_id_from_request(request)
        ClinicalFeatureDefinition.delete_by_user_id_and_album_id(
            g.user, album_id=album_id
        )
        return {"message": "feature definitions deleted successfully"}, 200


@bp.route("/clinical-features-definitions/guess", methods=["POST"])
def guess_clinical_feature_definitions():
    if request.method == "POST":
        response = {}
        clinical_features_df = load_df_from_request_dict(
            request.json["clinical_feature_map"]
        )
        default_categorical = {
            "feat_type": "Categorical",
            "encoding": "One-Hot Encoding",
            "missing_values": "Mode",
        }
        default_numeric = {
            "feat_type": "Number",
            "encoding": "Normalization",
            "missing_values": "Median",
        }
        for column_name in clinical_features_df.columns:
            if column_name == "PatientID" or column_name == "__parsed_extra":
                continue
            if clinical_features_df[column_name].unique().size <= 10:
                response[column_name] = default_categorical
            else:
                try:
                    _ = clinical_features_df[column_name].unique().astype(float)
                    response[column_name] = default_numeric
                except ValueError as e:
                    # If conversion to float fails, assign categorical type
                    response[column_name] = default_categorical

        return response

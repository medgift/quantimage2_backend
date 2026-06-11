import pandas as pd

from typing import Dict
from collections import defaultdict
from flask import Blueprint, jsonify, request, g
from werkzeug.exceptions import BadRequest, NotFound
from quantimage2_backend_common.models import (
    ClinicalFeatureDefinition,
    ClinicalFeatureFile,
    ClinicalFeatureValue,
    db,
)
from quantimage2_backend_common.const import CLINICAL_FEATURE_ID_SEPARATOR
from routes.utils import validate_decorate
from quantimage2_backend_common.models import ClinicalFeatureTypes
from service.clinical_features_dedup import compute_clinical_duplicate_advisories

from service.feature_transformation import PATIENT_ID_FIELD

# Define blueprint
bp = Blueprint("clinical_features", __name__)

NA_VALUE = "N/A"

DEFAULT_CLINICAL_FILE_NAME = "Clinical features"


def _require_file_id(payload: Dict) -> int:
    """Pull and validate `clinical_feature_file_id` from a JSON payload."""
    file_id = (payload or {}).get("clinical_feature_file_id")
    if file_id is None:
        raise BadRequest("clinical_feature_file_id is required for this request")
    try:
        return int(file_id)
    except (TypeError, ValueError):
        raise BadRequest("clinical_feature_file_id must be an integer")


def _load_file_or_404(file_id: int) -> ClinicalFeatureFile:
    """Load a ClinicalFeatureFile and verify it belongs to the current user."""
    f = ClinicalFeatureFile.find_by_id(file_id)
    if f is None or f.user_id != g.user:
        raise NotFound(f"clinical_feature_file {file_id} not found")
    return f


@bp.before_request
def before_request():
    validate_decorate(request)


def load_df_from_request_dict(request_dict: Dict) -> pd.DataFrame:
    clinical_features_list = []

    for patient_id, features in request_dict.items():
        features[PATIENT_ID_FIELD] = patient_id
        clinical_features_list.append(features)

    clinical_features_df = pd.DataFrame(clinical_features_list)
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

            feat_type = feature_types.get(column)
            if feat_type is None:
                # Value map carries a column with no matching definition; skip it
                # rather than KeyError-ing into a 500.
                continue

            if ClinicalFeatureTypes(feat_type) == ClinicalFeatureTypes.CATEGORICAL:
                frequency_of_occurrence = (
                    series.value_counts() / series.value_counts().sum()
                ) * 100

                response[column] = [
                    (
                        f"{value} ({round(percentage, 2)}%)"
                        if len(value) > 0
                        else f"{NA_VALUE} ({round(percentage, 2)}%)"
                    )
                    for value, percentage in frequency_of_occurrence.items()
                ]
            elif ClinicalFeatureTypes(feat_type) == ClinicalFeatureTypes.NUMBER:
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

        # SAVING FEATURES (scoped to a single file)
        if "clinical_feature_map" in request.json:
            clinical_features_df = load_df_from_request_dict(
                request.json["clinical_feature_map"]
            )
            album_id = get_album_id_from_request(request)
            file_id = _require_file_id(request.json)
            _load_file_or_404(file_id)

            clinical_feature_definitions = (
                ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
                    user_id=g.user,
                    album_id=album_id,
                    clinical_feature_file_id=file_id,
                )
            )

            # Only definitions whose column is actually present in the upload.
            present_definitions = [
                feature
                for feature in clinical_feature_definitions
                if feature.name in clinical_features_df.columns
            ]

            # Save clinical feature values to database
            values_to_insert = []

            for idx, row in clinical_features_df.iterrows():
                for feature in present_definitions:
                    values_to_insert.append(
                        {
                            "value": row[feature.name],
                            "clinical_feature_definition_id": feature.id,
                            "patient_id": row[PATIENT_ID_FIELD],
                        }
                    )

            print("Number of feature values to create", len(values_to_insert))

            # Saving values for a file is a replace, not an append: drop any
            # existing values for these definitions first so re-uploading the
            # same file does not accumulate duplicate (patient, definition) rows.
            # commit=False keeps the delete pending until insert_values commits,
            # so a failed insert rolls back the delete too instead of leaving
            # the file wiped.
            ClinicalFeatureValue.delete_by_clinical_feature_definition_ids(
                [feature.id for feature in present_definitions], commit=False
            )
            saved_features = ClinicalFeatureValue.insert_values(values_to_insert)

            return jsonify(saved_features)

        # READING FEATURES
        # Returns nested dict {patient_id: {<file_id>::<name>: value}} so that
        # multiple uploaded CSVs with overlapping column names disambiguate.
        elif "patient_ids" in request.json:
            patient_ids = request.json["patient_ids"]
            album_id = get_album_id_from_request(request)

            all_features_values = ClinicalFeatureValue.find_by_patient_ids(
                patient_ids, album_id, g.user
            )

            output = defaultdict(lambda: {})
            for feat_value in all_features_values:
                value_dict = feat_value[0].to_dict()
                definition_dict = feat_value[1].to_dict()

                key = f"{definition_dict['clinical_feature_file_id']}{CLINICAL_FEATURE_ID_SEPARATOR}{definition_dict['name']}"
                output[value_dict["patient_id"]][key] = value_dict["value"]

            return jsonify(output)


@bp.route("/clinical-features/duplicates", methods=["GET"])
def clinical_features_duplicates():
    """Advisory for feature names duplicated across an album's clinical files.

    Training uses each feature name only once (newest file wins); this returns
    what that dedup means for the data: per duplicated name, which file is
    kept, which are dropped, and whether the drop is harmless (identical) or
    loses/overrides patient values (coverage_loss / conflict).
    """
    album_id = get_album_id_from_request(request)
    return jsonify(compute_clinical_duplicate_advisories(g.user, album_id))


@bp.route("/clinical-features-definitions", methods=("GET", "POST", "PATCH", "DELETE"))
def clinical_feature_definitions():

    if request.method == "POST":
        album_id = get_album_id_from_request(request)
        file_id = _require_file_id(request.json)
        _load_file_or_404(file_id)

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
                    "clinical_feature_file_id": file_id,
                }
            )

        print(
            "Number of clinical feature definitions to insrt or update",
            len(clinical_feature_definitions_to_insert),
        )
        # bulk_insert_mappings does not backfill ids, so we re-query below
        # rather than using this call's return value.
        ClinicalFeatureDefinition.insert_values(clinical_feature_definitions_to_insert)

        # Return the freshly-saved rows (with their auto-assigned ids) so the
        # frontend can wire up a definition->id mapping immediately.
        saved = ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
            user_id=g.user, album_id=album_id, clinical_feature_file_id=file_id
        )
        return jsonify([d.to_dict() for d in saved])

    if request.method == "PATCH":
        updated_definitions = (request.json or {}).get("clinical_feature_definitions")
        if not updated_definitions:
            return {"message": "clinical_feature_definitions is required"}, 400

        ids = [d.get("id") for d in updated_definitions]
        if any(i is None for i in ids):
            return {"message": "every clinical feature definition requires an id"}, 400

        # Only let a user update definitions they own (prevents IDOR via id).
        owned = ClinicalFeatureDefinition.find_owned_ids(g.user, ids)
        missing = [i for i in ids if i not in owned]
        if missing:
            raise NotFound(f"clinical feature definitions not found: {missing}")

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
        # Per-album wipe is no longer supported now that uploads append.
        # Use DELETE /clinical-features-files/<id> to remove a single file.
        return (
            {
                "message": (
                    "DELETE on /clinical-features-definitions is disabled; "
                    "delete a specific file via /clinical-features-files/<id>."
                )
            },
            410,
        )


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


def _unique_file_name(user_id: str, album_id: str, base_name: str) -> str:
    """Return a name unique within (user_id, album_id), suffixing ' (n)' if needed."""
    name = base_name or DEFAULT_CLINICAL_FILE_NAME
    existing = {
        f.name
        for f in ClinicalFeatureFile.find_by_user_id_and_album_id(user_id, album_id)
    }
    if name not in existing:
        return name
    counter = 2
    while f"{name} ({counter})" in existing:
        counter += 1
    return f"{name} ({counter})"


@bp.route("/clinical-features-files", methods=("GET", "POST"))
def clinical_feature_files():
    """List or create clinical-feature CSV files for an album."""
    album_id = get_album_id_from_request(request)

    if request.method == "GET":
        files = ClinicalFeatureFile.find_by_user_id_and_album_id(
            user_id=g.user, album_id=album_id
        )
        return jsonify([f.to_dict() for f in files])

    if request.method == "POST":
        requested_name = (request.json or {}).get("name")
        name = _unique_file_name(g.user, album_id, requested_name)
        f = ClinicalFeatureFile(name=name, album_id=album_id, user_id=g.user)
        f.save_to_db()
        return jsonify(f.to_dict()), 201


@bp.route("/clinical-features-files/<int:file_id>", methods=("PATCH", "DELETE"))
def clinical_feature_file_detail(file_id: int):
    f = _load_file_or_404(file_id)

    if request.method == "PATCH":
        new_name = (request.json or {}).get("name")
        if not new_name or not isinstance(new_name, str):
            return {"message": "`name` is required"}, 400
        # Reject with 409 if a sibling file already has the requested name
        existing = ClinicalFeatureFile.find_by_user_id_album_id_and_name(
            g.user, f.album_id, new_name
        )
        if existing is not None and existing.id != f.id:
            return {"message": f"A file named '{new_name}' already exists"}, 409
        f.update(name=new_name)
        return jsonify(f.to_dict())

    if request.method == "DELETE":
        # ClinicalFeatureDefinition has ON DELETE CASCADE so values are removed
        # transitively (definition -> value cascade is already in place).
        ClinicalFeatureFile.delete_by_id(f.id)
        return {"message": "clinical feature file deleted"}, 200

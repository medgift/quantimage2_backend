import os
import traceback
import json
from typing import List, Dict

from collections import defaultdict
from flask import Blueprint, jsonify, request, g
from quantimage2_backend_common.models import ClinicalFeatureDefinition, ClinicalFeatureValue
import pandas as pd
from routes.utils import validate_decorate


# Define blueprint
bp = Blueprint(__name__, "clinical_features")


@bp.before_request
def before_request():
    validate_decorate(request)

def load_df_from_request_dict(request_dict: Dict) -> pd.core.frame.DataFrame:
    clinical_features_list = []

    for patient_id, features in request_dict.items():
        features["Patient ID"] = patient_id
        clinical_features_list.append(features)

    return pd.DataFrame.from_dict(clinical_features_list)

def get_album_id_from_request(request):
    album_id = request.args.get("album_id")

    if not album_id:
        raise ValueError("album_id is required for this request")

    return album_id

@bp.route("/clinical_features/get_unique_values", methods=["POST"])
def clinical_features_get_unique_values():
    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])

        response = {"frequency_of_occurence": {}}

        #Computing features with no data at all (using strings because we are not guarantee to get nulls from the request)
        for column in clinical_features_df.columns:
            frequency_of_occurence = (clinical_features_df[column].value_counts() / clinical_features_df[column].value_counts().sum()) * 100
            response["frequency_of_occurence"][column] = [f"{idx}-{round(i, 2)}%" for idx, i in frequency_of_occurence.items()]

        return response
    

@bp.route("/clinical_features/filter", methods=["POST"])
def clinical_features_filter():
    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])
        nulls_df = pd.DataFrame()

        response = {}
        response["only_one_value"] = []
        date_columns = []

        #Computing features with no data at all (using strings because we are not guarantee to get nulls from the request)
        for column in clinical_features_df.columns:
            if column == "__parsed_extra": # This happens when papa parse encounters some funkyness in the CSV
                continue
            nulls_df[column] = clinical_features_df[column].astype(str).apply(lambda x: len(x)) # we first create a dataframe with the same shape as the clinical features - but with the length of the string in each cell - len == 0 -> no data.

            # Number of unique values per feature
            n_unique = clinical_features_df[column].unique()
            if len(n_unique) == 1:
                response["only_one_value"].append(column)

            if "date" in column.lower():
                date_columns.append(column)


        columns_with_only_nulls = (nulls_df == 0).sum() == len(clinical_features_df)
        response["only_nulls"] = columns_with_only_nulls[columns_with_only_nulls].index.tolist()

        # Dropping features with too little data
        percent_nulls = ((nulls_df == 0).sum() / len(clinical_features_df)) >= 0.9
        response["too_little_data"] = percent_nulls[percent_nulls].index.tolist()

        # Columns that have date in the name
        response["date_columns"] = date_columns
        
        return response


@bp.route("/clinical_features", methods=("GET", "POST", "DELETE"))
def clinical_features():

    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])
        album_id = get_album_id_from_request(request)

        clinical_feature_definitions = ClinicalFeatureDefinition.find_by_user_id_and_album_id(user_id=g.user, album_id=album_id)

        saved_features = []

        # Save clinical feature values to database
        values_to_insert_or_update = []

        for idx, row in clinical_features_df.iterrows():
            for feature in clinical_feature_definitions:
                values_to_insert_or_update.append({"value": row[feature.name], "clinical_feature_definition_id": feature.id, "patient_id": row["Patient ID"]})
    
        ClinicalFeatureValue.insert_values(values_to_insert_or_update)

        return jsonify([i.to_dict() for i in saved_features])

    if request.method == "GET":
        patient_ids = request.args.get("patient_ids").split(",")
        album_id = get_album_id_from_request(request)

        all_features_values = ClinicalFeatureValue.find_by_patient_ids(patient_ids, album_id, g.user)

        output = defaultdict(lambda: {})
        for feat_value in all_features_values:
            out_dict = feat_value[0].to_dict()
            out_dict.update(feat_value[1].to_dict())

            output[out_dict["patient_id"]][out_dict["Name"]] = out_dict["value"]

        return jsonify(output)

@bp.route("/clinical_feature_definitions", methods=("GET", "POST", "DELETE"))
def clinical_feature_definitions():
    
    if request.method == "POST":
        created_features = []
        album_id = get_album_id_from_request(request)
        
        for feature_name, feature in request.json["clinical_feature_definitions"].items():
            feature_model = ClinicalFeatureDefinition.insert(name=feature_name, feat_type=feature["Type"], encoding=feature["Encoding"], user_id=g.user, album_id=album_id)
            created_features.append(feature_model)

        return jsonify([i.to_dict() for i in created_features])

    if request.method == "GET":
        album_id = get_album_id_from_request(request)
        clinical_feature_definitions = ClinicalFeatureDefinition.find_by_user_id_and_album_id(user_id=g.user, album_id=album_id)
        output = {}
        for feature in clinical_feature_definitions:
            output[feature.name] = feature.to_dict()

        return jsonify(output)
    
    if request.method == "DELETE":
        album_id = get_album_id_from_request(request)
        ClinicalFeatureDefinition.delete_by_user_id_and_album_id(g.user, album_id=album_id)
        return '', 200


@bp.route("/clinical_feature_definitions/guess", methods=["POST"])
def guess_clinical_feature_definitions():
    if request.method == "POST":
        response = {}
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])
        for column_name in clinical_features_df.columns:
            if column_name == "PatientID" or column_name == "__parsed_extra":
                continue
            if clinical_features_df[column_name].unique().size <= 10:
                response[column_name] = {"Type": "Categorical", "Encoding": "One-Hot Encoding"} # The strings here should be the same as the ones used by the frontend (src/config/constants.js - line 79 as of 20th june 2023)
            try:
                _ = clinical_features_df[column_name].unique().astype(float)
                response[column_name] = {"Type": "Float", "Encoding": "Normalization"}
            except:
                pass
    
        return response
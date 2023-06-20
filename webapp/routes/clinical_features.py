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
    print("in the validation")
    validate_decorate(request)

def load_df_from_request_dict(request_dict: Dict) -> pd.core.frame.DataFrame:
    clinical_features_list = []

    for patient_id, features in request_dict.items():
        features["Patient ID"] = patient_id
        clinical_features_list.append(features)

    return pd.DataFrame.from_dict(clinical_features_list)


@bp.route("/clinical_features/filter", methods=["POST"])
def clinical_features_filter():
    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])
        nulls_df = pd.DataFrame()

        response = {}

        #Computing rows with no data at all (using strings because we are not guarantee to get nulls from the request)
        for column in clinical_features_df.columns:
            nulls_df[column] = clinical_features_df[column].astype(str).apply(lambda x: len(x)) # we first create a dataframe with the same shape as the clinical features - but with the length of the string in each cell - len == 0 -> no data.

        columns_with_only_nulls = (nulls_df == 0).sum() == len(clinical_features_df)
        response["only_nulls"] = columns_with_only_nulls[columns_with_only_nulls].index.tolist()

        # Dropping rows with too little data
        percent_nulls = ((nulls_df == 0).sum() / len(clinical_features_df)) >= 0.9
        response["too_little_data"] = percent_nulls[percent_nulls].index.tolist()

        return response


@bp.route("/clinical_features", methods=("GET", "POST", "DELETE"))
def clinical_features():

    if request.method == "POST":
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])

        clinical_feature_definitions = ClinicalFeatureDefinition.find_by_user_id(user_id=g.user)

        saved_features = []

        # Save clinical feature values to database
        for idx, row in clinical_features_df.iterrows():
            for feature in clinical_feature_definitions:
                feature_name = feature.name
                patient_id = row["Patient ID"]

                val = ClinicalFeatureValue.insert_value(value=row[feature_name], clinical_feature_definition_id=feature.id, patient_id=patient_id)
                saved_features.append(val)

        print("saved features 0", saved_features[0])
        print("saved features 0 type", (saved_features[0]))
        return jsonify([i.to_dict() for i in saved_features])

    if request.method == "GET":
        patient_ids = request.args.get("patient_ids").split(",")
        all_features_values = ClinicalFeatureValue.find_by_patient_ids(patient_ids, g.user)

        output = defaultdict(lambda: {})
        for feat_value in all_features_values:
            out_dict = feat_value[0].to_dict()
            out_dict.update(feat_value[1].to_dict())

            print(feat_value[0].to_dict())

            output[out_dict["patient_id"]][out_dict["Name"]] = out_dict["value"]

        return jsonify(output)

    if request.method == "DELETE":
        ClinicalFeatureValue.delete_by_user_id(g.user)
        return '', 200

@bp.route("/clinical_feature_definitions", methods=("GET", "POST", "DELETE"))
def clinical_feature_definitions():
    
    if request.method == "POST":
        created_features = []
        print("Json content of the request", request.json["clinical_feature_definitions"])
        for feature_name, feature in request.json["clinical_feature_definitions"].items():
            feature_model = ClinicalFeatureDefinition.insert(name=feature_name, feat_type=feature["Type"], encoding=feature["Encoding"], user_id=g.user)
            created_features.append(feature_model)

        return jsonify([i.to_dict() for i in created_features])

    if request.method == "GET":
        clinical_feature_definitions = ClinicalFeatureDefinition.find_by_user_id(user_id=g.user)
        output = {}
        for feature in clinical_feature_definitions:
            output[feature.name] = feature.to_dict()

        return jsonify(output)
    
    if request.method == "DELETE":
        ClinicalFeatureDefinition.delete_by_user_id(g.user)
        return '', 200


@bp.route("/clinical_feature_definitions/guess", methods=["POST"])
def guess_clinical_feature_definitions():
    if request.method == "POST":
        response = {}
        clinical_features_df = load_df_from_request_dict(request.json["clinical_feature_map"])
        for column_name in clinical_features_df.columns:
            if column_name == "PatientID":
                continue
            if clinical_features_df[column_name].unique().size < 5:
                response[column_name] = {"Type": "Categorical", "Encoding": "One-Hot Encoding"} # The strings here should be the same as the ones used by the frontend (src/config/constants.js - line 79 as of 20th june 2023)
            try:
                _ = clinical_features_df[column_name].unique().astype(float)
                response[column_name] = {"Type": "Float", "Encoding": "Normalization"}
            except:
                pass
    
        return response
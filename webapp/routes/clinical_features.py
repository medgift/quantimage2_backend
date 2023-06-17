import os
import traceback
import json
from typing import List

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


@bp.route("/clinical_features", methods=("GET", "POST", "DELETE"))
def clinical_features():

    if request.method == "POST":
        response = {}
        clinical_features = request.json["clinical_feature_map"]
        clinical_features_list = []

        for patient_id, features in clinical_features.items():
            features["Patient ID"] = patient_id
            clinical_features_list.append(features)

        clinical_features_df = pd.DataFrame.from_dict(clinical_features_list)

        clinical_feature_definitions = ClinicalFeatureDefinition.find_by_user_id(user_id=g.user)

        saved_features = []

        # Save clinical feature values to database
        for idx, row in clinical_features_df.iterrows():
            for feature in clinical_feature_definitions:
                feature_name = feature.name
                patient_id = row["Patient ID"]

                val = ClinicalFeatureValue.insert_value(value=row[feature_name], clinical_feature_definition_id=feature.id, patient_id=row["Patient ID"])
                saved_features.append(val)

        return jsonify([i.to_dict() for i in saved_features])

    if request.method == "GET":
        patient_ids = request.args.get("patient_ids").split(",")
        all_features_values = ClinicalFeatureValue.find_by_patient_ids(patient_ids, g.user)

        output = defaultdict(lambda: {})
        for feat_value in all_features_values:
            out_dict = feat_value[0].to_dict()
            out_dict.update(feat_value[1].to_dict())

            output[out_dict["patient_id"]][out_dict["name"]] = out_dict["value"]

        return jsonify(output)

    if request.method == "DELETE":
        ClinicalFeatureValue.delete_by_user_id(g.user)

@bp.route("/clinical_feature_definitions", methods=("GET", "POST", "DELETE"))
def clinical_feature_definitions():
    
    if request.method == "POST":
        created_features = []
        for feature_name, feature in request.json["clinical_feature_definitions"].items():
            print(feature_name, feature)
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

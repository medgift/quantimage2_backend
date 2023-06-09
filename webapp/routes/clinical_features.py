import os
import traceback
import json
from typing import List

from sqlalchemy.orm import joinedload
from flask import Blueprint, jsonify, request, g, make_response
from quantimage2_backend_common.models import ClinicalFeatureDefinition, ClinicalFeatureValue
import pandas as pd
from routes.utils import validate_decorate
from service.classification import train_classification_model
from service.survival import train_survival_model


# Define blueprint
bp = Blueprint(__name__, "clinical_features")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/clinical_features", methods=("GET", "POST"))
def clinical_features():

    print(request.method)

    if request.method == "POST":
        response = {}
        clinical_features = request.json["clinical_feature_map"]
        clinical_features_list = []

        for patient_id, features in clinical_features.items():
            features["Patient ID"] = patient_id
            clinical_features_list.append(features)

        clinical_features_df = pd.DataFrame.from_dict(clinical_features_list)

        feature_names = [i for i in clinical_features_df.columns if i != "Patient ID"]
        response["feature_names"] = feature_names

        saved_features: List[ClinicalFeatureDefinition] = []

        # Save clinical feature definitions to database if they don't already exist
        for feature in feature_names:
            already_in_db = ClinicalFeatureDefinition.find_by_name([feature], user_id=g.user)
            
            if len(already_in_db) > 0:
                saved_features.append(already_in_db[0])
                continue
            feature_model = ClinicalFeatureDefinition(name=feature, user_id=g.user)
            feature_model.save_to_db()

            saved_features.append(feature_model)

        # Save clinical feature values to database
        for idx, row in clinical_features_df.iterrows():
            for feature in saved_features:
                feature_name = feature.name
                patient_id = row["Patient ID"]

                ClinicalFeatureValue.insert_value(value=row[feature_name], clinical_feature_definition_id=feature.id, patient_id=row["Patient ID"])

        return jsonify([i.to_dict() for i in saved_features])

    if request.method == "GET":
        patient_id = request.args.get("patient_id")
        clinical_feature_name = request.args.get("clinical_feature_name")
        print(clinical_feature_name, patient_id)
        return ClinicalFeatureValue.find_by_patient_id_and_name(patient_id, clinical_feature_name, g.user).to_dict()
import os
import traceback

from sqlalchemy.orm import joinedload
from flask import Blueprint, jsonify, request, g, make_response
from quantimage2_backend_common.models import Model, LabelCategory
from routes.utils import validate_decorate
from service.classification import train_classification_model
from service.survival import train_survival_model


# Define blueprint
bp = Blueprint(__name__, "clinical_features")


@bp.route("/clinical_features", methods=("GET", "POST"))
def clinical_features():

    print(request.method)

    if request.method == "POST":
        print(request["data"])
        print("I am in the POSt blck")
    print("hello")
    return {"aasdf": "basdf", 1: 2, 3: 4}


@bp.route("/clinical_feature_values", methods=("GET", "POST"))
def clinical_feature_values():
    print("hello")
    return {"a": "b"}

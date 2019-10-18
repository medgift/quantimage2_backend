import os

from flask import Blueprint, abort, jsonify, request, g, current_app

from ..models import FeatureFamily
from .utils import validate_decorate


# Define blueprint
bp = Blueprint(__name__, "feature_families")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/feature-families", methods=("GET", "POST"))
def create_feature_family():
    if request.method == "POST":
        # print(request)

        if request.files["file"] is None or not allowed_file(request.files["file"]):
            abort(400)

        file = request.files["file"]
        name = request.form["name"]

        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)

        file.save(file_path)

        # Create FeatureFamily object and send it back

        return jsonify("ok")

    if request.method == "GET":
        families = FeatureFamily.find_all()
        return jsonify(families)


def allowed_file(file):
    return file.content_type == "application/x-yaml"

import os
from pathlib import Path

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
def feature_families():
    if request.method == "POST":
        # print(request)

        if request.files["file"] is None or not allowed_file(request.files["file"]):
            abort(400)

        file = request.files["file"]

        file_path = save_feature_config(file)

        name = request.form["name"]

        # Create FeatureFamily object and send it back
        feature_family = FeatureFamily(name, file_path)
        feature_family.save_to_db()

        return jsonify(feature_family.to_dict())

    if request.method == "GET":
        feature_families = FeatureFamily.find_all()

        serialized_families = map(
            lambda family: format_family(family), feature_families
        )

        return jsonify(list(serialized_families))


@bp.route("/feature-families/<feature_family_id>", methods=("GET", "PATCH"))
def feature_family(feature_family_id):

    if request.method == "PATCH":

        feature_family = FeatureFamily.find_by_id(feature_family_id)

        feature_family.name = request.form["name"]

        if request.files.get("file") is not None:
            file = request.files["file"]

            file_path = save_feature_config(file)

            feature_family.config_path = file_path

        feature_family.save_to_db()

        return jsonify(feature_family.to_dict())

    if request.method == "GET":
        feature_family = FeatureFamily.find_by_id(feature_family_id)

        return jsonify(feature_family.to_dict())


def allowed_file(file):
    return file.content_type == "application/x-yaml"


def save_feature_config(file):
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)

    file.save(file_path)

    return file_path


def format_family(family):
    formatted_dict = family.to_dict()

    formatted_dict["config"] = Path(family.config_path).read_text()

    return formatted_dict

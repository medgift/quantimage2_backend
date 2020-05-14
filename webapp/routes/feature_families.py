import os
from pathlib import Path

import yaml
from flask import Blueprint, abort, jsonify, request, current_app

from imaginebackend_common.models import FeatureFamily
from .utils import validate_decorate, role_required
from imaginebackend_common.feature_backends import feature_backends_map


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
        return create_feature_family()

    if request.method == "GET":
        feature_families = FeatureFamily.find_all()

        serialized_families = list(
            map(lambda family: format_family(family), feature_families)
        )

        return jsonify(serialized_families)


@bp.route("/feature-families/<feature_family_id>", methods=("GET", "PATCH"))
def feature_family(feature_family_id):

    if request.method == "PATCH":
        return update_feature_family(feature_family_id)

    if request.method == "GET":
        feature_family = FeatureFamily.find_by_id(feature_family_id)

        return jsonify(feature_family.to_dict())


@role_required(os.environ["KEYCLOAK_FRONTEND_ADMIN_ROLE"])
def create_feature_family():
    if request.files["file"] is None or not allowed_file(request.files["file"]):
        abort(400)

    file = request.files["file"]

    file_path = save_feature_config(file)

    name = request.form["name"]

    # Create FeatureFamily object and send it back
    feature_family = FeatureFamily(name, file_path)
    feature_family.save_to_db()

    return jsonify(feature_family.to_dict())


@role_required(os.environ["KEYCLOAK_FRONTEND_ADMIN_ROLE"])
def update_feature_family(feature_family_id):
    feature_family = FeatureFamily.find_by_id(feature_family_id)

    feature_family.name = request.form["name"]

    if request.files.get("file") is not None:
        file = request.files["file"]

        file_path = save_feature_config(file)

        feature_family.config_path = file_path

    feature_family.save_to_db()

    return jsonify(feature_family.to_dict())


def allowed_file(file):
    return file.content_type == "application/x-yaml" or file.content_type == "application/octet-stream"


def save_feature_config(file):
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)

    file.save(file_path)

    return file_path


def format_family(family):
    formatted_dict = family.to_dict()

    feature_config_yaml = yaml.load(
        Path(family.config_path).read_text(), Loader=yaml.FullLoader
    )

    config = {"backends": {}}

    for feature_backend in feature_config_yaml["backends"]:
        feature_backend_object = feature_backends_map[feature_backend](
            feature_config_yaml["backends"][feature_backend]
        )
        config["backends"][feature_backend] = feature_backend_object.format_config()

    formatted_dict["config"] = config

    return formatted_dict

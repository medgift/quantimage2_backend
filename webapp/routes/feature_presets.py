import os
from pathlib import Path

import yaml
from flask import Blueprint, abort, jsonify, request, current_app

from quantimage2_backend_common.models import FeaturePreset
from .utils import validate_decorate, role_required

# Define blueprint
bp = Blueprint(__name__, "feature_presets")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/feature-presets", methods=("GET", "POST"))
def feature_presets():
    if request.method == "POST":
        return create_feature_preset()

    if request.method == "GET":
        feature_presets = FeaturePreset.find_all()

        serialized_presets = list(
            map(lambda preset: format_preset(preset), feature_presets)
        )

        return jsonify(serialized_presets)


@bp.route("/feature-presets/<feature_preset_id>", methods=("GET", "PATCH"))
def feature_preset(feature_preset_id):

    if request.method == "PATCH":
        return update_feature_preset(feature_preset_id)

    if request.method == "GET":
        feature_preset = FeaturePreset.find_by_id(feature_preset_id)

        return jsonify(format_preset(feature_preset))


@role_required(os.environ["KEYCLOAK_FRONTEND_ADMIN_ROLE"])
def create_feature_preset():
    if request.files["file"] is None or not allowed_file(request.files["file"]):
        abort(400)

    file = request.files["file"]

    file_path = save_feature_config(file)

    name = request.form["name"]

    # Create FeaturePreset object and send it back
    feature_preset = FeaturePreset(name, file_path)
    feature_preset.save_to_db()

    return jsonify(format_preset(feature_preset))


@role_required(os.environ["KEYCLOAK_FRONTEND_ADMIN_ROLE"])
def update_feature_preset(feature_preset_id):
    feature_preset = FeaturePreset.find_by_id(feature_preset_id)

    feature_preset.name = request.form["name"]

    if request.files.get("file") is not None:
        file = request.files["file"]

        file_path = save_feature_config(file)

        feature_preset.config_path = file_path

    feature_preset.save_to_db()

    return jsonify(format_preset(feature_preset))


def allowed_file(file):
    return (
        file.content_type == "application/x-yaml"
        or file.content_type == "application/octet-stream"
    )


def save_feature_config(file):
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)

    file.save(file_path)

    return file_path


def format_preset(preset):
    formatted_dict = preset.to_dict()

    config_yaml = Path(preset.config_path).read_text()

    feature_config_yaml = yaml.load(config_yaml, Loader=yaml.FullLoader)

    formatted_dict["config"] = feature_config_yaml
    formatted_dict["config-raw"] = config_yaml

    return formatted_dict


# def format_family(family):
#     formatted_dict = family.to_dict()
#
#     feature_config_yaml = yaml.load(
#         Path(family.config_path).read_text(), Loader=yaml.FullLoader
#     )
#
#     config = {"backends": {}}
#
#     for feature_backend in feature_config_yaml["backends"]:
#         feature_backend_object = feature_backends_map[feature_backend](
#             feature_config_yaml["backends"][feature_backend]
#         )
#         config["backends"][feature_backend] = feature_backend_object.format_config()
#
#     formatted_dict["config"] = config
#
#     return formatted_dict

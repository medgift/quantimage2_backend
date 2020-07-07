import os

import collections

import jsonpickle

from flask import Blueprint, jsonify, request, g, current_app, Response

from config import MODELS_BASE_DIR
from imaginebackend_common.models import Model
from pathlib import Path

# Define blueprint
from imaginebackend_common.utils import format_extraction
from routes.utils import validate_decorate
from service.feature_analysis import train_model_with_metric

bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


def format_model(model):
    model_dict = model.to_dict()

    # Format feature extraction to get feature n° and names
    formatted_extraction = format_extraction(model.feature_extraction, families=True)

    model_dict["extraction"] = formatted_extraction
    model_dict["feature-number"] = formatted_extraction["feature-number"]
    model_dict["feature-names"] = formatted_extraction["feature-names"]

    # De-serialize model (for metrics)
    f = open(model.model_path)
    json_str = f.read()
    model_object = jsonpickle.decode(json_str)

    # Convert metrics to native Python types
    metrics = collections.OrderedDict()
    for metric_name, metric_value in model_object.metrics.items():
        metrics[metric_name] = metric_value.item()

    model_dict["metrics"] = metrics

    return model_dict


@bp.route("/models/<album_id>", methods=("GET", "POST"))
def models_by_album(album_id):

    if request.method == "GET":
        models = Model.find_by_album(album_id, g.user)
        formatted_models = list(map(lambda model: format_model(model), models))
        return jsonify(formatted_models)

    if request.method == "POST":
        # Dictionary with the key corresponding
        # to a MODALITY + ROI combination and
        # the value being a JSON representation
        # of the features (including patient ID)
        body = request.json

        feature_extraction_id = body["extraction-id"]
        studies = body["studies"]
        album = body["album"]
        gt = body["labels"]
        model_type = body["model-type"]
        algorithm_type = body["algorithm-type"]
        modalities = body["modalities"]
        rois = body["rois"]

        model = train_model_with_metric(
            feature_extraction_id, studies, algorithm_type, modalities, rois, gt
        )

        model_path = get_model_path(
            g.user, album["album_id"], model_type, algorithm_type, modalities, rois
        )

        # Persist model in DB and on disk (pickle it)
        json_model = jsonpickle.encode(model)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Path(model_path).write_text(json_model)

        # Generate model name (for now)
        (file, ext) = os.path.splitext(os.path.basename(model_path))
        model_name = f"{album['name']}_{file}"

        db_model = Model(
            model_name,
            model_type,
            algorithm_type,
            modalities,
            rois,
            model_path,
            g.user,
            album["album_id"],
            feature_extraction_id,
        )
        db_model.save_to_db()

        return jsonify(format_model(db_model))


@bp.route("/models/<id>", methods=["DELETE"])
def model(id):
    the_model = Model.delete_by_id(id)
    formatted_model = format_model(the_model)
    model_path = the_model.model_path
    try:
        os.unlink(model_path)
    except FileNotFoundError as e:
        pass
    return jsonify(formatted_model)


@bp.route("/models")
def models_by_user():
    albums = Model.find_by_user(g.user)
    return jsonify(albums)


def get_model_path(user_id, album_id, model_type, algorithm_type, modalities, rois):
    # Define features path for storing the results
    models_dir = os.path.join(MODELS_BASE_DIR, user_id, album_id)
    models_filename = f"model_{model_type}_{algorithm_type}_{'-'.join(modalities)}_{'-'.join(rois)}.json"
    models_path = os.path.join(models_dir, models_filename)

    return models_path

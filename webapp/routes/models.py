import os

import jsonpickle

from flask import Blueprint, jsonify, request, g, current_app, Response

from config import MODELS_BASE_DIR
from imaginebackend_common.models import Model
from pathlib import Path

# Define blueprint
from routes.utils import validate_decorate
from service.feature_analysis import train_model_with_metric

bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/models/<album_id>", methods=("GET", "POST"))
def models_by_album(album_id):

    if request.method == "GET":
        albums = Model.find_by_album(album_id, g.user)
        formatted_albums = list(map(lambda album: album.to_dict(), albums,))
        return jsonify(formatted_albums)

    if request.method == "POST":
        # Dictionary with the key corresponding
        # to a MODALITY + ROI combination and
        # the value being a JSON representation
        # of the features (including patient ID)
        body = request.json

        extraction_id = body["extraction-id"]
        studies = body["studies"]
        album = body["album"]
        gt = body["labels"]
        model_type = body["model-type"]
        algorithm_type = body["algorithm-type"]

        model = train_model_with_metric(extraction_id, studies, album, gt)

        model_path = get_models_path(
            g.user, album["album_id"], model_type, algorithm_type
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
            model_path,
            g.user,
            album["album_id"],
        )
        db_model.save_to_db()

        return jsonify(db_model.to_dict())


@bp.route("/models/<id>", methods=["DELETE"])
def model(id):
    the_model = Model.delete_by_id(id)
    model_path = the_model.model_path
    try:
        os.unlink(model_path)
    except FileNotFoundError as e:
        pass
    return jsonify(the_model.to_dict())


@bp.route("/models")
def models_by_user():
    albums = Model.find_by_user(g.user)
    return jsonify(albums)


def get_models_path(user_id, album_id, model_type, algorithm_type):
    # Define features path for storing the results
    models_dir = os.path.join(MODELS_BASE_DIR, user_id, album_id)
    models_filename = f"model_{model_type}_{algorithm_type}.json"
    models_path = os.path.join(models_dir, models_filename)

    return models_path

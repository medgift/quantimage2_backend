import os

import collections

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd

from time import time

from sqlalchemy.orm import joinedload

jsonpickle_pd.register_handlers()

from imaginebackend_common.const import MODEL_TYPES

from flask import Blueprint, jsonify, request, g, current_app, Response

from config import MODELS_BASE_DIR
from imaginebackend_common.models import Model, FeatureExtraction, FeatureCollection
from imaginebackend_common.models import Label
from pathlib import Path

# Define blueprint
from imaginebackend_common.utils import format_extraction
from routes.utils import validate_decorate
from service.feature_analysis import train_model_with_metric
from service.survival_analysis import train_survival_model

bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


def format_model(model):

    model_dict = model.to_dict()

    # De-serialize model (for metrics etc.)
    f = open(model.model_path)
    json_str = f.read()
    model_object = jsonpickle.decode(json_str)

    # Convert metrics to native Python types
    if MODEL_TYPES(model.type) == MODEL_TYPES.CLASSIFICATION:
        metrics = model_object.metrics
    elif MODEL_TYPES(model.type) == MODEL_TYPES.SURVIVAL:
        metrics = collections.OrderedDict()
        metrics["concordance_index"] = model_object.concordance_index_
        # metrics["events_observed"] = len(model_object.event_observed)
    else:
        raise NotImplementedError

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
        collection_id = body["collection-id"]
        studies = body["studies"]
        album = body["album"]
        gt = body["labels"]
        model_type = body["model-type"]
        algorithm_type = body["algorithm-type"]
        validation_strategy = None
        data_normalization = body["data-normalization"]
        feature_selection = None

        if collection_id:
            feature_collection = FeatureCollection.find_by_id(collection_id)
            formatted_collection = feature_collection.format_collection()
            modalities = formatted_collection["modalities"]
            rois = formatted_collection["rois"]
            feature_names = formatted_collection["features"]
        else:
            feature_extraction = FeatureExtraction.find_by_id(feature_extraction_id)
            modalities = list(map(lambda m: m.name, feature_extraction.modalities))
            rois = list(map(lambda r: r.name, feature_extraction.rois))
            feature_names = list(
                map(lambda f: f.name, feature_extraction.feature_definitions)
            )

        if MODEL_TYPES(model_type) == MODEL_TYPES.CLASSIFICATION:
            model, validation_strategy, validation_params = train_model_with_metric(
                feature_extraction_id,
                collection_id,
                studies,
                algorithm_type,
                data_normalization,
                gt,
            )

            model_path = get_model_path(
                g.user, album["album_id"], model_type, algorithm_type, modalities, rois
            )
        elif MODEL_TYPES(model_type) == MODEL_TYPES.SURVIVAL:
            model, feature_selection, feature_names = train_survival_model(
                feature_extraction_id, collection_id, studies, gt
            )

            model_path = get_model_path(
                g.user, album["album_id"], model_type, None, modalities, rois
            )
        else:
            raise NotImplementedError

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
            f"{validation_strategy} ({validation_params['k']} folds, {validation_params['n']} repetitions)"
            if validation_strategy
            else None,
            data_normalization,
            feature_selection,
            feature_names,
            modalities,
            rois,
            model_path,
            g.user,
            album["album_id"],
            feature_extraction_id,
            collection_id,
        )
        db_model.save_to_db()

        return jsonify(format_model(db_model))


@bp.route("/models/<id>", methods=["DELETE"])
def model(id):
    the_model = Model.delete_by_id(id, options=joinedload(Model.feature_extraction))
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

    if algorithm_type:
        models_filename = f"model_{model_type}_{algorithm_type}_{'-'.join(modalities)}_{'-'.join(rois)}"
    else:
        models_filename = f"model_{model_type}_{'-'.join(modalities)}_{'-'.join(rois)}"

    models_filename += f"_{str(int(time()))}"
    models_filename += ".json"
    models_path = os.path.join(models_dir, models_filename)

    return models_path

import math
import os
import joblib

import traceback

import collections

from time import time

import numpy as np
from sqlalchemy.orm import joinedload

from imaginebackend_common.const import MODEL_TYPES, DATA_SPLITTING_TYPES

from flask import Blueprint, jsonify, request, g, make_response

from config import MODELS_BASE_DIR
from imaginebackend_common.models import (
    Model,
    FeatureExtraction,
    FeatureCollection,
    LabelCategory,
)

# Define blueprint
from imaginebackend_common.utils import format_extraction
from routes.utils import validate_decorate
from service.feature_analysis import train_classification_model
from service.survival_analysis import train_survival_model

bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


def format_model(model):

    model_dict = model.to_dict()

    # De-serialize model # TODO - May not be required anymore if the concordance index is saved in the DB directly
    model_object = joblib.load(model.model_path)

    # Convert metrics to native Python types
    if MODEL_TYPES(model.label_category.label_type) == MODEL_TYPES.CLASSIFICATION:
        final_metrics = model.metrics
        for metric in final_metrics:

            # Do we have a range or just a single value for the metric?
            if np.isscalar(final_metrics[metric]):
                if math.isnan(final_metrics[metric]):
                    final_metrics[metric] = "N/A"
            else:
                for value in final_metrics[metric]:
                    if math.isnan(final_metrics[metric][value]):
                        final_metrics[metric][value] = "N/A"

        metrics = final_metrics
    elif MODEL_TYPES(model.label_category.label_type) == MODEL_TYPES.SURVIVAL:
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

        label_category = LabelCategory.find_by_id(body["label-category-id"])

        feature_extraction_id = body["extraction-id"]
        collection_id = body["collection-id"]
        studies = body["studies"]
        album = body["album"]
        gt = body["labels"]
        data_splitting_type = body["data-splitting-type"]
        train_test_split_type = body["train-test-split-type"]
        training_patients = body["training-patients"]
        test_patients = body["test-patients"]
        training_validation = None
        test_validation = None
        feature_selection = None

        if collection_id:
            feature_collection = FeatureCollection.find_by_id(collection_id)
            formatted_collection = feature_collection.format_collection(
                with_values=True
            )
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

        try:
            if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION:
                (
                    trained_model,
                    training_validation,
                    training_validation_params,
                    test_validation,
                    test_validation_params,
                    metrics,
                ) = train_classification_model(
                    feature_extraction_id,
                    collection_id,
                    studies,
                    data_splitting_type,
                    training_patients,
                    test_patients,
                    gt,
                )

                model_path = get_model_path(
                    g.user,
                    album["album_id"],
                    label_category.label_type,
                    modalities,
                    rois,
                )
            elif MODEL_TYPES(label_category.label_type) == MODEL_TYPES.SURVIVAL:
                (
                    trained_model,
                    feature_selection,
                    patient_ids,
                    feature_names,
                ) = train_survival_model(
                    feature_extraction_id,
                    collection_id,
                    studies,
                    data_splitting_type,
                    gt,
                )

                model_path = get_model_path(
                    g.user,
                    album["album_id"],
                    label_category.label_type,
                    modalities,
                    rois,
                )
            else:
                raise NotImplementedError

            # Persist model in DB and on disk (pickle it)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_file = joblib.dump(model, model_path)

            # Generate model name (for now)
            (file, ext) = os.path.splitext(os.path.basename(model_path))
            model_name = f"{album['name']}_{file}"

            db_model = Model(
                model_name,
                "MULTIPLE",  # TODO - How to deal with trying multiple configurations?
                data_splitting_type,
                train_test_split_type,
                f"{training_validation} ({training_validation_params['k']} folds, {training_validation_params['n']} repetitions)"
                if training_validation and training_validation_params
                else "5-fold cross-validation",  # TODO - Get this from the survival analysis method
                f"{test_validation} ({test_validation_params['n']} repetitions)"
                if test_validation
                else None,
                "MULTIPLE",  # TODO - How to deal with trying multiple configurations?
                feature_selection,
                feature_names,
                modalities,
                rois,
                training_patients,
                test_patients,
                model_path,
                metrics,
                g.user,
                album["album_id"],
                label_category.id,
                feature_extraction_id,
                collection_id,
            )
            db_model.save_to_db()

            return jsonify(format_model(db_model))
        except Exception as e:
            traceback.print_exc()
            error_message = {"error": str(e)}
            return make_response(jsonify(error_message), 500)


@bp.route("/models/<id>", methods=["DELETE"])
def model(id):
    the_model = Model.delete_by_id(
        id,
        options=(
            joinedload(Model.feature_extraction),
            joinedload(Model.label_category),
        ),
    )
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


def get_model_path(user_id, album_id, model_type, modalities, rois):
    # Define features path for storing the results
    models_dir = os.path.join(MODELS_BASE_DIR, user_id, album_id)

    models_filename = f"model_{model_type}_{'-'.join(modalities)}_{'-'.join(rois)}"

    models_filename += f"_{str(int(time()))}"
    models_filename += ".joblib"
    models_path = os.path.join(models_dir, models_filename)

    return models_path

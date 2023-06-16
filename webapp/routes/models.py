import os
import traceback

from sqlalchemy.orm import joinedload
from flask import Blueprint, jsonify, request, g, make_response
from quantimage2_backend_common.const import MODEL_TYPES
from quantimage2_backend_common.models import Model, LabelCategory
from quantimage2_backend_common.utils import get_training_id, format_model
from routes.utils import validate_decorate
from service.machine_learning import train_model

# Define blueprint
bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


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
        feature_selection = None

        try:
            n_steps = train_model(
                feature_extraction_id,
                collection_id,
                album,
                studies,
                feature_selection,
                label_category,
                data_splitting_type,
                train_test_split_type,
                training_patients,
                test_patients,
                gt,
            )

            training_id = get_training_id(feature_extraction_id, collection_id)

            return jsonify({"training-id": training_id, "n-steps": n_steps})

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

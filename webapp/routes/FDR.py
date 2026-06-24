import json

from flask import Blueprint, jsonify, request, Response, g
from routes.utils import validate_decorate
from quantimage2_backend_common.models import LabelCategory
from service.machine_learning import compute_fdr

# Define blueprint
bp = Blueprint("fdr", __name__)


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/fdr/simpleFDR", methods=["POST"])
def simpleFDR():
    body = request.json
    print(body)

    label_category = LabelCategory.find_by_id(body["label_category_id"])
    print(label_category)
    # print(label_category.label_type) -->gives Classification, here (REMEMBER FOR LATER)
    user_id = g.user

    feature_extraction_id = body["extraction_id"]
    selected_feature_ids = body["selected_feature_ids"]
    collection_id = body["collection_id"]
    fdr_threshold = body["fdr_threshold"]
    album = body["album"]
    album_studies = body["album_studies"]
    gt = body["labels"]
    training_patients = body["training_patients"]
    test_patients = body["test_patients"]

    compute_fdr(
        feature_extraction_id,
        collection_id,
        album,
        album_studies,
        label_category,
        gt,
        training_patients,
        test_patients,
        user_id,
        selected_feature_ids,
        fdr_threshold,
    )

    return jsonify(["toDropTest", "toDropTest2"])

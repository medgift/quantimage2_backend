import json

from flask import Blueprint, jsonify, request, Response, g
from routes.utils import validate_decorate
from quantimage2_backend_common.models import LabelCategory
from service.FDR import compute_fdr

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
    user_id = g.user

    feature_extraction_id = body["extraction_id"]
    selected_feature_ids = body["selected_feature_ids"]
    collection_id = body["collection_id"]
    fdr_threshold_list = body["fdr_threshold_list"]
    album = body["album"]
    album_studies = body["album_studies"]
    gt = body["labels"]
    training_patients = body["training_patients"]

    results_by_qvalues = compute_fdr(
        feature_extraction_id,
        collection_id,
        album,
        album_studies,
        label_category,
        gt,
        training_patients,
        user_id,
        selected_feature_ids,
        fdr_threshold_list,
    )

    return jsonify(results_by_qvalues)

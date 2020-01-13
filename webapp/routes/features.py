from flask import Blueprint, jsonify, request, g

from imaginebackend_common.utils import fetch_extraction_result, format_extraction
from ..service.feature_extraction import run_feature_extraction

from imaginebackend_common.models import FeatureExtraction, FeatureExtractionTask

from .utils import validate_decorate

from .. import my_celery

# Define blueprint
bp = Blueprint(__name__, "features")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/")
def hello():
    return "Hello IMAGINE!"


# Extraction of a study
@bp.route("/extractions/study/<study_uid>")
def extraction_by_study(study_uid):
    user_id = g.user

    # Find the latest task linked to this study
    latest_task_of_study = FeatureExtractionTask.find_latest_by_user_and_study(
        user_id, study_uid
    )

    # Use the latest feature extraction for this study OR an album that includes this study
    if latest_task_of_study:
        latest_extraction_of_study = latest_task_of_study.feature_extraction
        return jsonify(format_extraction(latest_extraction_of_study))
    else:
        return jsonify(None)


# Extractions by album
@bp.route("/extractions/album/<album_id>")
def extractions_by_album(album_id):
    user_id = g.user

    # Find latest feature extraction for this album
    latest_extraction_of_album = FeatureExtraction.find_latest_by_user_and_album_id(
        user_id, album_id
    )

    if latest_extraction_of_album:
        return jsonify(format_extraction(latest_extraction_of_album))
    else:
        return jsonify(None)


# Status of a feature extraction
@bp.route("/extractions/<extraction_id>/status")
def extraction_status(extraction_id):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    status = fetch_extraction_result(my_celery, extraction.result_id)

    response = vars(status)

    return jsonify(response)


# Feature extraction for a study
@bp.route("/extract/study/<study_uid>", methods=["POST"])
def extract_study(study_uid):
    user_id = g.user

    feature_families_map = request.json

    # Define feature families to extract
    feature_extraction = run_feature_extraction(
        user_id, None, feature_families_map, study_uid
    )

    return jsonify(format_extraction(feature_extraction))


# Feature extraction for an album
@bp.route("/extract/album/<album_id>", methods=["POST"])
def extract_album(album_id):
    user_id = g.user
    token = g.token

    feature_families_map = request.json

    # Define feature families to extract
    feature_extraction = run_feature_extraction(
        user_id, album_id, feature_families_map, None, token
    )

    return jsonify(format_extraction(feature_extraction))

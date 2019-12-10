from flask import Blueprint, jsonify, request, g
from ..service.feature_extraction import run_feature_extraction, format_feature_tasks

from ..models import FeatureExtraction

from .utils import validate_decorate, fetch_extraction_result

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

    # Find all feature extractions for this study
    latest_extraction_of_study = FeatureExtraction.find_latest_by_user_and_study_uid(
        user_id, study_uid
    )

    if latest_extraction_of_study:
        return jsonify(format_extraction(latest_extraction_of_study))
    else:
        return jsonify(None)


# Status of a feature extraction
@bp.route("/extractions/<extraction_id>/status")
def extraction_status(extraction_id):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    status = fetch_extraction_result(extraction.result_id)

    response = vars(status)

    return jsonify(response)


def format_extraction(extraction):
    extraction_dict = extraction.to_dict()

    status = fetch_extraction_result(extraction.result_id)
    extraction_dict["status"] = vars(status)

    formatted_tasks = {"tasks": format_feature_tasks(extraction.tasks)}

    dict.update(extraction_dict, formatted_tasks)

    return extraction_dict


# Extractions by album
@bp.route("/extractions/album/<album_uid>")
def features_by_album(album_uid):
    pass


@bp.route("/extract/study/<study_uid>", methods=["POST"])
def extract_study(study_uid):
    user_id = g.user

    feature_families_map = request.json

    # Define feature families to extract
    feature_extraction = run_feature_extraction(
        g.token, user_id, None, feature_families_map, study_uid
    )

    return jsonify(format_extraction(feature_extraction))

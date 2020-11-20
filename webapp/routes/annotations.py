import os

import pandas
from flask import Blueprint, jsonify, request, g, current_app, Response

# Define blueprint
from sklearn.preprocessing import StandardScaler

from imaginebackend_common.const import MODEL_TYPES
from imaginebackend_common.models import FeatureExtraction, Label, Annotation
from routes.utils import decorate_if_possible
from service.feature_extraction import get_studies_from_album
from service.feature_transformation import (
    transform_studies_features_to_df,
    PATIENT_ID_FIELD,
    MODALITY_FIELD,
    ROI_FIELD,
    OUTCOME_FIELD_CLASSIFICATION,
)

bp = Blueprint(__name__, "annotations")


@bp.before_request
def before_request():
    decorate_if_possible(request)


@bp.route("/annotations/<album_id>", methods=("GET", "POST"))
def annotations_by_album(album_id):

    # TODO - Remove this hard-coded test route that's used by Julien
    if not hasattr(g, "user"):
        # Fixed user so far
        user_id = os.environ["KHEOPS_ANNOTATOR_USER_ID"]
    else:
        user_id = g.user

    if request.method == "POST":
        annotation = request.json
        created_annotation = Annotation.create_annotation(
            album_id,
            annotation["parent_id"] if "parent_id" in annotation else None,
            annotation["deleted"] if "deleted" in annotation else False,
            annotation["title"],
            annotation["text"],
            annotation["lines"] if "lines" in annotation else None,
            user_id,
        )

        return jsonify(created_annotation.to_dict())

    if request.method == "GET":
        annotations = Annotation.find_by_album(album_id, user_id)

        formatted_annotations = list(
            map(lambda annotation: annotation.to_dict(), annotations)
        )

        return jsonify(formatted_annotations)


@bp.route("/annotations/<id>", methods=("PATCH", "DELETE"))
def annotation(id):
    if request.method == "PATCH":
        annotation = Annotation.find_by_id(id)
        annotation.update(**request.json)
        return jsonify(annotation.to_dict())

    if request.method == "DELETE":
        deleted_annotation = Annotation.delete_by_id(id)
        return jsonify(deleted_annotation.to_dict())

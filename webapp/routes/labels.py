from flask import Blueprint, jsonify, request, g, current_app, Response

from imaginebackend_common.models import Label

# Define blueprint
from routes.utils import validate_decorate

bp = Blueprint(__name__, "labels")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/labels/<album_id>", methods=("GET", "POST"))
def labels_by_album(album_id):
    if request.method == "POST":
        return save_labels(album_id)

    if request.method == "GET":
        labels = Label.find_by_album(album_id, g.user)
        serialized_labels = list(map(lambda label: label.to_dict(), labels))
        return jsonify(serialized_labels)


def save_labels(album_id):
    user = g.user
    outcomes_dict = request.json

    outcomes = []

    # TODO - Allow choosing a mode (Patient only or patient + roi)
    # for patient_id, roi_dict in outcomes_dict.items():
    #     for roi, outcome in roi_dict.items():
    #         created_updated_label = Label.save_label(
    #             album_id, patient_id, roi, outcome, user
    #         )
    #         outcomes.append(created_updated_label.to_dict())

    for patient_id, outcome in outcomes_dict.items():
        created_updated_label = Label.save_label(album_id, patient_id, outcome, user)
        outcomes.append(created_updated_label.to_dict())

    return jsonify(outcomes)


@bp.route("/labels")
def labels_by_user():
    albums = Label.find_by_user(g.user)
    return jsonify(albums)

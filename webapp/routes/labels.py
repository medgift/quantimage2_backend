from flask import Blueprint, jsonify, request, g, current_app, Response

from imaginebackend_common.models import Label

# Define blueprint
from routes.utils import validate_decorate

bp = Blueprint(__name__, "labels")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/labels/<album_id>/<label_type>", methods=("GET", "POST"))
def labels_by_album(album_id, label_type):
    if request.method == "POST":
        return save_labels(album_id, label_type)

    if request.method == "GET":
        labels = Label.find_by_album(album_id, g.user, label_type)
        serialized_labels = list(map(lambda label: label.to_dict(), labels))
        return jsonify(serialized_labels)


def save_labels(album_id, label_type):
    user = g.user
    labels_dict = request.json

    labels = []

    # TODO - Allow choosing a mode (Patient only or patient + roi)
    # for patient_id, roi_dict in outcomes_dict.items():
    #     for roi, outcome in roi_dict.items():
    #         created_updated_label = Label.save_label(
    #             album_id, patient_id, roi, outcome, user
    #         )
    #         outcomes.append(created_updated_label.to_dict())

    for patient_id, label_content in labels_dict.items():
        created_updated_label = Label.save_label(
            album_id, patient_id, label_type, label_content, user
        )
        labels.append(created_updated_label.to_dict())

    return jsonify(labels)


@bp.route("/labels")
def labels_by_user():
    albums = Label.find_by_user(g.user)
    return jsonify(albums)

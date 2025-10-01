from flask import Blueprint, jsonify, request, g

from quantimage2_backend_common.models import Label, LabelCategory

# Define blueprint
from routes.utils import validate_decorate

bp = Blueprint(__name__, "labels")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/labels/<label_category_id>", methods=("GET", "POST", "PATCH", "DELETE"))
def labels(label_category_id):

    if request.method == "POST":
        return save_labels(
            label_category_id,
            request.json["label_map"],
            request.json["pos_label"] if "pos_label" in request.json else None,
        )

    if request.method == "PATCH":
        return edit_label_category(label_category_id, request.json["name"])

    if request.method == "DELETE":
        return delete_label_category(label_category_id)


@bp.route("/labels/patients_from_label_category/<label_category_id>", methods=["GET"])
def get_patient_ids_in_label_category(label_category_id):
    if request.method == "GET":
        labels = Label.find_by_label_category_id(label_category_id)
        return jsonify([i.patient_id for i in labels])


@bp.route("/label-categories/<album_id>", methods=("GET", "POST"))
def label_categories_by_album(album_id):
    if request.method == "POST":
        return save_label_category(
            album_id, request.json["label_type"], request.json["name"], g.user
        )

    if request.method == "GET":
        label_categories = LabelCategory.find_by_album(album_id, g.user)
        serialized_labels = list(
            map(lambda label_category: label_category.to_dict(), label_categories)
        )
        return jsonify(serialized_labels)


def save_label_category(album_id, label_type, name, user_id):
    new_category = LabelCategory(album_id, label_type, name, user_id)

    new_category.save_to_db()
    
    new_category.initialize_empty_labels()

    return jsonify(new_category.to_dict())


def edit_label_category(label_category_id, name):
    category = LabelCategory.find_by_id(label_category_id)

    category.name = name

    category.save_to_db()

    return jsonify(category.to_dict())


def delete_label_category(label_category_id):
    # First, delete all labels associated with this category
    Label.query.filter_by(label_category_id=label_category_id).delete()
    
    # Then delete the category itself
    category = LabelCategory.delete_by_id(label_category_id)
    
    return jsonify(category.to_dict())


def save_labels(label_category_id, labels_dict, pos_label):
    labels_to_save = []
    for patient_id, label_content in labels_dict.items():
        labels_to_save.append(Label(label_category_id, patient_id, label_content))

    labels = Label.save_labels(label_category_id, labels_to_save)

    # Define the positive label if defined
    category = LabelCategory.find_by_id(label_category_id)
    if pos_label:
        category.pos_label = pos_label
    category.save_to_db()

    return jsonify([l.to_dict() for l in labels])

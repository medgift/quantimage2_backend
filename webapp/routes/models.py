from flask import Blueprint, jsonify, request, g, current_app, Response

from imaginebackend_common.models import Model

# Define blueprint
from routes.utils import validate_decorate

bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/models/<album_id>")
def models_by_album(album_id):
    albums = Model.find_by_album(album_id, g.user)
    return jsonify(albums)


@bp.route("/models")
def models_by_user():
    albums = Model.find_by_user(g.user)
    return jsonify(albums)

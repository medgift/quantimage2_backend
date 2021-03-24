from flask import Blueprint, jsonify, request, g, current_app, Response

from imaginebackend_common.models import NavigationHistory

# Define blueprint
from routes.utils import validate_decorate

bp = Blueprint(__name__, "navigation")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/navigation", methods=["POST"])
def create_navigation_entry():
    path = request.json["path"]

    entry = NavigationHistory.create_entry(path, g.user)

    return jsonify(entry.to_dict())

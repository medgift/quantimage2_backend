import pandas as pd
from flask import Blueprint, jsonify, request, Response
from routes.utils import validate_decorate

# Define blueprint
bp = Blueprint("fdr", __name__)


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/fdr/test", methods=["GET"])
def test():
    print("Hello it's a test from backend")
    return jsonify({"ok": True})

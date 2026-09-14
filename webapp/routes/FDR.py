import logging
import traceback

from flask import Blueprint, g, jsonify, make_response, request

from quantimage2_backend_common.models import LabelCategory
from routes.utils import validate_decorate
from service.FDR import compute_fdr

logger = logging.getLogger(__name__)

# Define blueprint
bp = Blueprint("fdr", __name__)


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/fdr/simpleFDR", methods=["POST"])
def simpleFDR():
    # NOTE: never log `body` — it carries patient IDs and their outcomes.
    body = request.json

    label_category = LabelCategory.find_by_id(body["label_category_id"])
    if label_category is None:
        return make_response(
            jsonify({"error": f"Label category {body['label_category_id']} not found"}),
            404,
        )

    try:
        results_by_qvalues = compute_fdr(
            body["extraction_id"],
            body["collection_id"],
            body["album"],
            body["album_studies"],
            label_category,
            body["labels"],
            body["training_patients"],
            g.user,
            body["selected_feature_ids"],
            body["fdr_threshold_list"],
        )
    except ValueError as e:
        # Raised when the request itself is inconsistent (unknown features, no
        # features to screen), which is the caller's problem, not a server bug.
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        traceback.print_exc()
        return make_response(jsonify({"error": str(e)}), 500)

    return jsonify(results_by_qvalues)

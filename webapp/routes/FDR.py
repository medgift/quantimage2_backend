import logging
import traceback

from flask import Blueprint, g, jsonify, make_response, request

from quantimage2_backend_common.models import (
    FeatureCollection,
    FeatureExtraction,
    LabelCategory,
)
from routes.utils import validate_decorate
from service.FDR import compute_fdr

logger = logging.getLogger(__name__)

# Define blueprint
bp = Blueprint("fdr", __name__)


@bp.before_request
def before_request():
    validate_decorate(request)


def _not_found(message):
    return make_response(jsonify({"error": message}), 404)


@bp.route("/fdr/simpleFDR", methods=["POST"])
def simpleFDR():
    # NOTE: never log `body` — it carries patient IDs and their outcomes.
    body = request.json

    # Every object below is loaded by an ID taken from the request, and none of
    # the loaders filter by user, so check ownership here. Answer 404 rather
    # than 403 so a caller can't probe which IDs exist for other users.
    label_category = LabelCategory.find_by_id(body["label_category_id"])
    if label_category is None or label_category.user_id != g.user:
        return _not_found(f"Label category {body['label_category_id']} not found")

    extraction = FeatureExtraction.find_by_id(body["extraction_id"])
    if extraction is None or extraction.user_id != g.user:
        return _not_found(f"Feature extraction {body['extraction_id']} not found")

    # A collection has no owner column: it belongs to whoever owns its
    # extraction, so it must hang off the extraction checked above.
    collection_id = body["collection_id"]
    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        if collection is None or collection.feature_extraction_id != extraction.id:
            return _not_found(f"Feature collection {collection_id} not found")

    # The extraction, the labels and the clinical features (looked up by the
    # body's album) must all describe the same album, or the screening mixes
    # unrelated patients and outcomes.
    album_id = body["album"]["album_id"]
    if extraction.album_id != album_id or label_category.album_id != album_id:
        return make_response(
            jsonify(
                {"error": "Extraction and label category must belong to the album"}
            ),
            400,
        )

    try:
        results_by_qvalues = compute_fdr(
            extraction.id,
            collection_id,
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

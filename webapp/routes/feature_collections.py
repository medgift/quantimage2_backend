from flask import Blueprint, abort, jsonify, request, current_app, g

from imaginebackend_common.kheops_utils import dicomFields
from imaginebackend_common.models import (
    FeatureExtraction,
    Modality,
    ROI,
    FeatureValue,
    FeatureCollection,
    FeatureDefinition,
)
from service.feature_extraction import get_studies_from_album
from service.feature_transformation import MODALITY_FIELD, ROI_FIELD, PATIENT_ID_FIELD
from .utils import validate_decorate


# Define blueprint
bp = Blueprint(__name__, "feature_collections")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/feature-collections", methods=("GET", "POST"))
def feature_collections():
    if request.method == "POST":
        collection = save_feature_collection(
            request.json["featureExtractionID"],
            request.json["name"],
            request.json["modalities"],
            request.json["rois"],
            request.json["patients"],
            request.json["features"],
        )

        return jsonify(collection.format_collection())


@bp.route("/feature-collections/extraction/<extraction_id>")
def feature_collections_by_extraction(extraction_id):
    collections = FeatureCollection.find_by_extraction(extraction_id)

    # TODO - Maybe have another route for this, that returns just the collection objects (more standard)
    serialized_collections = list(map(lambda c: c.format_collection(), collections))
    return jsonify(serialized_collections)


@bp.route("/feature-collections/<feature_collection_id>", methods=("GET", "PATCH"))
def feature_collection(feature_collection_id):

    if request.method == "PATCH":
        return update_feature_collection(feature_collection_id)


def save_feature_collection(
    feature_extraction_id, name, modalities, rois, patients, feature_names
):
    token = g.token

    # Get all necessary elements from the DB
    extraction = FeatureExtraction.find_by_id(feature_extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    # Find all FeatureValues corresponding to the supplied criteria
    values = FeatureValue.find_by_collection_criteria(
        extraction, studies, modalities, rois, patients, feature_names
    )

    # Go through rows and build up the list of feature values to keep
    # values = []
    # for row in rows:
    #     patient_id = row[PATIENT_ID_FIELD]
    #     study_uid = next(
    #         s[dicomFields.STUDY_UID][dicomFields.VALUE][0]
    #         for s in studies
    #         if s[dicomFields.PATIENT_ID][dicomFields.VALUE][0] == patient_id
    #     )
    #     tasks = list(filter(lambda t: t.study_uid == study_uid, extraction.tasks))
    #     modality_name = row[MODALITY_FIELD]
    #     roi_name = row[ROI_FIELD]
    #
    #     modality_id = next(m.id for m in modalities if m.name == modality_name)
    #     roi_id = next(r.id for r in rois if r.name == roi_name)
    #
    #     row_values = FeatureValue.find_by_tasks_modality_roi_features(
    #         list(map(lambda t: t.id, tasks)),
    #         modality_id,
    #         roi_id,
    #         feature_definition_ids,
    #     )
    #
    #     values += row_values

    collection, created = FeatureCollection.get_or_create(
        criteria={"name": name, "feature_extraction_id": feature_extraction_id},
        defaults={"name": name, "feature_extraction_id": feature_extraction_id},
    )

    collection.values = values
    collection.save_to_db()

    return collection


def update_feature_collection():
    print("hai")

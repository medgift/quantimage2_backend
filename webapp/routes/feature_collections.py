import io
import os
from zipfile import ZipFile, ZIP_DEFLATED

from flask import Blueprint, abort, jsonify, request, current_app, g, Response
from ttictoc import tic, toc

from config import oidc_client
from imaginebackend_common.kheops_utils import dicomFields
from imaginebackend_common.models import (
    FeatureExtraction,
    Modality,
    ROI,
    FeatureValue,
    FeatureCollection,
    FeatureDefinition,
)
from service.feature_extraction import get_studies_from_album, get_album_details
from service.feature_transformation import (
    MODALITY_FIELD,
    ROI_FIELD,
    PATIENT_ID_FIELD,
    make_album_file_name,
    transform_studies_collection_features_to_df,
    make_album_collection_file_name,
)
from .utils import validate_decorate

# Define blueprint
bp = Blueprint(__name__, "feature_collections")


@bp.before_request
def before_request():
    if not request.path.endswith("download"):
        validate_decorate(request)


@bp.route("/feature-collections", methods=("GET", "POST"))
def feature_collections():
    if request.method == "POST":
        collection = save_feature_collection(
            request.json["featureExtractionID"],
            request.json["name"],
            request.json["featureIDs"],
            request.json["patientIDs"],
        )

        return jsonify(collection.format_collection(with_values=True))


@bp.route("/feature-collections/new", methods=["POST"])
def feature_collections_new():
    if request.method == "POST":
        collection = save_feature_collection_new(
            request.json["featureExtractionID"],
            request.json["name"],
            request.json["featureIDs"],
            request.json["patientIDs"],
        )

        return jsonify(collection.format_collection(with_values=True))


@bp.route("/feature-collections/extraction/<extraction_id>")
def feature_collections_by_extraction(extraction_id):
    collections = FeatureCollection.find_by_extraction(extraction_id)

    # TODO - Maybe have another route for this, that returns just the collection objects (more standard)
    serialized_collections = list(map(lambda c: c.format_collection(), collections))
    return jsonify(serialized_collections)


@bp.route(
    "/feature-collections/<feature_collection_id>", methods=("GET", "PATCH", "DELETE")
)
def feature_collection(feature_collection_id):
    if request.method == "GET":
        return get_feature_collection(feature_collection_id)

    if request.method == "PATCH":
        return update_feature_collection(feature_collection_id)

    if request.method == "DELETE":
        return delete_feature_collection(feature_collection_id)


# Download features in CSV format
@bp.route("/feature-collections/<feature_collection_id>/download")
def download_collection_by_id(feature_collection_id):
    # Get the feature collection from the DB
    feature_collection = FeatureCollection.find_by_id(feature_collection_id)

    # Get the feature extraction to process from the DB
    feature_extraction = FeatureExtraction.find_by_id(
        feature_collection.feature_extraction_id
    )

    # Identify user (in order to get a token)
    user_id = request.args.get("userID", None)

    # Get a token for the given user (possible thanks to token exchange in Keycloak)
    token = oidc_client.token_exchange(
        requested_token_type="urn:ietf:params:oauth:token-type:access_token",
        audience=os.environ["KEYCLOAK_IMAGINE_CLIENT_ID"],
        requested_subject=user_id,
    )["access_token"]

    # Get album name & list of studies
    album_name = get_album_details(feature_extraction.album_id, token)["name"]
    album_studies = get_studies_from_album(feature_extraction.album_id, token)

    # Transform the features into a DataFrame
    header, features_df = transform_studies_collection_features_to_df(
        feature_collection, album_studies
    )

    # Album : send back a zip file with CSV files separated by
    # - Modality : PT/CT features shouldn't be mixed for example
    # - ROI : Main tumor & metastases features shouldn't be mixed for example
    grouped_features = features_df.groupby([MODALITY_FIELD, ROI_FIELD])

    # Create ZIP file to return
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, "a", ZIP_DEFLATED, False) as zip_file:
        for group_name, group_data in grouped_features:
            group_csv_content = group_data.to_csv(index=False)

            group_file_name = f"features_album_{album_name.replace(' ', '-')}_collection_{feature_collection.name.replace(' ', '-')}_{'-'.join(group_name)}.csv"
            zip_file.writestr(group_file_name, group_csv_content)

    file_name = make_album_collection_file_name(album_name, feature_collection.name)

    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={
            "Content-disposition": f"attachment; filename={file_name}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


def save_feature_collection_new(feature_extraction_id, name, feature_ids, patient_ids):
    token = g.token

    # Get all necessary elements from the DB
    extraction = FeatureExtraction.find_by_id(feature_extraction_id)

    collection, created = FeatureCollection.get_or_create(
        criteria={
            "name": name,
            "feature_extraction_id": feature_extraction_id,
            "feature_ids": feature_ids,
            "patient_ids": patient_ids,
        },
        defaults={
            "name": name,
            "feature_extraction_id": feature_extraction_id,
            "feature_ids": feature_ids,
            "patient_ids": patient_ids,
        },
    )

    return collection


def save_feature_collection(feature_extraction_id, name, feature_ids, patient_ids):
    token = g.token

    # Get all necessary elements from the DB
    extraction = FeatureExtraction.find_by_id(feature_extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    # Find all FeatureValues corresponding to the supplied criteria
    collection, created = FeatureCollection.get_or_create(
        criteria={"name": name, "feature_extraction_id": feature_extraction_id},
        defaults={"name": name, "feature_extraction_id": feature_extraction_id},
    )

    collection.feature_ids = feature_ids
    collection.patient_ids = patient_ids
    collection.save_to_db()

    return collection


def get_feature_collection(feature_collection_id):
    collection = FeatureCollection.find_by_id(feature_collection_id)

    return jsonify(collection.format_collection(with_values=True))

    # return jsonify(collection.to_dict())


def update_feature_collection(feature_collection_id):
    collection = FeatureCollection.find_by_id(feature_collection_id)
    collection.update(**request.json)

    return jsonify(collection.to_dict())


def delete_feature_collection(feature_collection_id):
    deleted_collection = FeatureCollection.delete_by_id(feature_collection_id)
    return jsonify(deleted_collection.to_dict())

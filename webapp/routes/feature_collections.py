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
            request.json["modalities"],
            request.json["rois"],
            request.json["patients"],
            request.json["features"],
        )

        return jsonify(collection.format_collection(with_values=True))


@bp.route("/feature-collections/new", methods=["POST"])
def feature_collections_new():
    if request.method == "POST":
        collection = save_feature_collection_new(
            request.json["featureExtractionID"],
            request.json["name"],
            request.json["featureIDs"],
            request.json["patients"],
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
        album_studies, feature_collection
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


def save_feature_collection_new(feature_extraction_id, name, feature_ids, patients):
    token = g.token

    # Get all necessary elements from the DB
    extraction = FeatureExtraction.find_by_id(feature_extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    # Find all FeatureValues corresponding to the supplied criteria
    tic()
    value_ids = FeatureValue.find_id_by_collection_criteria_new(
        extraction, studies, feature_ids, patients
    )
    elapsed = toc()
    print("Getting the collection values from the DB took", elapsed)

    collection, created = FeatureCollection.get_or_create(
        criteria={"name": name, "feature_extraction_id": feature_extraction_id},
        defaults={"name": name, "feature_extraction_id": feature_extraction_id},
    )

    # Build instances to save in bulk
    feature_collection_value_instances = [
        {"feature_collection_id": collection.id, "feature_value_id": value_id.id}
        for value_id in value_ids
    ]

    # Batch create the instances
    tic()
    FeatureCollection.save_feature_collection_values_batch(
        feature_collection_value_instances
    )
    elapsed = toc()
    print("Saving the collection values to the DB took", elapsed)

    return collection


def save_feature_collection(
    feature_extraction_id, name, modalities, rois, patients, feature_names
):
    token = g.token

    # Get all necessary elements from the DB
    extraction = FeatureExtraction.find_by_id(feature_extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    # Find all FeatureValues corresponding to the supplied criteria
    tic()
    value_ids = FeatureValue.find_id_by_collection_criteria(
        extraction, studies, modalities, rois, patients, feature_names
    )
    elapsed = toc()
    print("Getting the collection values from the DB took", elapsed)

    collection, created = FeatureCollection.get_or_create(
        criteria={"name": name, "feature_extraction_id": feature_extraction_id},
        defaults={"name": name, "feature_extraction_id": feature_extraction_id},
    )

    # Build instances to save in bulk
    feature_collection_value_instances = [
        {"feature_collection_id": collection.id, "feature_value_id": value_id.id}
        for value_id in value_ids
    ]

    # Batch create the instances
    tic()
    FeatureCollection.save_feature_collection_values_batch(
        feature_collection_value_instances
    )
    elapsed = toc()
    print("Saving the collection values to the DB took", elapsed)

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

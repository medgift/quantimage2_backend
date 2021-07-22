import json

from flask import Blueprint, jsonify, request, g, current_app, Response

import yaml

import numpy

from random import randint

from keycloak.realm import KeycloakRealm
from ttictoc import tic, toc

from config import oidc_client
from imaginebackend_common.const import MODEL_TYPES
from imaginebackend_common.kheops_utils import dicomFields
from imaginebackend_common.utils import (
    fetch_extraction_result,
    format_extraction,
    read_feature_file,
)
from service.feature_extraction import (
    run_feature_extraction,
    get_studies_from_album,
    get_album_details,
)

from imaginebackend_common.models import (
    FeatureExtraction,
    FeatureExtractionTask,
    FeatureCollection,
    Label,
    Album,
)
from service.feature_transformation import (
    transform_studies_features_to_df,
    make_album_file_name,
    MODALITY_FIELD,
    ROI_FIELD,
    transform_studies_collection_features_to_df,
    get_data_points_extraction,
    get_data_points_collection,
)
from .charts import format_lasagna_data

from .utils import validate_decorate

from zipfile import ZipFile, ZIP_DEFLATED

import csv
import io
import os
import pandas


# Define blueprint
bp = Blueprint(__name__, "features")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    if not request.path.endswith("download"):
        validate_decorate(request)


@bp.route("/")
def hello():
    return "Hello IMAGINE!"


# Get feature payload for a given feature extraction
@bp.route("/extractions/<id>")
def extraction_by_id(id):
    feature_extraction = FeatureExtraction.find_by_id(id)

    return jsonify(format_extraction(feature_extraction, payload=True, tasks=True))


# Get feature details for a given extraction
# INCLUDING the data for the lasagna chart (to improve performance)
@bp.route(
    "/extractions/<extraction_id>/feature-details", defaults={"collection_id": None}
)
@bp.route("/extractions/<extraction_id>/collections/<collection_id>/feature-details")
def extraction_features_by_id(extraction_id, collection_id):
    token = g.token

    extraction = FeatureExtraction.find_by_id(extraction_id)
    album = Album.find_by_album_id(extraction.album_id)
    studies = get_studies_from_album(extraction.album_id, token)

    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        header, features_df = transform_studies_collection_features_to_df(
            studies, collection
        )
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

    labels = []

    # Get labels (if current outcome is defined & it's classification)
    if (
        album.current_outcome
        and MODEL_TYPES(album.current_outcome.label_type) == MODEL_TYPES.CLASSIFICATION
    ):
        labels = Label.find_by_label_category(album.current_outcome_id)

    tic()
    formatted_lasagna_data = format_lasagna_data(features_df, labels)

    features_json = json.loads(features_df.to_json(orient="records"))
    elapsed = toc()
    print("Formatting & serializing features took", elapsed)

    response = jsonify(
        {
            "header": header,
            "features": features_json,
            "visualization": formatted_lasagna_data,
        }
    )

    return response


# Get data points (PatientID/ROI) for a given extraction
@bp.route("/extractions/<extraction_id>/collections/<collection_id>/data-points")
def extraction_collection_data_points(extraction_id, collection_id):
    token = g.token

    extraction = FeatureExtraction.find_by_id(extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    tic()
    patient_ids = get_data_points_collection(studies, collection_id)
    elapsed = toc()
    print("Getting data points for collection took", elapsed)

    return jsonify({"data-points": patient_ids})


# Get data points (PatientID/ROI) for a given extraction
@bp.route("/extractions/<id>/data-points")
def extraction_data_points_by_id(id):
    token = g.token

    extraction = FeatureExtraction.find_by_id(id)

    tic()
    result = fetch_extraction_result(
        current_app.my_celery, extraction.result_id, tasks=extraction.tasks
    )
    elapsed = toc()
    print("Getting extraction result for data points took", elapsed)

    studies = get_studies_from_album(extraction.album_id, token)

    tic()
    patient_ids = get_data_points_extraction(result, studies)
    elapsed = toc()
    print("Getting data points for extraction took", elapsed)

    # TODO - Allow choosing a mode (patient only or patient + roi)
    return jsonify({"data-points": patient_ids})


# Download features in CSV format
@bp.route("/extractions/<id>/download")  # ?patientID=???&studyDate=??? OR ?userID=???
def download_extraction_by_id(id):

    # Get the feature extraction to process from the DB
    feature_extraction = FeatureExtraction.find_by_id(id)

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
    header, features_df = transform_studies_features_to_df(
        feature_extraction, album_studies
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

            group_file_name = f"features_album_{album_name.replace(' ', '-')}_{'-'.join(group_name)}.csv"
            zip_file.writestr(group_file_name, group_csv_content)

    file_name = make_album_file_name(album_name)

    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={
            "Content-disposition": f"attachment; filename={file_name}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# Extractions by album
@bp.route("/extractions/album/<album_id>")
def extractions_by_album(album_id):
    user_id = g.user

    # Find latest feature extraction for this album
    latest_extraction_of_album = FeatureExtraction.find_latest_by_user_and_album_id(
        user_id, album_id
    )

    if latest_extraction_of_album:
        return jsonify(format_extraction(latest_extraction_of_album))
    else:
        return jsonify(None)


# Status of a feature extraction
@bp.route("/extractions/<extraction_id>/status")
def extraction_status(extraction_id):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    status = fetch_extraction_result(current_app.my_celery, extraction.result_id)

    response = vars(status)

    return jsonify(response)


# Feature extraction for an album
@bp.route("/extract/album/<album_id>", methods=["POST"])
def extract_album(album_id):
    user_id = g.user
    token = g.token

    request_body = request.json

    feature_extraction_config_dict = request_body["config"]
    rois = request_body["rois"]

    # Get album metadata for hard-coded labels mapping
    album_metadata = get_album_details(album_id, token)

    # Run the feature extraction
    feature_extraction = run_feature_extraction(
        user_id,
        album_id,
        album_metadata["name"],
        feature_extraction_config_dict,
        rois,
        token,
    )

    return jsonify(format_extraction(feature_extraction))

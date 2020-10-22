from flask import Blueprint, jsonify, request, g, current_app, Response

import numpy

from random import randint

from keycloak.realm import KeycloakRealm

from config import oidc_client
from imaginebackend_common.kheops_utils import dicomFields
from imaginebackend_common.utils import (
    fetch_extraction_result,
    format_extraction,
    read_feature_file,
)
from service.feature_analysis import (
    train_model_with_metric,
    concatenate_modalities_rois,
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
)
from service.feature_transformation import (
    transform_study_features_to_tabular,
    make_study_file_name,
    transform_studies_features_to_df,
    get_csv_file_content,
    make_album_file_name,
    separate_features_by_modality_and_roi,
    MODALITY_FIELD,
    ROI_FIELD,
    transform_studies_collection_features_to_df,
)

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


# Extraction of a study
@bp.route("/extractions/study/<study_uid>")
def extraction_by_study(study_uid):
    user_id = g.user

    # Find the latest task linked to this study
    latest_task_of_study = FeatureExtractionTask.find_latest_by_user_and_study(
        user_id, study_uid
    )

    # Use the latest feature extraction for this study OR an album that includes this study
    if latest_task_of_study:
        latest_extraction_of_study = latest_task_of_study.feature_extraction
        return jsonify(format_extraction(latest_extraction_of_study))
    else:
        return jsonify(None)


# Get feature payload for a given feature extraction
@bp.route("/extractions/<id>")
def extraction_by_id(id):
    feature_extraction = FeatureExtraction.find_by_id(id)

    return jsonify(format_extraction(feature_extraction, payload=True, tasks=True))


# Get feature details for a given collection
@bp.route("/extractions/<extraction_id>/collections/<collection_id>/feature-details")
def extraction_collection_features(extraction_id, collection_id):
    token = g.token

    extraction = FeatureExtraction.find_by_id(extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)
    collection = FeatureCollection.find_by_id(collection_id)

    header, features_df = transform_studies_collection_features_to_df(
        extraction, studies, collection
    )

    return jsonify(
        {"header": header, "features": features_df.to_dict(orient="records")}
    )


# Get feature details for a given extraction
@bp.route("/extractions/<id>/feature-details")
def extraction_features_by_id(id):
    # TODO - Add support for making this work for a single study as well
    token = g.token

    extraction = FeatureExtraction.find_by_id(id)
    studies = get_studies_from_album(extraction.album_id, token)

    header, features_df = transform_studies_features_to_df(extraction, studies)

    # TODO - Check why this was done here, seems like it shouldn't!
    # Concatenate modalities & ROIs for now
    # features_df = concatenate_modalities_rois(features_df, [], [])

    return jsonify(
        {"header": header, "features": features_df.to_dict(orient="records")}
    )


# Get data points (PatientID/ROI) for a given extraction
@bp.route("/extractions/<extraction_id>/collections/<collection_id>/data-points")
def extraction_collection_data_points(extraction_id, collection_id):
    # TODO - Add support for making this work for a single study as well
    token = g.token

    extraction = FeatureExtraction.find_by_id(extraction_id)
    studies = get_studies_from_album(extraction.album_id, token)

    collection = FeatureCollection.find_by_id(collection_id)

    # Get studies included in the collection's feature values

    study_uids = set(
        list(map(lambda v: v.feature_extraction_task.study_uid, collection.values))
    )

    # Get Patient IDs from studies
    patient_ids = []
    for study in studies:
        patient_id = study[dicomFields.PATIENT_ID][dicomFields.VALUE][0]
        study_uid = study[dicomFields.STUDY_UID][dicomFields.VALUE][0]
        if not patient_id in patient_ids and study_uid in study_uids:
            patient_ids.append(patient_id)

    return jsonify({"data-points": patient_ids})


# Get data points (PatientID/ROI) for a given extraction
@bp.route("/extractions/<id>/data-points")
def extraction_data_points_by_id(id):
    # TODO - Add support for making this work for a single study as well
    token = g.token

    extraction = FeatureExtraction.find_by_id(id)
    studies = get_studies_from_album(extraction.album_id, token)

    # Get Patient IDs from studies
    patient_ids = []
    for study in studies:
        patient_id = study[dicomFields.PATIENT_ID][dicomFields.VALUE][0]
        if not patient_id in patient_ids:
            patient_ids.append(patient_id)

    # TODO - Allow choosing a mode (patient only or patient + roi)
    # Get ROIs from a first feature file
    # first_task = extraction.tasks[0]
    # features_content = read_feature_file(first_task.features_path)
    # first_modality = next(iter(features_content))
    # rois = features_content[first_modality].keys()

    # Generate data points
    # data_points = []
    # for patient_id in patient_ids:
    # for roi in rois:
    #     data_points.append([patient_id, roi])

    return jsonify({"data-points": patient_ids})


# Download features in CSV format
@bp.route("/extractions/<id>/download")  # ?patientID=???&studyDate=??? OR ?userID=???
def download_extraction_by_id(id):

    # Get the feature extraction to process from the DB
    feature_extraction = FeatureExtraction.find_by_id(id)

    # Get the names of the used feature families (for the file name so far)
    feature_families = []
    for extraction_family in feature_extraction.families:
        feature_families.append(extraction_family.feature_family.name)

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

            group_file_name = f"features_album_{album_name}_{'-'.join(feature_families)}_{'-'.join(group_name)}.csv"
            zip_file.writestr(group_file_name, group_csv_content)

    file_name = make_album_file_name(album_name, feature_families)

    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={"Content-disposition": f"attachment; filename={file_name}"},
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

    feature_families_map = request.json

    # Define feature families to extract
    feature_extraction = run_feature_extraction(
        user_id, album_id, feature_families_map, token
    )

    return jsonify(format_extraction(feature_extraction))

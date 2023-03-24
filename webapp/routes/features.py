from pathlib import Path

from flask import Blueprint, jsonify, request, g, current_app, Response
from quantimage2_backend_common.const import (
    FIRSTORDER_REPLACEMENT_SUV,
    PET_MODALITY,
    FIRSTORDER_REPLACEMENT_INTENSITY,
    FIRSTORDER_PYRADIOMICS_PREFIX,
)

from requests_toolbelt import MultipartEncoder

from ttictoc import tic, toc

from config import FEATURES_CACHE_BASE_DIR
from quantimage2_backend_common.kheops_utils import get_user_token
from quantimage2_backend_common.utils import (
    fetch_extraction_result,
    format_extraction,
    read_config_file,
)
from service.feature_extraction import (
    run_feature_extraction,
    get_studies_from_album,
    get_album_details,
)

from quantimage2_backend_common.models import (
    FeatureExtraction,
    Label,
    Album,
    LabelCategory,
    AlbumOutcome,
)
from service.feature_transformation import (
    transform_studies_features_to_df,
    make_album_file_name,
    MODALITY_FIELD,
    ROI_FIELD,
)
from .charts import format_chart_data

from .utils import validate_decorate

from zipfile import ZipFile, ZIP_DEFLATED

import io
import os
import pandas


# Define blueprint
bp = Blueprint(__name__, "features")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    validate_decorate(request)


# Get feature payload for a given feature extraction
@bp.route("/extractions/<id>", methods=["GET", "PATCH"])
def extraction_by_id(id):

    if request.method == "GET":
        feature_extraction = FeatureExtraction.find_by_id(id)

        return jsonify(format_extraction(feature_extraction, tasks=True))
    elif request.method == "PATCH":
        feature_extraction = FeatureExtraction.find_by_id(id)
        feature_extraction.update(**request.json)

        return jsonify(format_extraction(feature_extraction, tasks=True))


# Get feature details for a given extraction
# INCLUDING the data for the chart (to improve performance)
@bp.route("/extractions/<extraction_id>/feature-details")
def extraction_features_by_id(extraction_id):
    token = g.token

    extraction = FeatureExtraction.find_by_id(extraction_id)
    album = Album.find_by_album_id(extraction.album_id)
    album_outcome = AlbumOutcome.find_by_album_user_id(extraction.album_id, g.user)
    studies = get_studies_from_album(extraction.album_id, token)

    header, features_df = get_features_cache_or_db(extraction, studies)

    label_category = None
    labels = []

    # Get labels (if current outcome is defined for this user)
    if album_outcome and album_outcome.outcome_id:
        label_category = LabelCategory.find_by_id(album_outcome.outcome_id)
        labels = Label.find_by_label_category(label_category.id)

    tic()
    chart_df = format_chart_data(features_df, label_category, labels)
    elapsed = toc()
    print("Formatting & serializing features took", elapsed)

    rounded_df = features_df.round(3)
    rounded_chart_df = chart_df.round(3)

    m = MultipartEncoder(
        fields={
            "features_tabular": rounded_df.to_csv(index=False),
            "features_chart": rounded_chart_df.to_csv(index_label="FeatureID"),
        }
    )
    return Response(m.to_string(), mimetype=m.content_type)


def get_features_cache_or_db(extraction, studies):
    features_cache_folder = f"extraction-{extraction.id}"
    features_cache_path = f"{FEATURES_CACHE_BASE_DIR}/{features_cache_folder}"
    features_file_name = "features.h5"
    features_key = "features"
    features_cache_file_path = f"{features_cache_path}/{features_file_name}"

    # Does the features file exist?
    if Path(features_cache_file_path).exists():
        tic()
        features_df = pandas.read_hdf(features_cache_file_path, features_key)
        header = features_df.columns
        elapsed = toc()
        print("Parsing features from cached HDF5 file took", elapsed)
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

        # Persist features DataFrame for caching
        tic()
        os.makedirs(features_cache_path, exist_ok=True)
        features_df.to_hdf(
            features_cache_file_path,
            features_key,
            "w",
            format="fixed",
        )
        elapsed = toc()
        print("Serializing features to HDF5 took", elapsed)

    return header, features_df


@bp.route("/extractions/<id>/download-configuration")
def download_extraction_configuration(id):
    feature_extraction = FeatureExtraction.find_by_id(id)

    config_file = read_config_file(feature_extraction.config_file)

    return Response(
        config_file,
        mimetype="text/yaml",
        headers={
            "Content-disposition": f"attachment; filename={os.path.basename(feature_extraction.config_file)}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# Download features in CSV format
@bp.route("/extractions/<id>/download")  # ?patientID=???&studyDate=??? OR ?userID=???
def download_extraction_by_id(id):
    token = g.token

    # Get the feature extraction to process from the DB
    feature_extraction = FeatureExtraction.find_by_id(id)

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

            # Replace "firstorder" with intensity or SUV (depending on the modality)
            # TODO - This could be done more elegantly/consistently further upstream
            replacement = (
                FIRSTORDER_REPLACEMENT_SUV
                if group_name[0] == PET_MODALITY
                else FIRSTORDER_REPLACEMENT_INTENSITY
            )
            group_csv_content = group_csv_content.replace(
                FIRSTORDER_PYRADIOMICS_PREFIX, replacement
            )

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


# Update a feature extraction
@bp.route("/extractions/<extraction_id>", methods=["PATCH"])
def update_extraction(extraction_id):
    extraction = FeatureExtraction.find_by_id(extraction_id)
    extraction.update(**request.json)

    return jsonify(extraction.to_dict())


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

    # Get a read/write token for the user (valid only for 24 hours)
    user_token_id, user_token = get_user_token(album_id, token)

    # Run the feature extraction
    feature_extraction = run_feature_extraction(
        user_id,
        album_id,
        album_metadata["name"],
        feature_extraction_config_dict,
        rois,
        user_token,
    )

    return jsonify(format_extraction(feature_extraction))

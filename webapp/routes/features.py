import logging
from pathlib import Path

import eventlet
import eventlet.tpool

from flask import Blueprint, jsonify, request, g, current_app, Response

logger = logging.getLogger(__name__)
from celery.result import AsyncResult
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
bp = Blueprint("features", __name__)

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


# Cancel a feature extraction in progress
@bp.route("/extractions/<id>/cancel", methods=["POST"])
def cancel_extraction(id):
    from quantimage2_backend_common.models import (
        db,
        FeatureValue,
        FeatureExtractionTask,
    )

    user_id = g.user

    # Get the feature extraction to cancel
    feature_extraction = FeatureExtraction.find_by_id(id)

    if not feature_extraction:
        return jsonify({"error": "Feature extraction not found"}), 404

    # Verify the user owns this extraction
    if feature_extraction.user_id != user_id:
        return jsonify({"error": "Unauthorized"}), 403

    # Store information needed for cleanup before deletion
    extraction_id = feature_extraction.id
    album_id = feature_extraction.album_id
    task_ids = [task.id for task in feature_extraction.tasks]
    celery_task_ids = [
        task.task_id for task in feature_extraction.tasks if task.task_id
    ]

    # Find the previous extraction BEFORE deleting the current one
    previous_extraction = (
        FeatureExtraction.query.filter(
            FeatureExtraction.user_id == user_id,
            FeatureExtraction.album_id == album_id,
            FeatureExtraction.id < extraction_id,
        )
        .order_by(db.desc(FeatureExtraction.id))
        .first()
    )

    # STEP 1: Delete the extraction from DB FIRST
    # This ensures any queued/future tasks will immediately exit when they check for the extraction
    # Delete in correct order: FeatureValues first, then Tasks, then Extraction
    if task_ids:
        # Delete all feature values associated with these tasks
        db.session.execute(
            db.text(
                "DELETE FROM feature_value WHERE feature_extraction_task_id IN :task_ids"
            ).bindparams(db.bindparam("task_ids", expanding=True)),
            {"task_ids": list(task_ids)},
        )

        # Delete all extraction tasks
        db.session.execute(
            db.text(
                "DELETE FROM feature_extraction_task WHERE feature_extraction_id = :extraction_id"
            ),
            {"extraction_id": extraction_id},
        )

    # Delete the feature extraction itself
    db.session.execute(
        db.text("DELETE FROM feature_extraction WHERE id = :extraction_id"),
        {"extraction_id": extraction_id},
    )

    db.session.commit()
    current_app.logger.info(f"Deleted feature extraction {extraction_id} from database")

    # STEP 2: Revoke all Celery tasks (running and queued)
    # Tasks already running will be terminated
    # Tasks not yet started will be prevented from starting
    # Tasks that started after deletion will exit immediately when they check for the extraction
    for task_id in celery_task_ids:
        try:
            # terminate=True kills running tasks, and marks queued tasks as revoked
            AsyncResult(task_id, app=current_app.my_celery).revoke(terminate=True)
            current_app.logger.info(f"Revoked Celery task {task_id}")
        except Exception as e:
            current_app.logger.error(f"Error revoking task {task_id}: {e}")

    # STEP 3: Clean up cached features file if it exists
    features_cache_folder = f"extraction-{extraction_id}"
    features_cache_path = f"{FEATURES_CACHE_BASE_DIR}/{features_cache_folder}"
    if Path(features_cache_path).exists():
        try:
            import shutil

            shutil.rmtree(features_cache_path)
            current_app.logger.info(f"Deleted cached features at {features_cache_path}")
        except Exception as e:
            current_app.logger.error(f"Error deleting cached features: {e}")

    # Return success with previous extraction if available
    return jsonify(
        {
            "cancelled": True,
            "message": f"Feature extraction {extraction_id} cancelled and deleted",
            "previous_extraction": (
                format_extraction(previous_extraction, tasks=True)
                if previous_extraction
                else None
            ),
        }
    )


# Get feature details for a given extraction
# INCLUDING the data for the chart (to improve performance)
@bp.route("/extractions/<extraction_id>/feature-details")
def extraction_features_by_id(extraction_id):
    token = g.token

    tic()
    extraction = FeatureExtraction.find_by_id(extraction_id)
    album = Album.find_by_album_id(extraction.album_id)
    album_outcome = AlbumOutcome.find_by_album_user_id(extraction.album_id, g.user)
    logger.debug("DB lookups took %.3fs", toc())

    # studies is only needed on a cache miss — get_features_cache_or_db handles it internally
    tic()
    header, features_df = get_features_cache_or_db(extraction, token)
    logger.debug("get_features_cache_or_db total took %.3fs", toc())

    label_category = None
    labels = []

    # Get labels (if current outcome is defined for this user)
    if album_outcome and album_outcome.outcome_id:
        label_category = LabelCategory.find_by_id(album_outcome.outcome_id)
        labels = Label.find_by_label_category(label_category.id)

    # Run all CPU-bound pandas work in a real OS thread via tpool.execute.
    # This yields the eventlet hub so other concurrent greenlets (other users)
    # are not blocked while heavy computation runs.
    tic()
    chart_df = eventlet.tpool.execute(
        format_chart_data, features_df, label_category, labels
    )
    logger.debug("format_chart_data (tpool) took %.3fs", toc())

    tic()
    rounded_df, rounded_chart_df = eventlet.tpool.execute(
        lambda: (features_df.round(3), chart_df.round(3))
    )
    logger.debug("DataFrame rounding (tpool) took %.3fs", toc())

    tic()
    features_csv, chart_csv = eventlet.tpool.execute(
        lambda: (
            rounded_df.to_csv(index=False),
            rounded_chart_df.to_csv(index_label="FeatureID"),
        )
    )
    logger.debug("CSV serialization (tpool) took %.3fs", toc())

    m = MultipartEncoder(
        fields={
            "features_tabular": features_csv,
            "features_chart": chart_csv,
        }
    )
    return Response(m.to_string(), mimetype=m.content_type)


def get_features_cache_or_db(extraction, token):
    features_cache_folder = f"extraction-{extraction.id}"
    features_cache_path = f"{FEATURES_CACHE_BASE_DIR}/{features_cache_folder}"
    features_file_name = "features.h5"
    features_key = "features"
    features_cache_file_path = f"{features_cache_path}/{features_file_name}"

    # Does the features file exist?
    if Path(features_cache_file_path).exists():
        # read_hdf is disk-bound — run in tpool so event loop stays responsive
        tic()
        features_df = eventlet.tpool.execute(
            pandas.read_hdf, features_cache_file_path, features_key
        )
        header = features_df.columns
        logger.debug("Parsing features from cached HDF5 file (tpool) took %.3fs", toc())
    else:
        # Cache miss — only now fetch studies from Kheops (avoids the HTTP call on every cached request)
        tic()
        studies = get_studies_from_album(extraction.album_id, token)
        logger.debug(
            "get_studies_from_album (Kheops HTTP, cache miss only) took %.3fs", toc()
        )

        header, features_df = transform_studies_features_to_df(extraction, studies)

        # Persist features DataFrame for caching — to_hdf is disk-bound, run in tpool
        tic()
        _cache_path = features_cache_path
        _cache_file = features_cache_file_path
        _key = features_key
        _df = features_df

        def _write_hdf():
            os.makedirs(_cache_path, exist_ok=True)
            _df.to_hdf(_cache_file, _key, "w", format="fixed")

        eventlet.tpool.execute(_write_hdf)
        logger.debug("Serializing features to HDF5 (tpool) took %.3fs", toc())

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

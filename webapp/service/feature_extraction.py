import os

import requests

import yaml
from ttictoc import tic, toc
from celery import chord
from flask import current_app

from config import EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR
from quantimage2_backend_common.const import QUEUE_EXTRACTION

from quantimage2_backend_common.kheops_utils import endpoints, get_token_header, dicomFields
from quantimage2_backend_common.utils import (
    MessageType,
    get_socketio_body_extraction,
    fetch_extraction_result,
)
from quantimage2_backend_common.models import (
    FeatureExtraction,
    FeatureExtractionTask,
    db,
)


def run_feature_extraction(
    user_id,
    album_id,
    album_name,
    feature_extraction_config,
    rois,
    user_token=None,
):

    tic()

    # Create Feature Extraction object
    feature_extraction = FeatureExtraction(user_id, album_id)
    feature_extraction.save_to_db()

    print(f"Creating the feature extraction object")
    print(toc())
    print(f"---------------------------------------------------------")

    task_signatures = []

    tic()

    # Assemble study UIDs
    album_studies = get_studies_from_album(album_id, user_token)
    study_uids = list(
        map(
            lambda study: study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
            album_studies,
        )
    )
    print(study_uids)

    # Save config for this extraction
    config_path = save_config(
        feature_extraction, feature_extraction_config, user_id, album_id
    )
    feature_extraction.config_file = config_path
    feature_extraction.save_to_db()

    # Create extraction task and run it
    for study_uid in study_uids:
        feature_extraction_task = FeatureExtractionTask(
            feature_extraction.id,
            study_uid,
            None,
        )
        feature_extraction_task.save_to_db()

        # Create new task signature
        task_signature = current_app.my_celery.signature(
            "quantimage2tasks.extract",
            args=[
                feature_extraction.id,
                user_id,
                feature_extraction_task.id,
                study_uid,
                album_id,
                album_name,
                user_token,
                config_path,
                rois,
            ],
            kwargs={},
            countdown=1,
            link=current_app.my_celery.signature(
                "quantimage2tasks.finalize_extraction_task",
                args=[feature_extraction.id, feature_extraction_task.id],
                queue=QUEUE_EXTRACTION,
            ),
            queue=QUEUE_EXTRACTION,
        )

        task_signatures.append(task_signature)

    print(f"Creating the task signatures (and FeatureExtractionTasks)")
    print(toc())
    print(f"---------------------------------------------------------")

    finalize_signature = current_app.my_celery.signature(
        "quantimage2tasks.finalize_extraction",
        args=[feature_extraction.id],
        queue=QUEUE_EXTRACTION,
    )

    tic()
    db.session.commit()

    print(f"Committing the session")
    print(toc())
    print(f"---------------------------------------------------------")

    # Start the tasks as a chord
    job = chord(task_signatures, body=finalize_signature).apply_async(countdown=1)
    job.parent.save()

    # Persist group result manually, because by default it's using 24 hours
    # (not clear why, it should respect the result_expires setting used for normal results)
    current_app.my_celery.backend.client.persist(
        current_app.my_celery.backend.get_key_for_group(job.parent.id)
    )

    tic()

    feature_extraction.result_id = job.parent.id

    extraction_status = fetch_extraction_result(
        current_app.my_celery, feature_extraction.result_id
    )

    print(f"Getting the extraction status")
    print(toc())
    print(f"---------------------------------------------------------")

    tic()
    db.session.commit()

    print(f"Committing the session (again)")
    print(toc())
    print(f"---------------------------------------------------------")

    # Send a Socket.IO message to inform that the extraction has started
    if not album_id:
        socketio_body = get_socketio_body_extraction(
            feature_extraction.id, vars(extraction_status)
        )

        current_app.my_socketio.emit(MessageType.EXTRACTION_STATUS.value, socketio_body)

    feature_extraction = FeatureExtraction.find_by_id(feature_extraction.id)

    return feature_extraction


def get_album_details(album_id, token):
    album_url = f"{endpoints.albums}/{album_id}"

    access_token = get_token_header(token)

    album_details = requests.get(album_url, headers=access_token).json()

    return album_details


def get_studies_from_album(album_id, token):
    album_studies_url = f"{endpoints.studies}?{endpoints.album_parameter}={album_id}"

    access_token = get_token_header(token)

    album_studies = requests.get(album_studies_url, headers=access_token).json()

    return album_studies


def get_series_from_study(study_uid, modalities, album_id, token):
    study_series_url = f"{endpoints.studies}/{study_uid}/series?album={album_id}"

    if modalities is not None:
        params = [
            f"&{dicomFields.MODALITY}={modality}"
            for i, modality in enumerate(modalities)
        ]

        study_series_url += "".join(params)

    access_token = get_token_header(token)

    study_series = requests.get(study_series_url, headers=access_token).json()

    return study_series


def get_series_metadata(study_uid, series_uid, album_id, token):
    series_metadata_url = (
        f"{endpoints.studies}/{study_uid}/series/{series_uid}/metadata?album={album_id}"
    )

    access_token = get_token_header(token)

    series_metadata = requests.get(series_metadata_url, headers=access_token).json()

    return series_metadata


def save_config(feature_extraction, feature_config, user_id, album_id):
    # Define config path for storing the feature family configuration
    config_dir = os.path.join(EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, user_id, album_id)

    config_filename = f"extraction-{feature_extraction.id}-config.yaml"
    config_path = os.path.join(config_dir, config_filename)

    # Save the customized config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    stream = open(config_path, "w")
    yaml.dump(feature_config, stream)
    stream.close()

    return config_path

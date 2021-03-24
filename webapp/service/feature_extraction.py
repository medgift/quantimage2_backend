import json
import os

import requests

from pathlib import Path
from ttictoc import tic, toc
from celery import chord
from flask import current_app

from config import EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, FEATURES_SUBDIR

from imaginebackend_common.kheops_utils import endpoints, get_token_header, dicomFields
from imaginebackend_common.utils import (
    MessageType,
    get_socketio_body_extraction,
    fetch_extraction_result,
)
from imaginebackend_common.models import (
    FeatureExtraction,
    FeatureFamily,
    FeatureExtractionFamily,
    FeatureExtractionTask,
    db,
)


def run_feature_extraction(
    user_id, album_id, album_name, feature_families_map, token=None
):

    tic()

    # Create Feature Extraction object
    feature_extraction = FeatureExtraction(user_id, album_id)
    feature_extraction.save_to_db()

    print(f"Creating the feature extraction object")
    print(toc())
    print(f"---------------------------------------------------------")

    # For each feature family, create the association with the extraction
    # as well as the feature extraction task
    task_signatures = []

    tic()

    # Assemble study UIDs
    album_studies = get_studies_from_album(album_id, token)
    study_uids = list(
        map(
            lambda study: study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
            album_studies,
        )
    )
    print(study_uids)

    # Go through each study assembled above

    for feature_family_id in feature_families_map:
        # Convert feature family ID to int
        feature_family_id_int = int(feature_family_id)

        # Get the feature family from the DB
        feature_family = FeatureFamily.query.get(feature_family_id_int)

        # Save config for this extraction & family
        feature_config = feature_families_map[feature_family_id]
        config_path = save_config(
            feature_extraction, feature_family, feature_config, user_id, album_id,
        )

        # Create the association
        feature_extraction_family = FeatureExtractionFamily(
            feature_extraction=feature_extraction,
            feature_family=feature_family,
            family_config_path=config_path,
        )

        # Save the association to the DB
        feature_extraction_family.save_to_db()

        # Create extraction task and run it
        for study_uid in study_uids:
            # features_path = get_features_path(user_id, study_uid, feature_family.name)

            feature_extraction_task = FeatureExtractionTask(
                feature_extraction.id,
                study_uid,
                feature_family_id_int,
                None,
                # features_path,
            )
            feature_extraction_task.save_to_db()

            # Create new task signature
            task_signature = current_app.my_celery.signature(
                "imaginetasks.extract",
                args=[
                    feature_extraction.id,
                    user_id,
                    feature_extraction_task.id,
                    study_uid,
                    # features_path,
                    album_name,
                    config_path,
                ],
                kwargs={},
                countdown=1,
                link=current_app.my_celery.signature(
                    "imaginetasks.finalize_extraction_task",
                    args=[feature_extraction.id, feature_extraction_task.id],
                ),
            )

            task_signatures.append(task_signature)

    print(f"Creating the task signatures (and FeatureExtractionTasks)")
    print(toc())
    print(f"---------------------------------------------------------")

    finalize_signature = current_app.my_celery.signature(
        "imaginetasks.finalize_extraction", args=[feature_extraction.id],
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


# def get_features_path(user_id, study_uid, feature_family_name):
#     # Define features path for storing the results
#     features_dir = os.path.join(
#         EXTRACTIONS_BASE_DIR, FEATURES_SUBDIR, user_id, study_uid
#     )
#     features_filename = feature_family_name + ".json"
#     features_path = os.path.join(features_dir, features_filename)
#
#     return features_path


def save_config(feature_extraction, feature_family, feature_config, user_id, album_id):
    # Define config path for storing the feature family configuration
    config_dir = os.path.join(EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, user_id, album_id)

    config_filename = feature_family.name + ".json"
    config_path = os.path.join(config_dir, config_filename)

    # Save the customized config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    Path(config_path).write_text(json.dumps(feature_config))

    return config_path

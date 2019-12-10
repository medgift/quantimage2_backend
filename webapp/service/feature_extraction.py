import json
import os
import traceback
from pathlib import Path

import eventlet
from eventlet import tpool
from celery import group

from ..routes.utils import (
    fetch_task_result,
    read_feature_file,
    DATE_FORMAT,
    task_status_update,
    task_status_message,
    get_socketio_body_extraction,
    MessageType,
    fetch_extraction_result,
)
from ..config import EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, FEATURES_SUBDIR
from ..models import (
    FeatureExtraction,
    FeatureFamily,
    FeatureExtractionFamily,
    FeatureExtractionTask,
)

from .. import my_socketio
from .. import my_celery


def run_feature_extraction(
    token, user_id, album_id, feature_families_map, study_uid=None
):

    # Create Feature Extraction object
    feature_extraction = FeatureExtraction(user_id, album_id, study_uid)
    feature_extraction.save_to_db()

    extraction_status = fetch_extraction_result(feature_extraction.result_id)

    # Send a Socket.IO message to inform that the extraction has started
    socketio_body = get_socketio_body_extraction(
        feature_extraction.id, vars(extraction_status)
    )
    my_socketio.emit(MessageType.EXTRACTION_STATUS.value, socketio_body)

    # For each feature family, create the association with the extraction
    # as well as the feature extraction task

    task_signatures = []

    for feature_family_id in feature_families_map:

        # Convert feature family ID to int
        feature_family_id_int = int(feature_family_id)

        # Get the feature family from the DB
        feature_family = FeatureFamily.query.get(feature_family_id_int)

        # Save config for this extraction & family
        feature_config = feature_families_map[feature_family_id]
        config_path = save_config(
            feature_extraction,
            feature_family,
            feature_config,
            user_id,
            album_id,
            study_uid,
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
        features_path = get_features_path(user_id, study_uid, feature_family.name)

        feature_extraction_task = FeatureExtractionTask(
            feature_extraction.id, study_uid, feature_family_id_int, None, features_path
        )
        feature_extraction_task.save_to_db()

        # Create new task signature
        task_signature = my_celery.signature(
            "imaginetasks.extract",
            args=[
                token,
                feature_extraction_task.id,
                study_uid,
                features_path,
                config_path,
            ],
            kwargs={},
        )

        task_signatures.append(task_signature)

    # Start the tasks as a group
    job = group(task_signatures)
    group_result = job.apply_async(countdown=1)
    group_result.save()

    feature_extraction.result_id = group_result.id
    feature_extraction.save_to_db()

    # Spawn green thread to follow the job's progress
    eventlet.spawn(follow_job, group_result, feature_extraction.id)

    # TODO - Remove these alternative ways to spawn a thread to follow the job
    # my_socketio.start_background_task(follow_job, group_result, feature_extraction.id)
    # tpool.execute(follow_job, group_result, feature_extraction.id)

    feature_extraction = FeatureExtraction.find_by_id(feature_extraction.id)

    return feature_extraction


def job_done():
    print("awesome!")


def format_feature_tasks(feature_tasks):
    # Gather the feature tasks
    feature_task_list = []

    if feature_tasks:
        for feature_task in feature_tasks:
            formatted_feature_task = format_feature_task(feature_task)
            feature_task_list.append(formatted_feature_task)

    return feature_task_list


def format_feature_task(feature_task):
    status = ""
    status_message = ""
    sanitized_object = {}

    # Get the feature status & update the status if necessary!
    if feature_task.task_id:
        status_object = fetch_task_result(feature_task.task_id)
        result = status_object.result

        # Get the status message for the task
        status_message = task_status_message(result)

        status = status_object.status

        # Read the features file (if available)
        sanitized_object = read_feature_file(feature_task.features_path)

    # Read the config file (if available)
    # config = read_config_file(feature.config_path)

    return {
        "id": feature_task.id,
        "updated_at": feature_task.updated_at.strftime(DATE_FORMAT),
        "status": status,
        "status_message": status_message,
        "payload": sanitized_object,
        "study_uid": feature_task.study_uid,
        # "feature_family_id": feature.feature_family_id,
        "feature_family": feature_task.feature_family.to_dict(),
        # "config": json.loads(config),
    }


def get_features_path(user_id, study_uid, feature_family_name):
    # Define features path for storing the results
    features_dir = os.path.join(
        EXTRACTIONS_BASE_DIR, FEATURES_SUBDIR, user_id, study_uid
    )
    features_filename = feature_family_name + ".json"
    features_path = os.path.join(features_dir, features_filename)

    return features_path


def save_config(
    feature_extraction, feature_family, feature_config, user_id, album_id, study_uid
):
    # Define config path for storing the feature family configuration
    if feature_extraction.study_uid:
        config_dir = os.path.join(
            EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, user_id, study_uid
        )
    else:
        config_dir = os.path.join(
            EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, user_id, album_id
        )

    config_filename = feature_family.name + ".json"
    config_path = os.path.join(config_dir, config_filename)

    # Save the customized config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    Path(config_path).write_text(json.dumps(feature_config))

    return config_path


def follow_job(group_result, feature_extraction_id):
    print(
        f"Feature extraction {feature_extraction_id} - STARTING TO LISTEN FOR EVENTS!"
    )

    try:
        # Proxy the group result with the eventlet thread pool
        # This is to ensure that the "get" method will not be interrupted
        proxy = tpool.Proxy(group_result)
        proxy.get(on_message=task_status_update)
        print(f"Feature extraction {feature_extraction_id} - DONE!")

        # When the process ends, send Socket.IO message informing of the end
        extraction_status = fetch_extraction_result(group_result.id)

        socketio_body = get_socketio_body_extraction(
            feature_extraction_id, vars(extraction_status),
        )

        # Handle updating the extraction via Socket.IO
        my_socketio.emit(MessageType.EXTRACTION_STATUS.value, socketio_body)

        return group_result

    except Exception as e:
        print(e)
        traceback.print_exception(type(e), e, e.__traceback__)
        return None

    # TODO - Deal with errors in different tasks, how it should affect the job overall
    # socketio_body = get_socketio_body(result.id, celery_states.FAILURE, str(e))
    # my_socketio.emit("feature-status", socketio_body)
    #    print(e)
    #    return

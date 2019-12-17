import json
import os
from pathlib import Path

from ttictoc import TicToc

from celery import group, chord, states as celerystates

from imaginebackend_common.utils import (
    task_status_message_from_result,
    MessageType,
    get_socketio_body_extraction,
    StatusMessage,
)
from ..routes.utils import (
    fetch_task_result,
    read_feature_file,
    DATE_FORMAT,
    fetch_extraction_result,
    read_config_file,
)
from ..config import EXTRACTIONS_BASE_DIR, CONFIGS_SUBDIR, FEATURES_SUBDIR
from imaginebackend_common.models import (
    FeatureExtraction,
    FeatureFamily,
    FeatureExtractionFamily,
    FeatureExtractionTask,
    db,
)

from .. import my_socketio
from .. import my_celery


def run_feature_extraction(user_id, album_id, feature_families_map, study_uid=None):

    t = TicToc(print_toc=True)

    t.tic()

    # Create Feature Extraction object
    feature_extraction = FeatureExtraction(user_id, album_id, study_uid)
    feature_extraction.flush_to_db()

    print(f"Creating the feature extraction object")
    t.toc()
    print(f"---------------------------------------------------------")

    # For each feature family, create the association with the extraction
    # as well as the feature extraction task
    task_signatures = []

    t.tic()

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
        feature_extraction_family.flush_to_db()

        # Create extraction task and run it
        features_path = get_features_path(user_id, study_uid, feature_family.name)

        feature_extraction_task = FeatureExtractionTask(
            feature_extraction.id, study_uid, feature_family_id_int, None, features_path
        )
        feature_extraction_task.flush_to_db()

        # Create new task signature
        task_signature = my_celery.signature(
            "imaginetasks.extract",
            args=[
                user_id,
                feature_extraction_task.id,
                study_uid,
                features_path,
                config_path,
            ],
            kwargs={},
            countdown=1,
            link=my_celery.signature(
                "imaginetasks.finalize_extraction_task",
                args=[feature_extraction_task.id],
            ),
        )

        task_signatures.append(task_signature)

    print(f"Creating the task signatures (and FeatureExtractionTasks)")
    t.toc()
    print(f"---------------------------------------------------------")

    finalize_signature = my_celery.signature(
        "imaginetasks.finalize_extraction", args=[feature_extraction.id],
    )

    t.tic()
    db.session.commit()

    print(f"Committing the session")
    t.toc()
    print(f"---------------------------------------------------------")

    # Start the tasks as a group
    group_tasks = group(task_signatures)
    job = chord(group_tasks, finalize_signature).apply_async()
    job.parent.save()
    # group_result = job.apply_async(countdown=1)
    # group_result.save()

    t.tic()

    feature_extraction.result_id = job.parent.id

    extraction_status = fetch_extraction_result(feature_extraction.result_id)

    print(f"Getting the extraction status")
    t.toc()
    print(f"---------------------------------------------------------")

    t.tic()
    db.session.commit()

    print(f"Committing the session (again)")
    t.toc()
    print(f"---------------------------------------------------------")

    # Send a Socket.IO message to inform that the extraction has started
    socketio_body = get_socketio_body_extraction(
        feature_extraction.id, vars(extraction_status)
    )
    my_socketio.emit(MessageType.EXTRACTION_STATUS.value, socketio_body)

    feature_extraction = FeatureExtraction.find_by_id(feature_extraction.id)

    return feature_extraction


def format_feature_families(families):
    # Gather the feature families
    families_list = []

    if families:
        for family in families:
            formatted_family = format_family(family)
            families_list.append(formatted_family)

    return families_list


def format_family(family):
    config_str = read_config_file(family.family_config_path)

    config = json.loads(config_str)

    return {**family.to_dict(), "config": config}


def format_feature_tasks(feature_tasks):
    # Gather the feature tasks
    feature_task_list = []

    if feature_tasks:
        for feature_task in feature_tasks:
            formatted_feature_task = format_feature_task(feature_task)
            feature_task_list.append(formatted_feature_task)

    return feature_task_list


def format_feature_task(feature_task):
    status = celerystates.PENDING
    status_message = StatusMessage.WAITING_TO_START.value
    sanitized_object = {}

    # Get the feature status & update the status if necessary!
    if feature_task.task_id:
        status_object = fetch_task_result(feature_task.task_id)
        result = status_object.result

        print(
            f"Got Task n°{feature_task.id} Result (task id {feature_task.task_id}) : {result}"
        )

        # If the task is in progress, show the corresponding message
        # Otherwise it is still pending
        if result:
            status = status_object.status
            status_message = task_status_message_from_result(result)

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

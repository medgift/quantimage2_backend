import json
import os
import sys
from collections import OrderedDict
from enum import Enum


# Exceptions
from pathlib import Path
from ttictoc import tic, toc

import celery.states as celerystates
import jsonpickle
import requests
from celery import Celery
from flask import jsonify
from numpy.core.records import ndarray

from imaginebackend_common.models import FeatureExtraction

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"

# Celery instance
celery = Celery(
    "tasks",
    backend=os.environ["CELERY_RESULT_BACKEND"],
    broker=os.environ["CELERY_BROKER_URL"],
)


class CustomResult(object):
    pass


class CustomException(Exception):
    status_code = 500

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


class InvalidUsage(CustomException):
    def __init__(self, message):
        CustomException.__init__(self, message, 400)


class ComputationError(CustomException):
    def __init__(self, message):
        CustomException.__init__(self, message, 500)


# Socket.IO messages
class MessageType(Enum):
    FEATURE_TASK_STATUS = "feature-status"
    EXTRACTION_STATUS = "extraction-status"


# Format feature extraction
def format_extraction(extraction, payload=False):
    extraction_dict = extraction.to_dict()

    status = fetch_extraction_result(celery, extraction.result_id)
    extraction_dict["status"] = vars(status)

    formatted_families = {"families": format_feature_families(extraction.families)}
    formatted_tasks = {"tasks": format_feature_tasks(extraction.tasks, payload)}

    extraction_dict.update(formatted_families)
    extraction_dict.update(formatted_tasks)

    return extraction_dict


# Format feature families
def format_feature_families(families):
    # Gather the feature families
    families_list = []

    if families:
        for family in families:
            formatted_family = format_family(family)
            families_list.append(formatted_family)

    return families_list


# Format feature family
def format_family(family):
    config_str = read_config_file(family.family_config_path)

    config = json.loads(config_str)

    return {**family.to_dict(), "config": config}


# Format feature tasks
def format_feature_tasks(feature_tasks, payload=False):
    # Gather the feature tasks
    feature_task_list = []

    if feature_tasks:
        for feature_task in feature_tasks:
            formatted_feature_task = format_feature_task(feature_task, payload)
            feature_task_list.append(formatted_feature_task)

    return feature_task_list


# Formate feature task
def format_feature_task(feature_task, payload=True):
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

            if status != celerystates.FAILURE:
                status_message = task_status_message_from_result(result)
            else:
                status_message = result

        # Read the features file (if available)
        if payload:
            sanitized_object = read_feature_file(feature_task.features_path)

    # Read the config file (if available)
    # config = read_config_file(feature.config_path)

    response_dict = {
        "id": feature_task.id,
        "updated_at": feature_task.updated_at.strftime(DATE_FORMAT),
        "status": status,
        "status_message": status_message,
        "study_uid": feature_task.study_uid,
        # "feature_family_id": feature.feature_family_id,
        "feature_family": feature_task.feature_family.to_dict(),
        # "config": json.loads(config),
    }

    # If the payload should be included, add it to the response dictionary
    if payload:
        response_dict["payload"] = sanitized_object

    return response_dict


# Read Config File
def read_config_file(config_path):
    try:
        config = Path(config_path).read_text()
        return config
    except FileNotFoundError:
        print(f"{config_path} does not exist!")


# Read Feature File
def read_feature_file(feature_path):

    features_dict = {}
    if feature_path:
        try:
            tic()
            features_dict = jsonpickle.decode(open(feature_path).read())
            elapsed = toc()
            # TODO - Remove this once the serialization has been improved!
            # print(f"Unpickling took {elapsed}s")

            for modality, labels in features_dict.items():
                for label, features in labels.items():
                    features_dict[modality][label] = sanitize_features_object(
                        features_dict[modality][label]
                    )
        except FileNotFoundError:
            print(f"{feature_path} does not exist!")

    sanitized_features_dict = features_dict

    return sanitized_features_dict


# Sanitize features object
def sanitize_features_object(feature_object):
    sanitized_object = OrderedDict()

    for feature_name in feature_object:
        if is_jsonable(feature_object[feature_name]):
            sanitized_object[feature_name] = feature_object[feature_name]
        else:
            # Numpy NDArrays
            if type(feature_object[feature_name] is ndarray):
                sanitized_object[feature_name] = feature_object[feature_name].tolist()
            else:
                print(feature_name + " is unsupported", file=sys.stderr)

    return sanitized_object


# Get Extraction Status
def fetch_extraction_result(celery_app, result_id):
    status = ExtractionStatus()

    if result_id is not None:
        print(f"Getting result for extraction result {result_id}")

        result = celery_app.GroupResult.restore(result_id)

        one_started_task = next(
            (task for task in result.children if type(task.info) is dict), None
        )

        total_steps = (
            one_started_task.info["total"] * len(result.children)
            if one_started_task is not None
            else 0
        )

        completed_steps = sum(
            [
                task.info["completed"]
                for task in result.children
                if type(task.info) is dict
            ]
        )
        pending_tasks = [
            task for task in result.children if task.status == celerystates.PENDING
        ]

        status = ExtractionStatus(
            result.ready(),
            result.successful(),
            result.failed(),
            len(result.children),
            len(pending_tasks),
            result.completed_count(),
            total_steps,
            completed_steps,
        )

    return status


# Get feature extraction task result
def fetch_task_result(task_id):
    print(f"Getting result for task {task_id}")

    response = requests.get("http://flower:5555/api/task/result/" + task_id)

    task = CustomResult()

    if response.ok:
        body = response.json()
        task.status = body["state"]
        task.result = body["result"]
        print(f"State is : {body['state']}")
        print(f"Result is : {body['result']}")
        return task
    else:
        print(f"NO RESULT FOUND FOR TASK ID {task_id}, WHAT GIVES?!")
        print(response.status_code)
        print(response.text)
        task = CustomResult()
        task.status = celerystates.PENDING
        task.result = None
        return task


# Status messages
class StatusMessage(Enum):
    WAITING_TO_START = "Waiting to start"


def get_socketio_body_feature_task(
    task_id,
    feature_extraction_task_id,
    status,
    status_message,
    updated_at=None,
    payload=None,
):
    socketio_body = {
        "task_id": task_id,
        "feature_extraction_task_id": feature_extraction_task_id,
        "status": status,
        "status_message": status_message,
    }

    if updated_at:
        # Set the new updated date when complete
        socketio_body["updated_at"] = updated_at

    if payload:
        # Set the new feature payload when complete
        socketio_body["payload"] = payload

    return socketio_body


def get_socketio_body_extraction(feature_extraction_id, status):
    socketio_body = {
        "feature_extraction_id": feature_extraction_id,
        "status": status,
    }

    return socketio_body


def send_extraction_status_message(
    feature_extraction_id, celery, socketio, send_extraction=False
):
    feature_extraction = FeatureExtraction.find_by_id(feature_extraction_id)

    print("Send extraction is  " + str(send_extraction))
    if send_extraction:
        print(f"Sending whole feature extraction object with tasks and families etc. !")
        socketio_body = format_extraction(feature_extraction)
    else:
        extraction_status = fetch_extraction_result(
            celery, feature_extraction.result_id
        )

        # Send Socket.IO message
        socketio_body = get_socketio_body_extraction(
            feature_extraction_id, vars(extraction_status)
        )

    print(
        "Emitting EXTRACTION STATUS WITH "
        + json.dumps(jsonify(socketio_body).get_json())
    )
    socketio.emit(
        MessageType.EXTRACTION_STATUS.value, jsonify(socketio_body).get_json()
    )


# Feature Tasks
def task_status_message(current_step, total_steps, status_message):
    return f"{current_step}/{total_steps} - {status_message}"


def task_status_message_from_result(task_result):
    return task_status_message(
        task_result["current"], task_result["total"], task_result["status_message"]
    )


# Feature Extraction Status
class ExtractionStatus:
    ready = False
    successful = False
    failed = False
    total_tasks = 0
    completed_tasks = 0
    total_steps = 0
    completed_steps = 0

    def __init__(
        self,
        ready=False,
        successful=False,
        failed=False,
        total_tasks=0,
        pending_tasks=0,
        completed_tasks=0,
        total_steps=0,
        completed_steps=0,
    ):
        self.ready = ready
        self.successful = successful
        self.failed = failed
        self.total_tasks = total_tasks
        self.pending_tasks = pending_tasks
        self.completed_tasks = completed_tasks
        self.total_steps = total_steps
        self.completed_steps = completed_steps


# Misc
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

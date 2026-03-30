import json
import logging
import math
import os
from enum import Enum

logger = logging.getLogger(__name__)

# Exceptions
from pathlib import Path

import numpy as np
from numpy import ndarray
from sklearn.model_selection import RepeatedStratifiedKFold
from ttictoc import tic, toc

import celery.states as celerystates
import requests
from celery import Celery

from quantimage2_backend_common.models import FeatureExtraction

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"

CV_SPLITS = 5
CV_REPEATS = 1

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
        super().__init__(message)
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
        super().__init__(message, 400)


class ComputationError(CustomException):
    def __init__(self, message):
        super().__init__(message, 500)


# Socket.IO messages
class MessageType(Enum):
    FEATURE_TASK_STATUS = "feature-status"
    EXTRACTION_STATUS = "extraction-status"
    TRAINING_STATUS = "training-status"


# Format feature extraction
def format_extraction(extraction, tasks=False):
    extraction_dict = extraction.to_dict()

    tic()
    status = fetch_extraction_result(
        celery, extraction.result_id, tasks=extraction.tasks
    )
    extraction_dict["status"] = vars(status)
    elapsed = toc()
    print("Getting extraction result for formatted extraction took", elapsed)

    if tasks:
        formatted_tasks = {"tasks": format_feature_tasks(extraction.tasks)}
        extraction_dict.update(formatted_tasks)

    return extraction_dict


# Format feature tasks — batch-fetches all Celery results in ONE Redis MGET
# instead of N sequential HTTP requests to Flower (was ~70ms × N = 10-15s for large albums)
def format_feature_tasks(feature_tasks):
    if not feature_tasks:
        return []

    # 1. Collect task IDs that exist
    task_ids = [t.task_id for t in feature_tasks if t.task_id]

    # 2. Batch-fetch ALL results from Redis in a single round-trip
    batch_results = {}
    if task_ids:
        tic()
        try:
            # celery.backend.get_many() issues a single Redis MGET pipeline
            for task_id, meta in celery.backend.get_many(task_ids, timeout=10):
                result_obj = CustomResult()
                if meta:
                    result_obj.status = meta.get("status", celerystates.PENDING)
                    result_obj.result = meta.get("result")
                else:
                    result_obj.status = celerystates.PENDING
                    result_obj.result = None
                batch_results[task_id] = result_obj
        except Exception as e:
            logger.warning(
                "Batch task result fetch failed (%s), falling back to per-task requests",
                e,
            )
            for task_id in task_ids:
                batch_results[task_id] = fetch_task_result(task_id)
        elapsed = toc()
        logger.debug("Batch-fetched %d task results in %.3fs", len(task_ids), elapsed)

    # 3. Format each task using the pre-fetched result
    feature_task_list = []
    for feature_task in feature_tasks:
        result_obj = (
            batch_results.get(feature_task.task_id) if feature_task.task_id else None
        )
        feature_task_list.append(
            _format_feature_task_with_result(feature_task, result_obj)
        )

    return feature_task_list


def _format_feature_task_with_result(feature_task, result_obj):
    """Format a single feature task given a pre-fetched CustomResult (or None)."""
    status = celerystates.PENDING
    status_message = StatusMessage.WAITING_TO_START.value

    if result_obj is not None:
        result = result_obj.result
        if result:
            status = result_obj.status
            if status != celerystates.FAILURE:
                status_message = task_status_message_from_result(result)
            else:
                status_message = result

    return {
        "id": feature_task.id,
        "updated_at": feature_task.updated_at.strftime(DATE_FORMAT),
        "status": status,
        "status_message": status_message,
        "study_uid": feature_task.study_uid,
    }


# Format a single feature task (kept for backward compatibility; prefer format_feature_tasks for bulk)
def format_feature_task(feature_task):
    status = celerystates.PENDING
    status_message = StatusMessage.WAITING_TO_START.value

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

    return {
        "id": feature_task.id,
        "updated_at": feature_task.updated_at.strftime(DATE_FORMAT),
        "status": status,
        "status_message": status_message,
        "study_uid": feature_task.study_uid,
    }


# Read Config File
def read_config_file(config_path):
    try:
        config = Path(config_path).read_text()
        return config
    except FileNotFoundError:
        print(f"{config_path} does not exist!")


# Get Extraction Status
def fetch_extraction_result(celery_app, result_id, tasks=None):
    status = ExtractionStatus()
    errors = {}

    if result_id is not None:
        tic()
        result = celery_app.GroupResult.restore(result_id)
        elapsed = toc()
        print(f"Getting result for extraction result {result_id} took", elapsed)

        # Make an inventory of errors (if tasks are provided)
        if tasks is not None:

            for child in result.children:
                if isinstance(child.info, Exception):
                    task = next(filter(lambda t: t.task_id == child.task_id, tasks))
                    study_uid = task.study_uid

                    if study_uid not in errors:
                        errors[study_uid] = []

                    if not str(child.info) in errors[study_uid]:
                        errors[study_uid].append(str(child.info))

        pending_tasks = [
            task for task in result.children if task.status == celerystates.PENDING
        ]

        failed_tasks = [
            task for task in result.children if task.status == celerystates.FAILURE
        ]

        status = ExtractionStatus(
            result.ready(),
            result.successful(),
            result.failed(),
            len(result.children),
            len(pending_tasks),
            result.completed_count(),
            len(failed_tasks),
            errors=errors,
        )

    return status


# Get feature extraction task result
def fetch_task_result(task_id):
    print(f"Getting result for task {task_id}")

    response = requests.get("http://flower:3333/api/task/result/" + task_id)

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

    # If extraction was deleted (cancelled), silently skip status update
    if not feature_extraction:
        print(
            f"Feature extraction {feature_extraction_id} not found (likely cancelled)"
        )
        return

    print("Send extraction is  " + str(send_extraction))
    if send_extraction:
        print(f"Sending whole feature extraction object with tasks etc. !")
        socketio_body = format_extraction(feature_extraction, tasks=True)
    else:
        extraction_status = fetch_extraction_result(
            celery, feature_extraction.result_id, tasks=feature_extraction.tasks
        )

        # Send Socket.IO message
        socketio_body = get_socketio_body_extraction(
            feature_extraction_id, vars(extraction_status)
        )

    print("Emitting EXTRACTION STATUS WITH " + json.dumps(socketio_body, default=str))
    socketio.emit(MessageType.EXTRACTION_STATUS.value, socketio_body)


# Feature Tasks
def task_status_message(current_step, total_steps, status_message):
    return f"{current_step}/{total_steps} - {status_message}"


def task_status_message_from_result(task_result):
    return task_status_message(
        task_result["current"], task_result["total"], task_result["status_message"]
    )


# Get Training ID based on feature extration & (optionnally) collection IDs
def get_training_id(feature_extraction_id, collection_id=None):
    training_id = (
        f"train_{feature_extraction_id}_{collection_id}"
        if collection_id
        else f"train_{feature_extraction_id}"
    )

    return training_id


# Format ML models
def format_model(model):

    model_dict = model.to_dict()

    # Convert metrics to native Python types
    training_metrics = format_metrics(model.training_metrics)
    test_metrics = format_metrics(model.test_metrics) if model.test_metrics else None

    model_dict["training-metrics"] = training_metrics
    model_dict["test-metrics"] = test_metrics if test_metrics else None

    return model_dict


# Format model metrics
def format_metrics(metrics):

    formatted_metrics = {**metrics}

    for metric in metrics:

        # Do we have a range or just a single value for the metric?
        if np.isscalar(metrics[metric]):
            if math.isnan(metrics[metric]):
                formatted_metrics[metric] = "N/A"
        else:
            for value in formatted_metrics[metric]:
                if math.isnan(metrics[metric][value]):
                    formatted_metrics[metric][value] = "N/A"

    return formatted_metrics


# Feature Extraction Status
class ExtractionStatus:
    ready = False
    successful = False
    failed = False
    total_tasks = 0
    completed_tasks = 0
    failed_tasks = 0
    errors = None
    # total_steps = 0
    # completed_steps = 0
    # failed_steps = 0

    def __init__(
        self,
        ready=False,
        successful=False,
        failed=False,
        total_tasks=0,
        pending_tasks=0,
        completed_tasks=0,
        failed_tasks=0,
        errors=None,
        # total_steps=0,
        # completed_steps=0,
    ):
        self.ready = ready
        self.successful = successful
        self.failed = failed
        self.total_tasks = total_tasks
        self.pending_tasks = pending_tasks
        self.completed_tasks = completed_tasks
        self.failed_tasks = failed_tasks
        self.errors = errors
        # self.total_steps = total_steps
        # self.completed_steps = completed_steps


# Misc
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

import json
import os
import sys
from collections import OrderedDict
from enum import Enum
from pathlib import Path

import jsonpickle
import requests
import celery.states as celery_states

from flask import abort, g
from numpy.core.records import ndarray

from ..models import FeatureExtractionTask

from .. import my_socketio
from .. import my_celery
from ..config import keycloak_client


class CustomResult(object):
    pass


class ExtractionStatus:
    ready = False
    successful = False
    failed = False
    completed_count = 0

    def __init__(self, ready=False, successful=False, failed=False, completed_count=0):
        self.ready = ready
        self.successful = successful
        self.failed = failed
        self.completed_count = completed_count


# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


class MessageType(Enum):
    FEATURE_TASK_STATUS = "feature-status"
    EXTRACTION_STATUS = "extraction-status"


def validate_decorate(request):
    if request.method != "OPTIONS":
        if not validate_request(request):
            abort(401)
        else:
            g.user = userid_from_token(request.headers["Authorization"].split(" ")[1])
            g.token = request.headers["Authorization"].split(" ")[1]
        pass

    pass


def validate_request(request):
    authorization = request.headers["Authorization"]

    if not authorization.startswith("Bearer"):
        abort(400)
    else:
        token = authorization.split(" ")[1]
        # rpt = keycloak_client.entitlement(token, "resource_id")
        validated = keycloak_client.introspect(token)
        return validated["active"]


def userid_from_token(token):
    secret = f"-----BEGIN PUBLIC KEY-----\n{os.environ['KEYCLOAK_REALM_PUBLIC_KEY']}\n-----END PUBLIC KEY-----"

    # Verify signature & expiration
    options = {"verify_signature": True, "verify_aud": False, "exp": True}
    token_decoded = keycloak_client.decode_token(token, key=secret, options=options)

    id = token_decoded["sub"]

    return id


def fetch_extraction_result(result_id):
    status = ExtractionStatus()

    if result_id is not None:
        print(f"Getting result for extraction result {result_id}")

        result = my_celery.GroupResult.restore(result_id)

        status = ExtractionStatus(
            result.ready(),
            result.successful(),
            result.failed(),
            result.completed_count(),
        )

    return status


def fetch_task_result(task_id):
    print(f"Getting result for task {task_id}")

    response = requests.get("http://flower:5555/api/task/result/" + task_id)

    # task = my_celery.AsyncResult(task_id)

    task = CustomResult()

    if response.ok:
        body = response.json()
        task.status = body["state"]
        task.result = body["result"]
        return task
    else:
        task = CustomResult()
        task.status = celery_states.PENDING
        task.result = None
        return task


def task_status_message(task_result):

    # If task result has disappeared, return empty string (it's complete anyway)
    if task_result is None or "status_message" not in task_result:
        return ""

    return f"{task_result['current']}/{task_result['total']} - {task_result['status_message']}"


def task_status_update(body):

    status = body["status"]

    try:
        feature_extraction_task_id = body["result"]["feature_extraction_task_id"]

        print(
            f"Feature extraction task {feature_extraction_task_id} - Status: {status}, Message: {body['result']['status_message']}"
        )

        # Update the task ID of the feature extraction task (if not set yet)
        from ..app import flask_app

        with flask_app.app_context():
            feature_extraction_task = FeatureExtractionTask.find_by_id(
                feature_extraction_task_id
            )
            if not feature_extraction_task.task_id:
                feature_extraction_task.task_id = body["result"]["task_id"]
                feature_extraction_task.save_to_db()

            socketio_body = get_socketio_body_feature_task(
                feature_extraction_task_id, status, task_status_message(body["result"])
            )

            # Send Socket.IO message to clients
            my_socketio.emit(MessageType.FEATURE_TASK_STATUS.value, socketio_body)
    except Exception as e:
        print(f"Problem with this body : {body}")
        # print(e, file=sys.stderr)


def get_socketio_body_feature_task(
    feature_extraction_task_id, status, status_message, updated_at=None, payload=None
):
    socketio_body = {
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


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def read_feature_file(feature_path):

    sanitized_object = {}
    if feature_path:
        try:
            feature_object = jsonpickle.decode(open(feature_path).read())
            sanitized_object = sanitize_features_object(feature_object)
        except FileNotFoundError:
            print(f"{feature_path} does not exist!")

    return sanitized_object


def read_config_file(config_path):
    try:
        config = Path(config_path).read_text()
        return config
    except FileNotFoundError:
        print(f"{config_path} does not exist!")


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

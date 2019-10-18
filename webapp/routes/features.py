import os
import sys
import json
from collections import OrderedDict

import eventlet
import jsonpickle

import celery.states as celery_states
import requests
from flask import Blueprint, abort, jsonify, request, g, current_app
from numpy.core.records import ndarray

from imaginebackend_common import utils
from imaginebackend_common.utils import task_status_message, InvalidUsage

from ..config import FEATURES_BASE_DIR
from ..models import db, FeatureExtraction, Study, get_or_create, FeatureFamily

from .. import my_socketio
from .. import my_celery

from .utils import validate_decorate

# Define blueprint
bp = Blueprint(__name__, "features")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/")
def hello():
    return "Hello IMAGINE!"


@bp.route("/feature/<task_id>/status")
def feature_status(task_id):

    task = fetch_task_result(task_id)

    if task.status == celery_states.PENDING:
        abort(404)

    if task.status == celery_states.FAILURE:
        task.result

    response = {"status": task.status, "result": task.result}

    return jsonify(response)


@bp.route("/features")
def features_by_user():
    if not g.user:
        abort(400)

    user_id = g.user

    # Find all computed features for this user
    features_of_user = FeatureExtraction.find_by_user(user_id)

    feature_list = format_features(features_of_user)

    return jsonify(feature_list)


@bp.route("/features/types")
def feature_types():
    feature_families = FeatureFamily.find_all()

    feature_family_names = map(lambda family: family.name, feature_families)

    return jsonify(list(feature_family_names))


@bp.route("/features/<study_uid>")
def features_by_study(study_uid):
    if not g.user:
        abort(400)

    user_id = g.user

    # Find all computed features for this study
    features_of_study = FeatureExtraction.find_by_user_and_study_uid(user_id, study_uid)

    feature_list = format_features(features_of_study)

    return jsonify(feature_list)


@bp.route("/extract/<study_uid>/<feature_name>")
def extract(study_uid, feature_name):
    if not g.user:
        abort(400)

    user_id = g.user

    # Only support pyradiomics (for now)
    if feature_name != "pyradiomics":
        raise InvalidUsage("This feature is not supported yet!")

    # Get the associated study from DB
    study = get_or_create(Study, uid=study_uid)

    # Define features path for storing the results
    features_dir = os.path.join(FEATURES_BASE_DIR, user_id, study_uid)
    features_filename = feature_name + ".json"
    features_path = os.path.join(features_dir, features_filename)

    # Currently update any existing feature with the same path
    feature = FeatureExtraction.find_by_path(features_path)

    # If feature exists, set it to "in progress" again
    if not feature:
        feature = FeatureExtraction(feature_name, features_path, user_id, study.id)
        feature.save_to_db()

    # Start Celery
    result = my_celery.send_task(
        "imaginetasks.extract",
        countdown=0.1,
        args=[g.token, feature.id, study_uid, features_dir, features_path],
    )

    # follow_task(result, feature.id)
    eventlet.spawn(follow_task, current_app._get_current_object(), result, feature.id)

    # Assign the task to the feature
    feature.task_id = result.id
    db.session.commit()

    formatted_feature = format_feature(feature, celery_states.STARTED)

    # Spawn thread to follow the task's status
    # eventlet.spawn(follow_task, result, feature.id)

    # follow_task(result, feature.id)

    return jsonify(formatted_feature)


def format_features(features):
    # Gather the features
    feature_list = []

    if features:
        for feature in features:
            formatted_feature = format_feature(feature)
            feature_list.append(formatted_feature)

    return feature_list


def format_feature(feature, status=None):
    status_message = ""
    final_status = celery_states.SUCCESS

    if status:
        final_status = status
    else:
        # Get the feature status & update the status if necessary!
        if feature.task_id:
            status_object = fetch_task_result(feature.task_id)
            result = status_object.result

            # Get the status message for the task
            status_message = task_status_message(result)

            final_status = status_object.status

    # Read the features file (if available)
    sanitized_object = read_feature_file(feature.path)

    return {
        "id": feature.id,
        "name": feature.name,
        "updated_at": feature.updated_at.strftime(DATE_FORMAT),
        "status": final_status,
        "status_message": status_message,
        "payload": sanitized_object,
        "study_uid": feature.study.uid,
    }


class CustomResult(object):
    pass


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


def follow_task(app, result, feature_id):
    print(f"Feature {feature_id} - STARTING TO LISTEN FOR EVENTS!")

    try:
        result.get(on_message=task_status_update, propagate=True)
    except Exception as e:
        socketio_body = get_socketio_body(feature_id, celery_states.FAILURE, str(e))
        my_socketio.emit("feature-status", socketio_body)

    print(f"Feature {feature_id} - DONE!")

    # When the process ends, set the feature status to complete
    # if status == celery_states.SUCCESS:
    with app.app_context():
        feature = FeatureExtraction.find_by_id(feature_id)
        feature.save_to_db()

        socketio_body = get_socketio_body(
            feature_id,
            celery_states.SUCCESS,
            "Extraction complete",
            feature.updated_at.isoformat() + "Z",
            read_feature_file(feature.path),
        )

        my_socketio.emit("feature-status", socketio_body)

    return result


def task_status_update(body):

    status = body["status"]

    print("Got an update : " + str(body))

    # Don't send a message about pending or successful tasks (this is handled elsewhere)
    if (
        status == celery_states.PENDING
        or status == celery_states.SUCCESS
        or status == celery_states.FAILURE
    ):
        return

    feature_id = body["result"]["feature_id"]

    print(
        f"Feature {feature_id} - Status: {status}, Message: {body['result']['status_message']}"
    )

    socketio_body = get_socketio_body(
        feature_id, status, utils.task_status_message(body["result"])
    )

    # Send Socket.IO message to clients
    my_socketio.emit("feature-status", socketio_body)


def get_socketio_body(
    feature_id, status, status_message, updated_at=None, payload=None
):
    socketio_body = {
        "feature_id": feature_id,
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


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


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


def read_feature_file(feature_path):

    sanitized_object = {}
    if feature_path:
        try:
            feature_object = jsonpickle.decode(open(feature_path).read())
            sanitized_object = sanitize_features_object(feature_object)
        except FileNotFoundError:
            print(f"{feature_path} does not exist!")

    return sanitized_object

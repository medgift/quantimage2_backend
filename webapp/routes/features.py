import os, sys, json
import traceback
from collections import OrderedDict

import eventlet
import flask
import jsonpickle
import requests

from celery.result import AsyncResult
from flask import Blueprint, abort, jsonify, request, g
from kombu import uuid
from numpy.core.records import ndarray

from imaginebackend_common import utils
from imaginebackend_common.misc_enums import FeatureStatus
from imaginebackend_common.utils import task_status_message, InvalidUsage

from ..config import (
    FEATURES_BASE_DIR,
    keycloak_client,
    FEATURE_TYPES,
    KHEOPS_ENDPOINTS,
    token,
)
from ..models import db, Feature, get_or_create, Study

bp = Blueprint(__name__, "features")

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


@bp.before_request
def before_request():
    if request.method != "OPTIONS":
        if not validate_request(request):
            abort(401)
        else:
            g.user = kheops_userid_from_token(request.headers["Authorization"])
        pass

    pass


@bp.route("/")
def hello():
    return "Hello IMAGINE!"


@bp.route("/feature/<task_id>/status")
def feature_status(task_id):

    task = fetch_task_result(task_id)

    if task.status == "PENDING":
        abort(404)

    response = {"status": task.status, "result": task.result}

    return jsonify(response)


@bp.route("/features")
def features_by_user():
    if not g.user:
        abort(400)

    user_id = g.user

    # Find all computed features for this user
    features_of_user = Feature.find_by_user(user_id)

    feature_list = format_features(features_of_user)

    return jsonify(feature_list)


@bp.route("/features/types")
def feature_types():
    return jsonify(FEATURE_TYPES)


@bp.route("/features/<study_uid>")
def features_by_study(study_uid):
    if not g.user:
        abort(400)

    user_id = g.user

    # Find all computed features for this study
    features_of_study = Feature.find_by_user_and_study_uid(user_id, study_uid)

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
    feature = Feature.find_by_path(features_path)

    # If feature exists, set it to "in progress" again
    if feature:
        feature.status = FeatureStatus.IN_PROGRESS
    else:
        feature = Feature(feature_name, features_path, user_id, study.id)
        feature.save_to_db()

    # Generate UUID for the task
    task_id = uuid()

    # Result
    result = AsyncResult(task_id)

    # Spawn thread to follow the task's status
    eventlet.spawn(follow_task, result)

    # Start Celery
    from ..app import my_celery

    my_celery.send_task(
        "imaginetasks.extract",
        task_id=task_id,
        args=[feature.id, study_uid, features_dir, features_path],
        countdown=1,
    )

    # Assign the task to the feature
    feature.task_id = task_id
    db.session.commit()

    return jsonify(feature.to_dict())


def format_features(features):
    # Gather the features
    feature_list = []
    for feature in features:

        status_message = ""

        # Get the feature status & update the status if necessary!
        if feature.task_id:
            status_object = fetch_task_result(feature.task_id)
            result = status_object.result

            # Get the status message for the task
            status_message = task_status_message(result)

        # Read the features file (if available)
        sanitized_object = read_feature_file(feature.path)

        feature_list.append(
            {
                "id": feature.id,
                "name": feature.name,
                "updated_at": feature.updated_at.strftime(DATE_FORMAT),
                "status": feature.status,
                "status_message": status_message,
                "payload": sanitized_object,
                "study_uid": feature.study.uid,
            }
        )

    return feature_list


def fetch_task_result(task_id):
    task = AsyncResult(task_id)

    return task


def follow_task(result):
    print("STARTING TO LISTEN FOR EVENTS!")
    exc = result.get(on_message=task_status_update, propagate=False)
    return result


def task_status_update(body):

    status = body["status"]

    if status == "PENDING":
        return

    print("Got status update!")
    print(f"Status: {body['status']}, Message: {body['result']['status_message']}")

    feature_id = body["result"]["feature_id"]
    feature_status = (FeatureStatus.IN_PROGRESS, FeatureStatus.COMPLETE)[
        body["status"] == "SUCCESS"
    ]

    socketio_body = {
        "feature_id": feature_id,
        "status": int(feature_status),
        "status_message": utils.task_status_message(body["result"]),
    }

    # When the process ends, set the feature status to complete
    if feature_status == FeatureStatus.COMPLETE:

        from ..app import app

        with app.app_context():
            feature = Feature.find_by_id(feature_id)
            feature.status = feature_status
            db.session.commit()

            # Set the new updated date when complete
            socketio_body["updated_at"] = feature.updated_at.isoformat() + "Z"

            # Set the new feature payload when complete
            socketio_body["payload"] = read_feature_file(feature.path)

    # Send Socket.IO message to clients
    from ..app import my_socketio

    my_socketio.emit("feature-status", socketio_body)


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
        feature_object = jsonpickle.decode(open(feature_path).read())
        sanitized_object = sanitize_features_object(feature_object)

    return sanitized_object


def validate_request(request):
    token = request.headers["Authorization"]
    # rpt = keycloak_client.entitlement(token, "resource_id")
    validated = keycloak_client.introspect(token)
    return validated["active"]


def kheops_userid_from_token(token):
    secret = f"-----BEGIN PUBLIC KEY-----\n{os.environ['KEYCLOAK_REALM_PUBLIC_KEY']}\n-----END PUBLIC KEY-----"

    # Verify signature & expiration
    options = {"verify_signature": True, "verify_aud": False, "exp": True}
    token_decoded = keycloak_client.decode_token(token, key=secret, options=options)

    email = token_decoded["email"]

    user_ref_url = f"{KHEOPS_ENDPOINTS.users}{KHEOPS_ENDPOINTS.users_suffix}{email}"

    user = requests.get(user_ref_url, headers=get_token_header()).json()

    return user["sub"]


def get_token_header():
    return {"Authorization": "Bearer " + token}

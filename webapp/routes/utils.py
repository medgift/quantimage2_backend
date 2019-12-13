import os
import sys
from collections import OrderedDict
from pathlib import Path

import jsonpickle
import requests
import celery.states as celery_states

from flask import abort, g
from numpy.core.records import ndarray

from imaginebackend_common.utils import is_jsonable, ExtractionStatus
from .. import my_celery
from ..config import keycloak_client


class CustomResult(object):
    pass


# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"


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
        task.status = celery_states.PENDING
        task.result = None
        return task


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

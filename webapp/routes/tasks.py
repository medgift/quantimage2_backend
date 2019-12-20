from flask import Blueprint, abort, jsonify, request, g

from imaginebackend_common.models import FeatureExtractionTask
from imaginebackend_common.utils import fetch_task_result
from .utils import validate_decorate
import celery.states as celery_states

# Define blueprint
bp = Blueprint(__name__, "tasks")


@bp.before_request
def before_request():
    validate_decorate(request)


# All feature extraction tasks of a user
@bp.route("/tasks")
def tasks_by_user():
    tasks = FeatureExtractionTask.find_by_user(g.user)

    return jsonify(tasks)


# Feature extraction tasks for a given study
@bp.route("/tasks/<study_uid>")
def tasks_by_study(study_uid):
    tasks = FeatureExtractionTask.find_by_user_and_study(g.user, study_uid)

    return jsonify(tasks)


# Status of a feature extraction task
@bp.route("/tasks/<task_id>/status")
def feature_status(task_id):

    task = fetch_task_result(task_id)

    if task.status == celery_states.PENDING:
        abort(404)

    if task.status == celery_states.FAILURE:
        task.result

    response = {"status": task.status, "result": task.result}

    return jsonify(response)

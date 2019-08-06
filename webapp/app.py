import eventlet
from celery import Celery

eventlet.monkey_patch()

import celery

import os
import sys

from collections import OrderedDict

from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO
from numpy.core.records import ndarray
from config import FEATURES_BASE_DIR, FEATURE_TYPES

import logging
import pydevd_pycharm
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from imaginebackend_common.misc_enums import FeatureStatus
from imaginebackend_common.utils import task_status_message

# Handle specific numpy formats when pickling
jsonpickle_numpy.register_handlers()

db = SQLAlchemy()

# Constants
DATE_FORMAT = "%d.%m.%Y %H:%M"

from models import *

if "DEBUGGER_IP" in os.environ:
    try:
        pydevd_pycharm.settrace(
            os.environ["DEBUGGER_IP"],
            port=int(os.environ["DEBUGGER_PORT"]),
            suspend=False,
            stderrToServer=True,
            stdoutToServer=True,
        )
    except ConnectionRefusedError:
        logging.warning("No debug server running")


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "mysql://"
        + os.environ["MYSQL_USER"]
        + ":"
        + os.environ["MYSQL_PASSWORD"]
        + "@db/"
        + os.environ["MYSQL_DATABASE"]
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # app.config["SQLALCHEMY_ECHO"] = True
    app.config["SECRET-KEY"] = "cookies are delicious!"

    app.config["CELERY_BROKER_URL"] = os.environ["CELERY_BROKER_URL"]
    app.config["CELERY_RESULT_BACKEND"] = os.environ["CELERY_RESULT_BACKEND"]

    CORS(app)

    db.init_app(app)
    db.create_all(app=app)

    @app.route("/")
    def index():
        return "This is the IMAGINE Python Web BACKEND"

    @app.route("/feature/<task_id>/status")
    def feature_status(task_id):

        task = celery.AsyncResult(task_id)

        if task.status == "PENDING":
            abort(404)

        response = {"status": task.status, "result": task.result}

        return jsonify(response)

    @app.route("/features/types")
    def feature_types():
        return jsonify(FEATURE_TYPES)

    @app.route("/features/<study_uid>")
    def features(study_uid):

        # Get user from headers (for now)
        user_id = request.headers["X-User-ID"]

        # Find all computed features for this study
        features = Feature.find_by_user_and_study_uid(user_id, study_uid)

        # Gather the features
        feature_list = []
        for feature in features:

            status_message = ""

            # Get the feature status & update the status if necessary!
            if feature.task_id:
                status_object = feature_status(feature.task_id).json
                result = status_object["result"]

                # Update the feature status in the DB if the celery task is COMPLETE
                if (
                    feature.status != FeatureStatus.COMPLETE
                    and status_object["status"] == "SUCCESS"
                ):
                    feature.status = FeatureStatus.COMPLETE
                    db.session.commit()

                # Get the status message for the task
                status_message = task_status_message(result)

            # Read the features file (if available)
            sanitized_object = {}
            if feature.path:
                feature_object = jsonpickle.decode(open(feature.path).read())
                sanitized_object = sanitize_features_object(feature_object)

            feature_list.append(
                {
                    "id": feature.id,
                    "name": feature.name,
                    "updated_at": feature.updated_at.strftime(DATE_FORMAT),
                    "status": feature.status,
                    "status_message": status_message,
                    "payload": sanitized_object,
                }
            )

        return jsonify(feature_list)

    @app.route("/extract/<study_uid>/<feature_name>")
    def extract(study_uid, feature_name):

        # Get user from headers (for now)
        user_id = request.headers["X-User-ID"]

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

        # Start Celery task
        task = celery.send_task(
            "imaginetasks.extract",
            args=[feature.id, study_uid, features_dir, features_path],
        )

        # Send Socket.IO message (for testing)
        socketio.emit(
            "feature-status",
            {
                "feature_id": feature.id,
                "status": FeatureStatus.IN_PROGRESS,
                "status_message": "Starting...",
            },
        )

        # Assign the task to the feature
        feature.task_id = task.id
        db.session.commit()

        return jsonify(feature.to_dict())

    @app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    return app


def sanitize_features_object(feature_object):
    sanitized_object = OrderedDict()

    for feature_name in feature_object:
        if is_jsonable(feature_object[feature_name]):
            print(feature_object[feature_name])
            sanitized_object[feature_name] = feature_object[feature_name]
        else:
            # Numpy NDArrays
            if type(feature_object[feature_name] is ndarray):
                sanitized_object[feature_name] = feature_object[feature_name].tolist()
            else:
                print(feature_name + " is unsupported", file=sys.stderr)

    return sanitized_object


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class InvalidUsage(Exception):
    status_code = 400

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


def setup_sockets(socketio):
    @socketio.on("connect")
    def connection():
        print("client " + request.sid + " connected!")

    @socketio.on("disconnect")
    def disconnection():
        print("client " + request.sid + " disconnected!")


def make_celery(app):
    pass


""" 
This function:
- creates a new Celery object 
- configures it with the broker from the application config 
- updates the rest of the Celery config from the Flask config
- creates a subclass of the task that wraps the task execution in an application context 
This is necessary to properly integrate Celery with Flask. 
"""


def make_celery(app):
    celery = Celery(
        "tasks",
        backend=app.config["CELERY_RESULT_BACKEND"],
        broker=app.config["CELERY_BROKER_URL"],
    )
    celery.conf.update(app.config)

    return celery


if __name__ == "__main__":
    app = create_app()
    socketio = SocketIO(
        app,
        cors_allowed_origins=[os.environ["CORS_ALLOWED_ORIGINS"]],
        async_mode="eventlet",
        logger=True,
        engineio_logger=True,
        message_queue=os.environ["SOCKET_MESSAGE_QUEUE"],
    )

    # the app is passed to make_celery function, this function sets up celery in order to integrate with the flask application
    celery = make_celery(app)

    setup_sockets(socketio)
    socketio.run(app, host="0.0.0.0")
    # app.run(host="0.0.0.0")

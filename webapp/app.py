# Important to monkey-patch in the beginning!
import eventlet

from imaginebackend_common.utils import InvalidUsage

eventlet.monkey_patch()

import os

from celery import Celery

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

import logging
import pydevd_pycharm

# Import rest of the app files
from .routes.features import bp as features_bp
from .models import *

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
    app.config["SQLALCHEMY_ECHO"] = False
    app.config["SECRET-KEY"] = "cookies are delicious!"

    app.config["CELERY_BROKER_URL"] = os.environ["CELERY_BROKER_URL"]
    app.config["CELERY_RESULT_BACKEND"] = os.environ["CELERY_RESULT_BACKEND"]

    @app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    return app


def setup_app(app):
    CORS(app)

    # Init SQLAlchemy & create all tables
    db.init_app(app)
    db.create_all(app=app)

    # Register routes
    app.register_blueprint(features_bp)


def make_socketio(app):
    socketio = SocketIO(
        app,
        cors_allowed_origins=[os.environ["CORS_ALLOWED_ORIGINS"]],
        async_mode="eventlet",
        async_handlers=True,
        logger=False,
        engineio_logger=False,
        # message_queue=os.environ["SOCKET_MESSAGE_QUEUE"],
    )

    setup_sockets(socketio)

    return socketio


def setup_sockets(socketio):
    @socketio.on("connect")
    def connection():
        print("client " + request.sid + " connected!")

    @socketio.on("disconnect")
    def disconnection():
        print("client " + request.sid + " disconnected!")


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


# Create the Flask app
app = create_app()

# Setup celery
my_celery = make_celery(app)

# Setup Socket.IO
my_socketio = make_socketio(app)

# Setup the app
setup_app(app)

# Run the app (through socket.io)
my_socketio.run(app, host="0.0.0.0")

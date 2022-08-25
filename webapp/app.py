# Important to monkey-patch in the beginning!
print("Monkey Patching!")
import eventlet

eventlet.monkey_patch()

# System packages
import os
import logging
import socket

# Debugging
import pydevd_pycharm

# Flask
from flask_cors import CORS

from celery import Celery
from flask import request
from flask_socketio import SocketIO

from flask import jsonify

from imaginebackend_common.flask_init import create_app
from imaginebackend_common.models import db
from imaginebackend_common.utils import InvalidUsage

# Routes
from routes.features import bp as features_bp
from routes.feature_presets import bp as feature_presets_bp
from routes.feature_collections import bp as feature_collections_bp
from routes.tasks import bp as tasks_bp
from routes.models import bp as models_bp
from routes.labels import bp as labels_bp
from routes.charts import bp as charts_bp
from routes.navigation_history import bp as navigation_bp
from routes.albums import bp as albums_bp

print("App is Starting!")


def setup_sockets(socketio):
    @socketio.on("connect")
    def connection():
        print("client " + request.sid + " connected!")

    @socketio.on("disconnect")
    def disconnection():
        print("client " + request.sid + " disconnected!")


def make_socketio():

    allowed_origins = (
        os.environ["CORS_ALLOWED_ORIGINS"]
        if "," not in os.environ["CORS_ALLOWED_ORIGINS"]
        else os.environ["CORS_ALLOWED_ORIGINS"].split(",")
    )

    socketio = SocketIO(
        cors_allowed_origins=allowed_origins,
        async_mode="eventlet",
        async_handlers=True,
        logger=True,
        engineio_logger=False,
    )

    setup_sockets(socketio)

    return socketio


def start_app():

    print("Creating the app!")

    # Setup Debugger
    if "DEBUGGER_IP" in os.environ and os.environ["DEBUGGER_IP"] != "":
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
        except socket.timeout:
            logging.warning("Could not connect to the debugger")

    # Create the Flask app
    flask_app = create_app()

    # Setup the Flask app (DB, celery, routes, etc.)
    setup_app(flask_app)

    # Define error handler
    @flask_app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    # CORS
    CORS(flask_app)

    print("Running the app through Socket.IO!")

    # Run the app (through socket.io)
    flask_app.my_socketio.run(flask_app, host="0.0.0.0")


def setup_app(app):

    db.create_all(app=app)

    # Celery
    # set broker url and result backend from app config
    # Create celery
    my_celery = Celery(__name__)
    my_celery.conf.update(app.config)

    # Setup Socket.IO
    my_socketio = make_socketio()

    # Socket.IO
    my_socketio.init_app(
        app,
        message_queue=os.environ["SOCKET_MESSAGE_QUEUE"],
    )

    app.my_celery = my_celery
    app.my_socketio = my_socketio

    with app.app_context():
        app.register_blueprint(features_bp)
        app.register_blueprint(feature_presets_bp)
        app.register_blueprint(feature_collections_bp)
        app.register_blueprint(tasks_bp)
        app.register_blueprint(models_bp)
        app.register_blueprint(labels_bp)
        app.register_blueprint(charts_bp)
        app.register_blueprint(navigation_bp)
        app.register_blueprint(albums_bp)


if __name__ == "__main__":
    start_app()

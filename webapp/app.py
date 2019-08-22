# Important to monkey-patch in the beginning!
import eventlet
from celery import Celery

eventlet.monkey_patch()

# System packages
import os, logging

# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

# Debugging
import pydevd_pycharm

# Common imports
from imaginebackend_common.utils import InvalidUsage

# Import rest of the app files
from .routes.features import bp as features_bp
from .models import db

# Setup Debugger
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

# App factory
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

    # Setup plugins etc.
    setup_app(app)

    return app


def setup_app(app):

    # CORS
    CORS.init_app(app)

    # Init SQLAlchemy & create all tables
    db.init_app(app)
    db.create_all(app=app)

    # Celery
    from .routes.features import my_celery
    # set broker url and result backend from app config
    my_celery.conf.broker_url = app.config['CELERY_BROKER_URL']
    my_celery.conf.result_backend = app.config['CELERY_RESULT_BACKEND']

    # SocketIO
    SocketIO.init_app(app)

    # Register routes
    app.register_blueprint(features_bp)


def make_socketio():

    socketio = SocketIO(
        app=None,
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

# Setup Socket.IO
my_socketio = make_socketio()

# Create the Flask app
app = create_app()

# Run the app (through socket.io)
my_socketio.run(app, host="0.0.0.0")

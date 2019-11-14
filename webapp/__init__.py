import os

from celery import Celery
from flask import request, Flask, jsonify
from flask_socketio import SocketIO

# App factory
from imaginebackend_common.utils import InvalidUsage
from .models import db


def make_socketio():

    allowed_origins = (
        os.environ["CORS_ALLOWED_ORIGINS"]
        if "," not in os.environ["CORS_ALLOWED_ORIGINS"]
        else os.environ["CORS_ALLOWED_ORIGINS"].split(",")
    )

    socketio = SocketIO(
        app=None,
        cors_allowed_origins=allowed_origins,
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


# Setup Socket.IO
my_socketio = make_socketio()

# Create celery
my_celery = Celery(__name__)


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

    app.config["UPLOAD_FOLDER"] = os.environ["UPLOAD_FOLDER"]

    @app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    # Setup plugins etc.
    setup_app(app)

    return app


def setup_app(app):

    # Init SQLAlchemy & create all tables
    db.init_app(app)
    db.create_all(app=app)

    # Celery
    # set broker url and result backend from app config
    my_celery.conf.update(app.config)

    # Socket.IO
    my_socketio.init_app(app)

    # Register routes
    from .routes.features import bp as features_bp
    from .routes.feature_families import bp as feature_families_bp

    with app.app_context():
        app.register_blueprint(features_bp)
        app.register_blueprint(feature_families_bp)

import os

from flask import Flask

from flask_compress import Compress
from get_docker_secret import get_docker_secret

from quantimage2_backend_common.models import db

# Initialize Flask Compress
compress = Compress()


def create_app():

    # create and configure the app
    debug = False

    if "DEBUG" in os.environ and os.environ["DEBUG"] == "true":
        debug = True

    app = Flask(__name__, instance_relative_config=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "mysql://"
        + os.environ["DB_USER"]
        + ":"
        + get_docker_secret("db-user-password")
        + "@db/"
        + os.environ["DB_DATABASE"]
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ECHO"] = False
    app.config["SECRET-KEY"] = "cookies are delicious!"
    app.config["DEBUG"] = debug

    app.config["CELERY_BROKER_URL"] = os.environ["CELERY_BROKER_URL"]
    app.config["CELERY_RESULT_BACKEND"] = os.environ["CELERY_RESULT_BACKEND"]

    app.config["UPLOAD_FOLDER"] = "/quantimage2-data/feature-presets"

    # app.config[
    #    "COMPRESS_REGISTER"
    # ] = False  # disable default compression of all eligible requests

    app.config["COMPRESS_MIMETYPES"] = [
        "text/html",
        "text/css",
        "text/xml",
        "application/json",
        "application/javascript",
        "multipart/form-data",
    ]

    # Disable key sorting
    app.config["JSON_SORT_KEYS"] = False

    # Create feature presets folder (if necessary)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Setup plugins etc.
    init_extensions(app)

    return app


def init_extensions(app):
    # Init SQLAlchemy & create all tables
    db.init_app(app)

    compress.init_app(app)

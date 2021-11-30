import os

from flask import Flask

from flask_compress import Compress

from imaginebackend_common.models import db

# Initialize Flask Compress
compress = Compress()


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

    app.config["UPLOAD_FOLDER"] = "/imagine-data/feature-presets"

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

    # Create feature presets folder (if necessary)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Setup plugins etc.
    init_extensions(app)

    return app


def init_extensions(app):
    # Init SQLAlchemy & create all tables
    db.init_app(app)

    compress.init_app(app)

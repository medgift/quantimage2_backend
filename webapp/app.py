from flask import jsonify

from imaginebackend_common.flask_init import create_app
from imaginebackend_common.models import db
from imaginebackend_common.utils import InvalidUsage

print("App is Starting!")

# System packages
import os
import logging

# Flask
from flask_cors import CORS

# Debugging
import pydevd_pycharm

# Import rest of the app files
from . import my_socketio, my_celery

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

# Create the Flask app
flask_app = create_app()


def setup_app(app):

    db.create_all(app=app)

    # Celery
    # set broker url and result backend from app config
    my_celery.conf.update(app.config)

    # Socket.IO
    my_socketio.init_app(
        app, message_queue=os.environ["SOCKET_MESSAGE_QUEUE"],
    )

    # Register routes
    from .routes.features import bp as features_bp
    from .routes.feature_families import bp as feature_families_bp
    from .routes.tasks import bp as tasks_bp

    with app.app_context():
        app.register_blueprint(features_bp)
        app.register_blueprint(feature_families_bp)
        app.register_blueprint(tasks_bp)


setup_app(flask_app)

# Define error handler
@flask_app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


# CORS
CORS(flask_app)

# Run the app (through socket.io)
my_socketio.run(flask_app, host="0.0.0.0")

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
from . import create_app, my_socketio

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

# Create the Flask app
app = create_app()

# CORS
CORS(app)

# Run the app (through socket.io)
my_socketio.run(app, host="0.0.0.0")

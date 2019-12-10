print("App is Starting!")

# System packages
import os
import logging

# Flask
from flask_cors import CORS

# Debugging
import pydevd_pycharm

# Import rest of the app files
from . import create_app, my_socketio

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

# CORS
CORS(flask_app)

# Run the app (through socket.io)
my_socketio.run(flask_app, host="0.0.0.0")

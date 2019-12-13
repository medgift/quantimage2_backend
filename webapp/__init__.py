# Important to monkey-patch in the beginning!

print("Monkey Patching!")
import eventlet

eventlet.monkey_patch()

import os

from celery import Celery
from flask import request
from flask_socketio import SocketIO


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

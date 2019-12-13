import json
from enum import Enum


# Exceptions
class CustomException(Exception):
    status_code = 500

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


class InvalidUsage(CustomException):
    def __init__(self, message):
        CustomException.__init__(self, message, 400)


class ComputationError(CustomException):
    def __init__(self, message):
        CustomException.__init__(self, message, 500)


# Socket.IO messages
class MessageType(Enum):
    FEATURE_TASK_STATUS = "feature-status"
    EXTRACTION_STATUS = "extraction-status"

# Status messages
class StatusMessage(Enum):
    WAITING_TO_START = "Waiting to start"


def get_socketio_body_feature_task(
    task_id,
    feature_extraction_task_id,
    status,
    status_message,
    updated_at=None,
    payload=None,
):
    socketio_body = {
        "task_id": task_id,
        "feature_extraction_task_id": feature_extraction_task_id,
        "status": status,
        "status_message": status_message,
    }

    if updated_at:
        # Set the new updated date when complete
        socketio_body["updated_at"] = updated_at

    if payload:
        # Set the new feature payload when complete
        socketio_body["payload"] = payload

    return socketio_body


def get_socketio_body_extraction(feature_extraction_id, status):
    socketio_body = {
        "feature_extraction_id": feature_extraction_id,
        "status": status,
    }

    return socketio_body


# Feature Tasks
def task_status_message(current_step, total_steps, status_message):
    return f"{current_step}/{total_steps} - {status_message}"


def task_status_message_from_result(task_result):
    return task_status_message(
        task_result["current"], task_result["total"], task_result["status_message"]
    )


# Feature Extraction Status
class ExtractionStatus:
    ready = False
    successful = False
    failed = False
    completed_count = 0

    def __init__(self, ready=False, successful=False, failed=False, completed_count=0):
        self.ready = ready
        self.successful = successful
        self.failed = failed
        self.completed_count = completed_count


# Misc
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

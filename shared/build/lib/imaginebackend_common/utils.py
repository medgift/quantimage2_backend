def task_status_message(task_result):

    # If task result has disappeared, return empty string (it's complete anyway)
    if task_result is None or "status_message" not in task_result:
        return ""

    return f"{task_result['current']}/{task_result['total']} - {task_result['status_message']}"


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

def task_status_message(task_result):

    # If task result has disappeared, return empty string (it's complete anyway)
    if task_result is None:
        return ""

    return f"{task_result['current']}/{task_result['total']} - {task_result['status_message']}"

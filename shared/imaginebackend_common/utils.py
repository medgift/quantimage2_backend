def task_status_message(task_result):
    return f"{task_result['current']}/{task_result['total']} - {task_result['status_message']}"

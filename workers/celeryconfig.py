import os

result_persistent = True
result_expires = None
worker_concurrency = os.environ["CELERY_WORKER_CONCURRENCY"]

# Enable task events for Flower monitoring
worker_send_task_events = True
task_send_sent_event = True

# Redis-py 5.x compatibility settings
# These settings help prevent connection issues and task state update failures
result_backend_transport_options = {
    'retry_on_timeout': True,
    'socket_keepalive': True,
    'socket_keepalive_options': {
        1: 1,  # TCP_KEEPIDLE
        2: 1,  # TCP_KEEPINTVL
        3: 5   # TCP_KEEPCNT
    },
    'health_check_interval': 30,
}

# Ensure proper task state tracking and acknowledgment
task_track_started = True
task_acks_late = True

del os

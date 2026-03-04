import os

result_persistent = True
result_expires = None
worker_concurrency = os.environ["CELERY_WORKER_CONCURRENCY"]

# Enable task events for Flower monitoring
worker_send_task_events = True
task_send_sent_event = True

# Redis keepalive options (shared between broker and result backend).
# Keys are Linux TCP socket option constants (from /usr/include/netinet/tcp.h):
#   TCP_KEEPIDLE  = 4  (seconds idle before sending the first keepalive probe)
#   TCP_KEEPINTVL = 5  (seconds between subsequent probes)
#   TCP_KEEPCNT   = 6  (number of failed probes before dropping the connection)
_redis_transport_options = {
    'retry_on_timeout': True,
    'socket_keepalive': True,
    'socket_keepalive_options': {
        4: 60,  # TCP_KEEPIDLE  — start probing after 60s idle
        5: 10,  # TCP_KEEPINTVL — probe every 10s
        6: 5,   # TCP_KEEPCNT   — drop after 5 failed probes
    },
    'health_check_interval': 30,
}

# Apply keepalive to the result backend (where task state is written)
result_backend_transport_options = _redis_transport_options

# Apply the same keepalive to the broker (where tasks are queued).
# Without this, the broker connection can silently drop under the same
# conditions that caused tasks to get stuck.
broker_transport_options = _redis_transport_options

# Ensure proper task state tracking
task_track_started = True

# Acknowledge the task message AFTER the task finishes (not before).
# This prevents a task from being lost if the worker crashes mid-execution.
task_acks_late = True

# Also acknowledge on hard time-limit kill (SIGKILL) or exception.
# Without this, a task that hits time_limit gets SIGKILLed, the message is
# never acknowledged, and Redis re-queues it → infinite retry loop.
task_acks_on_failure_or_timeout = True

del os

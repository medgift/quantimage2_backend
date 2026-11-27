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
#
# visibility_timeout: with acks_late, a message still unacknowledged after this
# many seconds is assumed lost and redelivered to another worker, while the
# first copy keeps running. The Redis default (3600 s) made every training
# longer than 1 h run again in parallel (#54). It must stay above:
#   - the hard time_limit of every acks_late task (run_extraction, 7200 s);
#   - the hard time_limit of train (TRAINING_TIME_LIMIT, default 22200 s):
#     train is acked when it starts, but a training reserved by a busy worker
#     (prefetch) stays unacked until a slot frees, which can take that long.
# Every process that consumes from the broker restores with its own value, so
# Flower must use the same one (flower/flower_app.py).
# Set on a copy so the result backend does not get it.
broker_transport_options = {
    **_redis_transport_options,
    "visibility_timeout": 43200,  # 12 h
}

# Reserve only one message per worker process beyond the ones running.
# Reserved messages stay unacknowledged until they start, so a larger prefetch
# buffer exposes queued tasks to the visibility timeout too.
worker_prefetch_multiplier = 1

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

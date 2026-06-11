# Celery Task Patterns

## Celery Task Patterns

### Task Definitions (workers/tasks.py)

```python
from celery import Celery

app = Celery("quantimage2tasks")
app.config_from_envvar("CELERY_CONFIG_MODULE")

@app.task(
    bind=True,
    name="quantimage2tasks.extract",
    queue="extraction",
    time_limit=7200,
    soft_time_limit=6600,
)
def run_extraction(self, study_uid: str, album_id: str, ...):
    ...

@app.task(
    bind=True,
    name="quantimage2tasks.train",
    queue="training",
    serializer="pickle",
)
def train_model(self, ...):
    ...
```

### Task Queues

| Queue | Worker | Purpose | Concurrency |
|---|---|---|---|
| `extraction` | `celery_extraction` | Feature extraction (CPU/IO bound) | 2 |
| `training` | `celery_training` | ML model training (CPU bound) | 4 |

### Orchestration Pattern (Celery Chord)

Feature extraction uses a **chord** — parallel tasks with a final callback:

```python
from celery import chord

# One task per study, then finalize
extraction_chord = chord(
    [
        quantimage2tasks.extract.s(study_uid, album_id, ...)
        | finalize_extraction_task.s(task_id)
        for study_uid, task_id in studies_and_tasks
    ],
    finalize_extraction.si(extraction_id),
)
extraction_chord.apply_async()
```

### Progress Reporting

Tasks report progress via **Socket.IO** (through Redis message queue):

```python
from flask_socketio import SocketIO

socketio = SocketIO(message_queue=os.environ["SOCKET_MESSAGE_QUEUE"])

# Emit to all clients:
socketio.emit("extraction-status", status_dict)
socketio.emit("training-status", progress_dict)
```

### Socket.IO Event Types

| Event | Payload | Purpose |
|---|---|---|
| `extraction-status` | `{feature_extraction_id, ready, status, ...}` | Overall extraction progress |
| `feature-status` | `{task_id, status, ...}` | Per-study task status |
| `training-status` | `{training_id, phase, current, total, ...}` | Model training progress |

### Celery Configuration (celeryconfig.py)

```python
result_persistent = True
result_expires = None
task_track_started = True
task_acks_late = True
task_acks_on_failure_or_timeout = True
# Redis keepalive (detect broken connections):
broker_transport_options = {
    "socket_keepalive": True,
    "socket_keepalive_options": {TCP_KEEPIDLE: 60, TCP_KEEPINTVL: 10, TCP_KEEPCNT: 5},
    "health_check_interval": 30,
}
```

### Rules for New Tasks

- Always set `bind=True` to access `self` for state updates.
- Always specify `queue=` explicitly — never rely on the default queue.
- Set `time_limit` and `soft_time_limit` for extraction tasks.
- Use `serializer="pickle"` only for training tasks (scikit-learn objects).
- Report progress via Socket.IO, not just Celery state updates.
- Handle `SoftTimeLimitExceeded` gracefully (clean up temp files, update DB status).

---


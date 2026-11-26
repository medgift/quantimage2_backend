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
worker_prefetch_multiplier = 1
# Redis keepalive (detect broken connections), shared by broker and result
# backend; the broker gets its own copy with a visibility timeout:
result_backend_transport_options = _redis_transport_options
broker_transport_options = {**_redis_transport_options, "visibility_timeout": 43200}
```

**Acks, visibility timeout and prefetch.** With `acks_late`, a message stays
unacknowledged in Redis until the task finishes. If it is still unacknowledged
after `visibility_timeout` seconds, kombu assumes it was lost and redelivers it,
while the first copy keeps running (#54). Hence:

- The broker `visibility_timeout` (43200 s) must exceed the hard `time_limit`
  of every `acks_late` task (`run_extraction` 7200 s, the finalize callbacks
  600 s) and of `train` (see below). Don't put it on the result-backend options.
- Every process that consumes from the broker restores unacked messages with
  **its own** visibility timeout. Flower does (event receiver + `inspect` calls
  from its UI/API), so `flower/flower_app.py` carries the same value; keep the
  two in sync (`tests/test_flower_visibility_timeout.py` checks it).
- `train` sets `acks_late=False`: it is acknowledged when it starts, so a long
  training can never be redelivered. A training lost in a worker crash is not
  re-run; the user resubmits. Its limits come from `TRAINING_SOFT_TIME_LIMIT`
  (default 21600) and `TRAINING_TIME_LIMIT` (default soft + 600); the soft
  limit is reported as a failed training on `training-status`. A training
  reserved by a busy worker (prefetch) stays unacked until a slot frees, i.e.
  up to the `train` hard limit, hence the 12 h visibility timeout. Grid search
  runs through `utils.fit_grid_search`, which re-raises a soft limit that
  `GridSearchCV` (`error_score=nan`) would otherwise record as a failed fit
  when the fits run in-process (`n_jobs=1`).
- `worker_prefetch_multiplier = 1`: reserved-but-not-started messages are
  unacknowledged too, so a large prefetch buffer exposes them as well.

**Training CPU budget.** `GRID_SEARCH_CONCURRENCY` reaches `train_model` as
`n_jobs` and goes through `utils.resolve_n_jobs`: `-1` (or `None`/`<= 0`) means
the worker's available cores divided by `CELERY_WORKER_CONCURRENCY` (min 1;
a concurrency of 0 counts as one slot per core, as in Celery); a
positive value is an explicit override. The resolved value is used by
`GridSearchCV` and `permutation_importance`. The progress scorer
`utils.fake_score` is module-level (picklable into loky workers) and reuses one
cached Socket.IO client per process.

### Rules for New Tasks

- Always set `bind=True` to access `self` for state updates.
- Always specify `queue=` explicitly — never rely on the default queue.
- Set `time_limit` and `soft_time_limit` for long tasks; an `acks_late` task's
  hard `time_limit` must stay below the broker `visibility_timeout`.
- Use `serializer="pickle"` only for training tasks (scikit-learn objects).
- Report progress via Socket.IO, not just Celery state updates.
- Handle `SoftTimeLimitExceeded` gracefully (clean up temp files, update DB status).

---


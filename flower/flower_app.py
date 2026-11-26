"""Celery app Flower runs with (``celery -A flower_app flower``).

Flower is a broker consumer too: its event receiver and ``inspect`` calls make
kombu restore every unacknowledged message older than *Flower's own*
``visibility_timeout`` (Redis default 3600 s), whatever the workers use. Left
at the default, it re-queued long extractions and queued trainings after 1 h,
so the same task ran twice (#54). Keep this value equal to
``broker_transport_options["visibility_timeout"]`` in ``workers/celeryconfig.py``.
"""

import os

from celery import Celery

app = Celery(
    "flower",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)
app.conf.broker_transport_options = {"visibility_timeout": 43200}  # 12 h

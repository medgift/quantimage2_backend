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

# Set in env_files/common.env, like for the workers
app = Celery(
    "flower",
    broker=os.environ["CELERY_BROKER_URL"],
    backend=os.environ["CELERY_RESULT_BACKEND"],
)
app.conf.broker_transport_options = {"visibility_timeout": 43200}  # 12 h

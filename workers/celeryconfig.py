import os

result_persistent = True
result_expires = None
worker_concurrency = os.environ["CELERY_WORKER_CONCURRENCY"]

del os

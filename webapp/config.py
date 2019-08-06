import os

FEATURES_BASE_DIR = "/tmp/imagine/features"

FEATURE_TYPES = ["pyradiomics", "okapy"]


class BackendEndpoints(object):
    pass


ENDPOINTS = BackendEndpoints()
ENDPOINTS.feature_status = "/feature"
ENDPOINTS.feature_status_suffix = "/status"

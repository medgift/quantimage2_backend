import os
from keycloak import KeycloakOpenID

# Keycloak
keycloak_client = KeycloakOpenID(
    server_url=os.environ["KEYCLOAK_BASE_URL"],
    client_id=os.environ["KEYCLOAK_IMAGINE_CLIENT_ID"],
    client_secret_key=os.environ["KEYCLOAK_IMAGINE_CLIENT_SECRET"],
    realm_name=os.environ["KEYCLOAK_REALM_NAME"],
)

# Features
FEATURES_BASE_DIR = "/tmp/imagine/features"
FEATURE_TYPES = ["pyradiomics", "okapy"]

# Backend
class BackendEndpoints(object):
    pass


ENDPOINTS = BackendEndpoints()
ENDPOINTS.feature_status = "/feature"
ENDPOINTS.feature_status_suffix = "/status"

# Kheops
token = os.environ["KHEOPS_TOKEN"]

kheopsBaseURL = os.environ["KHEOPS_BASE_URL"]
kheopsBaseEndpoint = kheopsBaseURL + "/api"


class KheopsEndpoints(object):
    pass


KHEOPS_ENDPOINTS = KheopsEndpoints()
KHEOPS_ENDPOINTS.users = kheopsBaseEndpoint + "/users"
KHEOPS_ENDPOINTS.users_suffix = "?reference="

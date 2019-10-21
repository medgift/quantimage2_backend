import os
from keycloak import KeycloakOpenID

# Keycloak

# Backend client
keycloak_client = KeycloakOpenID(
    server_url=os.environ["KEYCLOAK_BASE_URL"] + "/auth/",
    client_id=os.environ["KEYCLOAK_IMAGINE_CLIENT_ID"],
    client_secret_key=os.environ["KEYCLOAK_IMAGINE_CLIENT_SECRET"],
    realm_name=os.environ["KEYCLOAK_REALM_NAME"],
)

# Extractions
EXTRACTIONS_BASE_DIR = "/tmp/imagine/extractions"
FEATURES_SUBDIR = "features"
CONFIGS_SUBDIR = "configs"

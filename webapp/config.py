import os

from keycloak.realm import KeycloakRealm

# Keycloak Realm
realm = KeycloakRealm(
    server_url=os.environ["KEYCLOAK_BASE_URL"],
    realm_name=os.environ["KEYCLOAK_REALM_NAME"],
)

# Backend client
oidc_client = realm.open_id_connect(
    client_id=os.environ["KEYCLOAK_IMAGINE_CLIENT_ID"],
    client_secret=os.environ["KEYCLOAK_IMAGINE_CLIENT_SECRET"],
)

# Extractions
EXTRACTIONS_BASE_DIR = "/imagine-data/extractions"
FEATURES_SUBDIR = "features"
CONFIGS_SUBDIR = "configs"

# Models
MODELS_BASE_DIR = "/imagine-data/models"

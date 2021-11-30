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
CONFIGS_SUBDIR = "configs"

# Feature Cache
FEATURES_CACHE_BASE_DIR = "/imagine-data/features-cache"

# Models
MODELS_BASE_DIR = "/imagine-data/models"

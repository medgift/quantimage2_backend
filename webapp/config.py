import os

from keycloak import KeycloakOpenID

# Keycloak Client (for decoding tokens)
oidc_client = KeycloakOpenID(
    server_url=f"{os.environ['KEYCLOAK_BASE_URL']}",
    realm_name=os.environ["KEYCLOAK_REALM_NAME"],
    client_id=os.environ["KEYCLOAK_IMAGINE_FRONTEND_CLIENT_ID"],
)

# Extractions
EXTRACTIONS_BASE_DIR = "/imagine-data/extractions"
CONFIGS_SUBDIR = "configs"

# Feature Cache
FEATURES_CACHE_BASE_DIR = "/imagine-data/features-cache"

import os
from functools import wraps

from flask import abort, g
from config import keycloak_client

KEYCLOAK_RESOURCE_ACCESS = "resource_access"
KEYCLOAK_ROLES = "roles"


def role_required(role_name):
    def decorator(func):
        @wraps(func)
        def authorize(*args, **kwargs):
            token_decoded = decode_token(g.token)

            print(token_decoded)

            if (
                not os.environ["KEYCLOAK_IMAGINE_FRONTEND_CLIENT_ID"]
                in token_decoded[KEYCLOAK_RESOURCE_ACCESS]
                or not os.environ["KEYCLOAK_FRONTEND_ADMIN_ROLE"]
                in token_decoded[KEYCLOAK_RESOURCE_ACCESS][
                    os.environ["KEYCLOAK_IMAGINE_FRONTEND_CLIENT_ID"]
                ][KEYCLOAK_ROLES]
            ):
                abort(401)  # not authorized

            return func(*args, **kwargs)

        return authorize

    return decorator


def validate_decorate(request):
    if request.method != "OPTIONS":
        if not validate_request(request):
            abort(401)
        else:
            g.user = userid_from_token(request.headers["Authorization"].split(" ")[1])
            g.token = request.headers["Authorization"].split(" ")[1]
        pass

    pass


def validate_request(request):
    authorization = request.headers["Authorization"]

    if not authorization.startswith("Bearer"):
        abort(400)
    else:
        token = authorization.split(" ")[1]
        # rpt = keycloak_client.entitlement(token, "resource_id")
        validated = keycloak_client.introspect(token)
        return validated["active"]


def decode_token(token):
    secret = f"-----BEGIN PUBLIC KEY-----\n{os.environ['KEYCLOAK_REALM_PUBLIC_KEY']}\n-----END PUBLIC KEY-----"

    # Verify signature & expiration
    options = {"verify_signature": True, "verify_aud": False, "exp": True}
    token_decoded = keycloak_client.decode_token(token, key=secret, options=options)

    return token_decoded


def userid_from_token(token):

    token_decoded = decode_token(token)

    id = token_decoded["sub"]

    return id

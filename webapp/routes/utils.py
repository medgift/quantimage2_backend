import os
from functools import wraps

import requests
from flask import abort, g
from jose import JWTError, ExpiredSignatureError
from jose.exceptions import JWTClaimsError

from config import oidc_client

KEYCLOAK_RESOURCE_ACCESS = "resource_access"
KEYCLOAK_ROLES = "roles"

KEYCLOAK_REALM_PUBLIC_KEY = None


def role_required(role_name):
    def decorator(func):
        @wraps(func)
        def authorize(*args, **kwargs):
            token_decoded = decode_token(g.token)

            if (
                not os.environ["KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID"]
                in token_decoded[KEYCLOAK_RESOURCE_ACCESS]
            ) or (
                not role_name
                in token_decoded[KEYCLOAK_RESOURCE_ACCESS][
                    os.environ["KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID"]
                ][KEYCLOAK_ROLES]
            ):
                abort(401)  # not authorized

            return func(*args, **kwargs)

        return authorize

    return decorator


def decorate_if_possible(request):

    try:
        authorization = request.headers["Authorization"]

        if authorization.startswith("Bearer"):
            g.user = userid_from_token(request.headers["Authorization"].split(" ")[1])
            g.token = request.headers["Authorization"].split(" ")[1]
    except KeyError:
        return True

    return True


def validate_decorate(request):
    if request.method != "OPTIONS":
        is_request_valid, error_message = validate_request(request)
        if not is_request_valid:
            abort(401, error_message)
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
        try:
            token_decoded = decode_token(token)
            return True, ""
        except (JWTError, ExpiredSignatureError, JWTClaimsError) as e:
            return False, str(e)


def decode_token(token):
    global KEYCLOAK_REALM_PUBLIC_KEY

    if KEYCLOAK_REALM_PUBLIC_KEY is None:
        KEYCLOAK_REALM_PUBLIC_KEY = f"-----BEGIN PUBLIC KEY-----\n{oidc_client.public_key()}\n-----END PUBLIC KEY-----"

    # Verify signature & expiration
    options = {"verify_signature": True, "verify_exp": True, "verify_aud": False}
    token_decoded = oidc_client.decode_token(
        token, key=KEYCLOAK_REALM_PUBLIC_KEY, options=options
    )

    return token_decoded


def userid_from_token(token):

    token_decoded = decode_token(token)

    id = token_decoded["sub"]

    return id

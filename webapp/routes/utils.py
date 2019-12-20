import os
import sys
from collections import OrderedDict
from pathlib import Path

import jsonpickle
import requests
import celery.states as celery_states

from flask import abort, g
from numpy.core.records import ndarray

from imaginebackend_common.utils import is_jsonable
from ..config import keycloak_client


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


def userid_from_token(token):
    secret = f"-----BEGIN PUBLIC KEY-----\n{os.environ['KEYCLOAK_REALM_PUBLIC_KEY']}\n-----END PUBLIC KEY-----"

    # Verify signature & expiration
    options = {"verify_signature": True, "verify_aud": False, "exp": True}
    token_decoded = keycloak_client.decode_token(token, key=secret, options=options)

    id = token_decoded["sub"]

    return id

#!/bin/bash
KEYCLOAK_USER=$KEYCLOAK_USER
KEYCLOAK_PASSWORD=$KEYCLOAK_PASSWORD
CLIENT=loginConnect
CLIENT_AUTH=kheopsAuthorization
WEBORIGINS=*
REDIRECTURIS=*
REALM=$KEYCLOAK_REALM_NAME

# Connect to keycloak
/opt/jboss/keycloak/bin/kcadm.sh config credentials --server http://localhost:8080/auth --realm master --user ${KEYCLOAK_USER} --password ${KEYCLOAK_PASSWORD}

# Create REALM
/opt/jboss/keycloak/bin/kcadm.sh create realms -s realm=${REALM} \
  -s enabled=true \
  -s sslRequired=external \
  -s registrationAllowed=true \
  -s registrationEmailAsUsername=true

# Create client loginConnect
/opt/jboss/keycloak/bin/kcadm.sh create clients -r ${REALM} \
  -s clientId=${CLIENT} \
  -s enabled=true \
  -s publicClient=true \
  -s 'webOrigins=["'${WEBORIGINS}'"]' \
  -s 'redirectUris=["'${REDIRECTURIS}'"]' \
  -s standardFlowEnabled=true \
  -s implicitFlowEnabled=true \
  -s directAccessGrantsEnabled=false \
  -s fullScopeAllowed=false

# Create client kheopsAuthorization
/opt/jboss/keycloak/bin/kcadm.sh create clients -r ${REALM} \
  -s clientId=${CLIENT_AUTH} \
  -s enabled=true \
  -s standardFlowEnabled=false \
  -s implicitFlowEnabled=false \
  -s directAccessGrantsEnabled=false \
  -s fullScopeAllowed=false \
  -s serviceAccountsEnabled=true \
  -s publicClient=false

# Add roles in Service Account User in client kheopsAuthorization
lowerCase="${CLIENT_AUTH,,}"

/opt/jboss/keycloak/bin/kcadm.sh add-roles -r ${REALM} \
  --uusername service-account-${lowerCase} \
  --cclientid realm-management \
  --rolename view-users

# Remove roles in Service Account User in client kheopsAuthorization
/opt/jboss/keycloak/bin/kcadm.sh  remove-roles -r ${REALM} \
  --uusername service-account-${lowerCase} \
  --rolename offline_access \
  --rolename uma_authorization

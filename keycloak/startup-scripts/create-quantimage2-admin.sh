#!/bin/bash

function configure_keycloak() {
  # Path to the keycloak admin CLI
  export PATH=/opt/jboss/keycloak/bin:$PATH

  # Get the keycloak admin password from the secret
  KC_MASTER_REALM=master
  KC_ADMIN_USER=$KEYCLOAK_USER
  KC_ADMIN_PASSWORD=$(cat /run/secrets/keycloak-admin-password)

  # Create Admin user for QuantImage v2
  QI2_ADMIN_USER=$QUANTIMAGE2_ADMIN_USER
  QI2_ADMIN_PASS=$(cat /run/secrets/quantimage2-admin-password)

  while :; do
    if curl http://localhost:8080/auth; then
      # Configure credentials
      kcadm.sh config credentials --server http://localhost:8080/auth --realm $KC_MASTER_REALM --user $KC_ADMIN_USER --password $KC_ADMIN_PASSWORD

      # Create and configure user
      echo "Creating Admin User for QuantImage v2"
      QI2_ADMIN_ID=$(kcadm.sh create users -r $KEYCLOAK_REALM_NAME -s enabled=true -s username=$QI2_ADMIN_USER -s email=$QI2_ADMIN_USER -s firstName=Admin -s lastName=QuantImage-v2)

      # Set Password of user
      kcadm.sh set-password -r $KEYCLOAK_REALM_NAME --username $QI2_ADMIN_USER --new-password $QI2_ADMIN_PASS

      # Add "admin" role for the IMAGINE client
      kcadm.sh add-roles -r $KEYCLOAK_REALM_NAME --uusername $QI2_ADMIN_USER --cclientid $KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID --rolename $KEYCLOAK_FRONTEND_ADMIN_ROLE

      break
    else
      echo "Server not ready yet, waiting for 5 seconds..."
      sleep 5
    fi
  done
}

#configure_keycloak & disown; # &>/dev/null
echo "Launching background process to configure Keycloak"
configure_keycloak &
echo "Finished background process to configure Keycloak"

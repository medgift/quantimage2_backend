#!/bin/bash

# If something goes wrong, this script does not run forever, but times out
TIMEOUT_SECONDS=300

# Define log file & export file paths
LOGFILE=/tmp/export.log
EXPORT_DIR=/tmp/export/quantimage2-realm

# Remove previous log & backup
rm -rf ${LOGFILE} ${EXPORT_DIR}

# Remove "keycloak-add-user.json" that might be left over after a mid-startup restart
ADD_USER_FILE=/opt/jboss/keycloak/standalone/configuration/keycloak-add-user.json
rm ${ADD_USER_FILE}

# Check if/how users should be exported
EXPORT_USERS="REALM_FILE"
if [ ! -z "$1" ]; then
  EXPORT_USERS=$1
fi

echo "Exporting realm with user export strategy '${EXPORT_USERS}'"

# Start a new keycloak instance with exporting options enabled.
# Use the port offset argument to prevent port conflicts
# with the "real" keycloak instance.
timeout ${TIMEOUT_SECONDS}s \
  /opt/jboss/tools/docker-entrypoint.sh \
  -Djboss.socket.binding.port-offset=500 \
  -Dkeycloak.migration.action=export \
  -Dkeycloak.migration.provider=dir \
  -Dkeycloak.migration.dir=${EXPORT_DIR} \
  -Dkeycloak.migration.realmName=QuantImage-v2 \
  -Dkeycloak.migration.usersExportStrategy=${EXPORT_USERS} \
  -Dkeycloak.profile=preview &>${LOGFILE} &

# Grab the keycloak export instance process id
PID="${!}"

# Wait for the export to finish
# It will wait till it sees the string, which indicates
# a successful finished backup.
# If it will take too long (>TIMEOUT_SECONDS), it will be stopped.
timeout ${TIMEOUT_SECONDS}s \
  grep -m 1 "Export finished successfully" <(tail -f ${LOGFILE})

# Stop the keycloak export instance
echo "Export done, going to kill PID ${PID}"
kill ${PID}
exit $?

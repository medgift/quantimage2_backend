#!/usr/bin/env bash
# ============================================================================
# QuantImage v2 — Live Performance Monitoring Suite
# Configuration file — shared by all monitoring modules
# ============================================================================

# ── Sampling interval (seconds) ──
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

# ── Output directories ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
REPORT_DIR="${SCRIPT_DIR}/reports"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# ── Session ID (timestamp-based for grouping logs) ──
SESSION_ID="${SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
export SESSION_ID

# ── QuantImage v2 Docker container names ──
QI2_BACKEND="quantimage2-backend-1"
QI2_CELERY_EXTRACTION="quantimage2-celery_extraction-1"
QI2_CELERY_TRAINING="quantimage2-celery_training-1"
QI2_FLOWER="quantimage2-flower-1"
QI2_DB="quantimage2-db-1"
QI2_REDIS="quantimage2-redis-1"
QI2_REDIS_SOCKET="quantimage2-redis-socket-1"
QI2_PHPMYADMIN="quantimage2-phpmyadmin-1"
QI2_OHIF="quantimage2-ohif-1"

# All QI2 containers for resource monitoring
QI2_CONTAINERS=(
    "$QI2_BACKEND"
    "$QI2_CELERY_EXTRACTION"
    "$QI2_CELERY_TRAINING"
    "$QI2_FLOWER"
    "$QI2_DB"
    "$QI2_REDIS"
    "$QI2_REDIS_SOCKET"
    "$QI2_OHIF"
)

# ── Kheops Docker container names ──
KHEOPS_REVERSE_PROXY="kheopsreverseproxy"
KHEOPS_AUTHORIZATION="kheopsauthorization"
KHEOPS_DICOMWEB_PROXY="kheopsdicomwebproxy"
KHEOPS_ZIPPER="kheopszipper"
KHEOPS_POSTGRES="kheopspostgres"
KHEOPS_UI="kheopsui"
PACS_ARC="pacsarc"
PACS_POSTGRES="pacspostgres"
PACS_AUTH_PROXY="pacsauthorizationproxy"
LDAP="ldap"

KHEOPS_CONTAINERS=(
    "$KHEOPS_REVERSE_PROXY"
    "$KHEOPS_AUTHORIZATION"
    "$KHEOPS_DICOMWEB_PROXY"
    "$KHEOPS_ZIPPER"
    "$KHEOPS_POSTGRES"
    "$KHEOPS_UI"
    "$PACS_ARC"
    "$PACS_POSTGRES"
    "$PACS_AUTH_PROXY"
    "$LDAP"
)

# Combined list for resource monitoring
ALL_CONTAINERS=("${QI2_CONTAINERS[@]}" "${KHEOPS_CONTAINERS[@]}")

# ── MySQL connection ──
MYSQL_HOST="127.0.0.1"
MYSQL_PORT="3307"
MYSQL_USER="quantimage2"
MYSQL_PASSWORD="1m4g1n3__backend--YOUs3r"
MYSQL_ROOT_PASSWORD="1m4g1n3__backend--r000T"
MYSQL_DATABASE="quantimage2"

# ── Redis connection ──
REDIS_CELERY_CONTAINER="$QI2_REDIS"
REDIS_SOCKET_CONTAINER="$QI2_REDIS_SOCKET"

# ── Kheops URLs ──
KHEOPS_URL="https://kheops.ehealth.hevs.ch"
QI2_BACKEND_URL="https://quantimage2.ehealth.hevs.ch"

# ── Traefik container ──
TRAEFIK_CONTAINER="ehealth-traefik-config-traefik-1"

# ── Self-monitoring PID file ──
MONITOR_PID_FILE="${LOG_DIR}/monitor_pids_${SESSION_ID}.txt"

# ── Color codes for terminal output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helper functions ──
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_header() {
    local module="$1"
    echo -e "${BOLD}${CYAN}[$(timestamp)] [$module]${NC} $2"
}

log_ok() {
    local module="$1"
    echo -e "${GREEN}[$(timestamp)] [$module]${NC} $2"
}

log_warn() {
    local module="$1"
    echo -e "${YELLOW}[$(timestamp)] [$module]${NC} ⚠ $2"
}

log_err() {
    local module="$1"
    echo -e "${RED}[$(timestamp)] [$module]${NC} ✗ $2"
}

# Check if a container is running
container_running() {
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true
}

# Get CSV-compatible timestamp
csv_ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

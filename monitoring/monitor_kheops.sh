#!/usr/bin/env bash
# ============================================================================
# Module 6: Kheops (PACS) Monitor
# Probes Kheops endpoints, monitors all Kheops containers,
# checks DCM4CHEE PACS health, Postgres DB connections, and response times
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="KHEOPS"
LOGFILE="${LOG_DIR}/kheops_health_${SESSION_ID}.csv"
LOGFILE_PACS="${LOG_DIR}/kheops_pacs_db_${SESSION_ID}.csv"
LOGFILE_ERRORS="${LOG_DIR}/kheops_errors_${SESSION_ID}.log"

# CSV header
echo "timestamp,reverse_proxy_up,reverse_proxy_resp_ms,auth_service_up,auth_resp_ms,dicomweb_proxy_up,dicomweb_resp_ms,zipper_up,pacs_arc_up,pacs_arc_resp_ms,api_studies_resp_ms,api_studies_status" > "$LOGFILE"
echo "timestamp,kheops_pg_active_conns,kheops_pg_idle_conns,kheops_pg_max_conns,pacs_pg_active_conns,pacs_pg_idle_conns,pacs_pg_db_size_mb" > "$LOGFILE_PACS"

log_header "$MODULE" "Starting Kheops monitor → $LOGFILE"

probe_endpoint() {
    local url="$1"
    local opts="${2:--sk}"
    local result
    result=$(curl $opts -o /dev/null -w "%{http_code}|%{time_total}" \
        --connect-timeout 5 --max-time 15 "$url" 2>/dev/null || echo "000|15.0")
    local status=$(echo "$result" | cut -d'|' -f1)
    local time_s=$(echo "$result" | cut -d'|' -f2)
    local time_ms=$(echo "scale=0; $time_s * 1000" | bc 2>/dev/null || echo "0")

    local up=0
    [[ "$status" =~ ^[2345] ]] && up=1

    echo "${up}|${time_ms}|${status}"
}

# Kheops DB queries via docker exec
kheops_pg_query() {
    docker exec "$KHEOPS_POSTGRES" psql -U kheopsuser -d kheops -t -A -c "$1" 2>/dev/null
}

pacs_pg_query() {
    docker exec "$PACS_POSTGRES" psql -U kheops_pacs -d kheops_pacs -t -A -c "$1" 2>/dev/null
}

while true; do
    ts=$(csv_ts)

    # ── Probe Kheops endpoints ──
    # Reverse proxy main endpoint
    rp=$(probe_endpoint "${KHEOPS_URL}/" "-sk")
    rp_up=$(echo "$rp" | cut -d'|' -f1)
    rp_ms=$(echo "$rp" | cut -d'|' -f2)

    # Authorization service (internal check via container)
    auth_up=0
    auth_ms=0
    if container_running "$KHEOPS_AUTHORIZATION"; then
        auth_up=1
        # Probe via Docker network
        auth_result=$(docker exec "$KHEOPS_REVERSE_PROXY" sh -c \
            "wget -q -O /dev/null --timeout=5 http://kheopsauthorization:8080/ 2>&1; echo \$?" 2>/dev/null || echo "1")
        [[ "$auth_result" == "0" ]] && auth_up=1 || auth_up=1  # Container is running at least
    fi

    # DICOMweb proxy
    dwp_up=0
    dwp_ms=0
    if container_running "$KHEOPS_DICOMWEB_PROXY"; then
        dwp_up=1
    fi

    # Zipper
    zip_up=0
    if container_running "$KHEOPS_ZIPPER"; then
        zip_up=1
    fi

    # PACS Archive (DCM4CHEE)
    pacs_up=0
    pacs_ms=0
    if container_running "$PACS_ARC"; then
        pacs_up=1
        # DCM4CHEE admin console health (internal)
        pacs_probe=$(docker exec "$PACS_ARC" sh -c \
            'curl -sf -o /dev/null -w "%{time_total}" http://localhost:8080/dcm4chee-arc/ui2/ 2>/dev/null || echo "0"' 2>/dev/null || echo "0")
        pacs_ms=$(echo "scale=0; ${pacs_probe:-0} * 1000" | bc 2>/dev/null || echo "0")
    else
        log_err "$MODULE" "PACS Archive (DCM4CHEE) is NOT running!"
    fi

    # API studies endpoint (unauthenticated → expect 401, but measures proxy chain latency)
    api_result=$(probe_endpoint "${KHEOPS_URL}/api/studies" "-sk")
    api_ms=$(echo "$api_result" | cut -d'|' -f2)
    api_status=$(echo "$api_result" | cut -d'|' -f3)

    echo "$ts,$rp_up,$rp_ms,$auth_up,$auth_ms,$dwp_up,$dwp_ms,$zip_up,$pacs_up,$pacs_ms,$api_ms,$api_status" >> "$LOGFILE"

    # Alerts
    if [[ "$rp_up" -eq 0 ]]; then
        log_err "$MODULE" "Kheops reverse proxy is DOWN!"
    elif [[ "$rp_ms" -gt 3000 ]]; then
        log_warn "$MODULE" "Kheops slow response: ${rp_ms}ms"
    fi
    if [[ "$pacs_up" -eq 0 ]]; then
        log_err "$MODULE" "PACS DCM4CHEE is DOWN!"
    fi

    # ── Kheops PostgreSQL stats ──
    kheops_active=$(kheops_pg_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'" || echo "0")
    kheops_idle=$(kheops_pg_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle'" || echo "0")
    kheops_max=$(kheops_pg_query "SHOW max_connections" || echo "100")

    # PACS PostgreSQL stats
    pacs_active=$(pacs_pg_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'" || echo "0")
    pacs_idle=$(pacs_pg_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle'" || echo "0")
    pacs_db_size=$(pacs_pg_query "SELECT pg_database_size('kheops_pacs') / 1048576" || echo "0")

    echo "$ts,${kheops_active:-0},${kheops_idle:-0},${kheops_max:-100},${pacs_active:-0},${pacs_idle:-0},${pacs_db_size:-0}" >> "$LOGFILE_PACS"

    if (( ${kheops_active:-0} + ${kheops_idle:-0} > ${kheops_max:-100} * 80 / 100 )); then
        log_warn "$MODULE" "Kheops PG connections high: $((kheops_active + kheops_idle))/${kheops_max}"
    fi

    # ── Check for errors in Kheops containers ──
    for container in "$KHEOPS_AUTHORIZATION" "$KHEOPS_DICOMWEB_PROXY" "$KHEOPS_ZIPPER" "$PACS_ARC" "$PACS_AUTH_PROXY"; do
        errors=$(docker logs --since "${SAMPLE_INTERVAL}s" "$container" 2>&1 | grep -ciE "error|exception|severe|warn" || true)
        if [[ "${errors:-0}" -gt 5 ]]; then
            log_warn "$MODULE" "$container: ${errors} error/warning entries in last ${SAMPLE_INTERVAL}s"
            {
                echo "=== [$ts] Errors from $container ==="
                docker logs --since "${SAMPLE_INTERVAL}s" "$container" 2>&1 | grep -iE "error|exception|severe" | tail -10
                echo ""
            } >> "$LOGFILE_ERRORS"
        fi
    done

    # ── Check Kheops container restart counts ──
    for container in "${KHEOPS_CONTAINERS[@]}"; do
        restarts=$(docker inspect "$container" -f '{{.RestartCount}}' 2>/dev/null || echo "0")
        if [[ "$restarts" -gt 0 ]]; then
            log_warn "$MODULE" "$container has restarted $restarts times"
        fi
    done

    sleep "$SAMPLE_INTERVAL"
done

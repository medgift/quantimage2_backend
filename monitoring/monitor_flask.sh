#!/usr/bin/env bash
# ============================================================================
# Module 5: Flask API / Backend Monitor
# Probes backend health, measures response times, monitors error rates
# from container logs, tracks Socket.IO connections and eventlet greenlets
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="FLASK"
LOGFILE="${LOG_DIR}/flask_api_${SESSION_ID}.csv"
LOGFILE_ERRORS="${LOG_DIR}/flask_errors_${SESSION_ID}.log"
LOGFILE_SLOW="${LOG_DIR}/flask_slow_requests_${SESSION_ID}.log"

# CSV header
echo "timestamp,backend_reachable,response_time_ms,http_status,error_count_delta,warn_count_delta,socketio_rooms,active_greenlets,open_fds,container_restarts" > "$LOGFILE"

log_header "$MODULE" "Starting Flask API monitor → $LOGFILE"

# Determine the best URL to probe
PROBE_URL="$QI2_BACKEND_URL"

# Test connectivity
if ! curl -sk -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 5 "$PROBE_URL" 2>/dev/null | grep -qE "^[23]"; then
    PROBE_URL="http://localhost:5000"
    if ! curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 5 "$PROBE_URL" 2>/dev/null | grep -qE "^[23]"; then
        log_warn "$MODULE" "Backend not reachable at standard URLs, will keep trying"
    fi
fi

log_header "$MODULE" "Probing backend at: $PROBE_URL"

prev_restart_count=$(docker inspect "$QI2_BACKEND" -f '{{.RestartCount}}' 2>/dev/null || echo "0")
prev_log_position=""

while true; do
    ts=$(csv_ts)
    reachable=0
    resp_time=0
    http_status=0
    error_count=0
    warn_count=0
    socketio_rooms=0
    greenlets=0
    open_fds=0
    restart_count=0

    # ── Health probe (lightweight GET to root) ──
    probe_result=$(curl -sk -o /dev/null -w "%{http_code}|%{time_total}" \
        --connect-timeout 3 --max-time 10 \
        "$PROBE_URL" 2>/dev/null || echo "000|10.0")

    http_status=$(echo "$probe_result" | cut -d'|' -f1)
    resp_time_s=$(echo "$probe_result" | cut -d'|' -f2)
    resp_time=$(echo "scale=0; ${resp_time_s} * 1000" | bc 2>/dev/null || echo "0")

    if [[ "$http_status" =~ ^[23] ]]; then
        reachable=1
    else
        log_err "$MODULE" "Backend returned HTTP $http_status (response time: ${resp_time}ms)"
    fi

    # Alert on slow responses
    if (( $(echo "$resp_time > 2000" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "Slow response: ${resp_time}ms for GET $PROBE_URL"
        echo "=== [$ts] Slow response: ${resp_time}ms GET $PROBE_URL ===" >> "$LOGFILE_SLOW"
    fi

    # ── Container restart count ──
    restart_count=$(docker inspect "$QI2_BACKEND" -f '{{.RestartCount}}' 2>/dev/null || echo "0")
    if [[ "$restart_count" -gt "$prev_restart_count" ]]; then
        log_err "$MODULE" "Backend container restarted! (count: $restart_count)"
    fi
    prev_restart_count=$restart_count

    # ── Error/warning count from recent logs ──
    recent_logs=$(docker logs --since "${SAMPLE_INTERVAL}s" "$QI2_BACKEND" 2>&1 || true)
    error_count=$(echo "$recent_logs" | grep -ciE "error|exception|traceback" || true)
    warn_count=$(echo "$recent_logs" | grep -ciE "warning|warn" || true)

    if [[ "$error_count" -gt 0 ]]; then
        log_warn "$MODULE" "${error_count} errors in last ${SAMPLE_INTERVAL}s"
        {
            echo "=== [$ts] Backend errors (${error_count}) ==="
            echo "$recent_logs" | grep -iE "error|exception|traceback" | tail -10
            echo ""
        } >> "$LOGFILE_ERRORS"
    fi

    # ── Open file descriptors in backend container ──
    open_fds=$(docker exec "$QI2_BACKEND" sh -c 'ls /proc/1/fd 2>/dev/null | wc -l' 2>/dev/null || echo "0")
    if [[ "$open_fds" -gt 500 ]]; then
        log_warn "$MODULE" "Backend has $open_fds open file descriptors (possible leak)"
    fi

    # ── Check eventlet greenlets (via /proc) ──
    greenlets=$(docker exec "$QI2_BACKEND" sh -c 'ls /proc/*/status 2>/dev/null | wc -l' 2>/dev/null || echo "0")

    # ── Socket.IO: check Redis pubsub channels as a proxy for active connections ──
    socketio_rooms=$(docker exec "$REDIS_SOCKET_CONTAINER" redis-cli PUBSUB CHANNELS '*' 2>/dev/null | wc -l || echo "0")

    echo "$ts,$reachable,$resp_time,$http_status,$error_count,$warn_count,$socketio_rooms,$greenlets,$open_fds,$restart_count" >> "$LOGFILE"

    # ── Monitor request patterns from logs (HTTP access-like patterns) ──
    slow_requests=$(echo "$recent_logs" | grep -oP '\d+\.\d+s' | awk -F's' '{if ($1 > 2.0) print $0}' | head -5 || true)
    if [[ -n "$slow_requests" ]]; then
        {
            echo "=== [$ts] Slow request patterns ==="
            echo "$recent_logs" | grep -B1 -A1 -P '\d+\.\d+s' | head -20
            echo ""
        } >> "$LOGFILE_SLOW"
    fi

    sleep "$SAMPLE_INTERVAL"
done

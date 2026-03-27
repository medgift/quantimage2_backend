#!/usr/bin/env bash
# ============================================================================
# Module 4: Celery Worker Monitor
# Tracks worker health, active/reserved/scheduled tasks, task throughput,
# and worker event rates
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="CELERY"
LOGFILE="${LOG_DIR}/celery_workers_${SESSION_ID}.csv"
LOGFILE_EVENTS="${LOG_DIR}/celery_events_${SESSION_ID}.log"

# CSV header
echo "timestamp,extraction_worker_alive,training_worker_alive,extraction_active_tasks,extraction_reserved_tasks,training_active_tasks,training_reserved_tasks,extraction_prefetch_count,training_prefetch_count" > "$LOGFILE"

log_header "$MODULE" "Starting Celery worker monitor → $LOGFILE"

# We use Flower API (lightweight REST) instead of `celery inspect` which is heavy
FLOWER_URL="http://localhost:3333"

flower_api() {
    curl -sf --connect-timeout 3 --max-time 5 "${FLOWER_URL}${1}" 2>/dev/null
}

# Check Flower is accessible
if ! flower_api "/api/workers" > /dev/null 2>&1; then
    # Try via Docker network
    FLOWER_URL="http://$(docker inspect ${QI2_FLOWER} -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null | head -1):3333"
    if ! flower_api "/api/workers" > /dev/null 2>&1; then
        log_warn "$MODULE" "Cannot reach Flower at localhost:3333 or via Docker IP. Falling back to docker exec"
        FLOWER_URL=""
    fi
fi

while true; do
    ts=$(csv_ts)

    ext_alive=0
    train_alive=0
    ext_active=0
    ext_reserved=0
    train_active=0
    train_reserved=0
    ext_prefetch=0
    train_prefetch=0

    if [[ -n "$FLOWER_URL" ]]; then
        # ── Use Flower REST API (lightweight) ──
        workers_json=$(flower_api "/api/workers?refresh=true" || echo "{}")

        if [[ -n "$workers_json" && "$workers_json" != "{}" ]]; then
            # Parse worker info with jq
            # Workers are named like: celery@<hostname>
            ext_alive=$(echo "$workers_json" | jq '[to_entries[] | select(.key | contains("extraction")) | select(.value.status == true or .value.status == null)] | length' 2>/dev/null || echo "0")
            train_alive=$(echo "$workers_json" | jq '[to_entries[] | select(.key | contains("training")) | select(.value.status == true or .value.status == null)] | length' 2>/dev/null || echo "0")

            # Get active tasks
            active_json=$(flower_api "/api/tasks?state=STARTED&limit=50" || echo "{}")
            if [[ -n "$active_json" && "$active_json" != "{}" ]]; then
                ext_active=$(echo "$active_json" | jq '[to_entries[] | select(.value.queue == "extraction" or (.value.worker // "" | contains("extraction")))] | length' 2>/dev/null || echo "0")
                train_active=$(echo "$active_json" | jq '[to_entries[] | select(.value.queue == "training" or (.value.worker // "" | contains("training")))] | length' 2>/dev/null || echo "0")
            fi
        fi
    fi

    # ── Also check container health directly ──
    if container_running "$QI2_CELERY_EXTRACTION"; then
        if [[ "$ext_alive" -eq 0 ]]; then ext_alive=1; fi
    else
        ext_alive=0
        log_err "$MODULE" "Extraction worker container is NOT running!"
    fi

    if container_running "$QI2_CELERY_TRAINING"; then
        if [[ "$train_alive" -eq 0 ]]; then train_alive=1; fi
    else
        train_alive=0
        log_err "$MODULE" "Training worker container is NOT running!"
    fi

    # ── Get prefetch counts from Redis (tasks reserved but not started) ──
    ext_prefetch=$(docker exec "$REDIS_CELERY_CONTAINER" redis-cli LLEN extraction 2>/dev/null || echo "0")
    train_prefetch=$(docker exec "$REDIS_CELERY_CONTAINER" redis-cli LLEN training 2>/dev/null || echo "0")

    echo "$ts,$ext_alive,$train_alive,$ext_active,$ext_reserved,$train_active,$train_reserved,$ext_prefetch,$train_prefetch" >> "$LOGFILE"

    # ── Log extraction/training container recent logs for errors ──
    for container in "$QI2_CELERY_EXTRACTION" "$QI2_CELERY_TRAINING"; do
        errors=$(docker logs --since "${SAMPLE_INTERVAL}s" "$container" 2>&1 | grep -iE "error|exception|traceback|killed|oom" | tail -5 || true)
        if [[ -n "$errors" ]]; then
            log_warn "$MODULE" "Errors in ${container}:"
            {
                echo "=== [$ts] Errors from $container ==="
                echo "$errors"
                echo ""
            } >> "$LOGFILE_EVENTS"
        fi
    done

    # ── Check for memory issues in workers ──
    for container in "$QI2_CELERY_EXTRACTION" "$QI2_CELERY_TRAINING"; do
        mem_info=$(docker stats --no-stream --format '{{.MemPerc}}|{{.MemUsage}}' "$container" 2>/dev/null || echo "0%|0")
        mem_pct=$(echo "$mem_info" | cut -d'|' -f1 | tr -d '%')
        if (( $(echo "${mem_pct:-0} > 70" | bc -l 2>/dev/null || echo 0) )); then
            log_warn "$MODULE" "$container memory: ${mem_info}"
        fi
    done

    sleep "$SAMPLE_INTERVAL"
done

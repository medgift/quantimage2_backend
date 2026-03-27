#!/usr/bin/env bash
# ============================================================================
# Module 3: Redis Monitor (Celery broker + Socket.IO)
# Tracks memory, connections, ops/sec, keyspace, queue depths,
# and task result counts
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="REDIS"
LOGFILE_CELERY="${LOG_DIR}/redis_celery_${SESSION_ID}.csv"
LOGFILE_SOCKET="${LOG_DIR}/redis_socket_${SESSION_ID}.csv"
LOGFILE_QUEUES="${LOG_DIR}/redis_queues_${SESSION_ID}.csv"
LOGFILE_TASKS="${LOG_DIR}/redis_task_counts_${SESSION_ID}.csv"

redis_celery() {
    docker exec "$REDIS_CELERY_CONTAINER" redis-cli "$@" 2>/dev/null
}

redis_socket() {
    docker exec "$REDIS_SOCKET_CONTAINER" redis-cli "$@" 2>/dev/null
}

# CSV headers
echo "timestamp,used_memory_mb,used_memory_peak_mb,maxmemory_mb,mem_fragmentation_ratio,connected_clients,blocked_clients,total_commands_processed,ops_per_sec,keyspace_hits,keyspace_misses,hit_rate_pct,expired_keys,evicted_keys,total_keys" > "$LOGFILE_CELERY"
echo "timestamp,used_memory_mb,used_memory_peak_mb,connected_clients,blocked_clients,ops_per_sec,total_keys,pubsub_channels,pubsub_patterns" > "$LOGFILE_SOCKET"
echo "timestamp,extraction_queue_len,training_queue_len,active_reservations" > "$LOGFILE_QUEUES"
echo "timestamp,tasks_started,tasks_success,tasks_failure,tasks_pending,tasks_progress,tasks_revoked,tasks_retry" > "$LOGFILE_TASKS"

log_header "$MODULE" "Starting Redis monitor → ${LOGFILE_CELERY##*/}, ${LOGFILE_SOCKET##*/}"

get_redis_val() {
    echo "$1" | grep -i "^$2:" | cut -d: -f2 | tr -d $'\r' | xargs 2>/dev/null || echo ""
}

prev_commands_celery=0
first_run=true

while true; do
    ts=$(csv_ts)

    # ─── Celery Redis (broker + result backend) ───
    info_celery=$(redis_celery INFO 2>/dev/null || echo "")
    if [[ -n "$info_celery" ]]; then
        mem_used=$(get_redis_val "$info_celery" "used_memory")
        mem_used_mb=$(echo "scale=2; ${mem_used:-0} / 1048576" | bc 2>/dev/null || echo "0")
        mem_peak=$(get_redis_val "$info_celery" "used_memory_peak")
        mem_peak_mb=$(echo "scale=2; ${mem_peak:-0} / 1048576" | bc 2>/dev/null || echo "0")
        maxmem=$(get_redis_val "$info_celery" "maxmemory")
        maxmem_mb=$(echo "scale=2; ${maxmem:-0} / 1048576" | bc 2>/dev/null || echo "0")
        frag=$(get_redis_val "$info_celery" "mem_fragmentation_ratio")
        clients=$(get_redis_val "$info_celery" "connected_clients")
        blocked=$(get_redis_val "$info_celery" "blocked_clients")
        commands=$(get_redis_val "$info_celery" "total_commands_processed")
        ops=$(get_redis_val "$info_celery" "instantaneous_ops_per_sec")
        hits=$(get_redis_val "$info_celery" "keyspace_hits")
        misses=$(get_redis_val "$info_celery" "keyspace_misses")
        expired=$(get_redis_val "$info_celery" "expired_keys")
        evicted=$(get_redis_val "$info_celery" "evicted_keys")

        # Hit rate
        total_lookups=$(( ${hits:-0} + ${misses:-0} ))
        if [[ $total_lookups -gt 0 ]]; then
            hit_rate=$(echo "scale=1; ${hits:-0} * 100 / $total_lookups" | bc 2>/dev/null || echo "0")
        else
            hit_rate="100.0"
        fi

        # Total keys in db0
        db0_info=$(get_redis_val "$info_celery" "db0")
        total_keys=$(echo "$db0_info" | grep -oP 'keys=\K\d+' || echo "0")

        echo "$ts,${mem_used_mb},${mem_peak_mb},${maxmem_mb},${frag:-0},${clients:-0},${blocked:-0},${commands:-0},${ops:-0},${hits:-0},${misses:-0},${hit_rate},${expired:-0},${evicted:-0},${total_keys}" >> "$LOGFILE_CELERY"

        # Alerts
        if [[ "${maxmem:-0}" -gt 0 ]]; then
            mem_pct=$(echo "scale=0; ${mem_used:-0} * 100 / ${maxmem:-1}" | bc 2>/dev/null || echo "0")
            if [[ "$mem_pct" -gt 80 ]]; then
                log_warn "$MODULE" "Celery Redis memory at ${mem_pct}% (${mem_used_mb}MB / ${maxmem_mb}MB)"
            fi
        fi
        if (( $(echo "${frag:-1} > 1.5" | bc -l 2>/dev/null || echo 0) )); then
            log_warn "$MODULE" "Celery Redis fragmentation ratio: ${frag} (>1.5 indicates fragmentation)"
        fi
        if [[ "${blocked:-0}" -gt 0 ]]; then
            log_warn "$MODULE" "Celery Redis: ${blocked} blocked clients"
        fi
        if [[ "${evicted:-0}" -gt 0 ]]; then
            log_warn "$MODULE" "Celery Redis has evicted ${evicted} keys!"
        fi
    else
        log_warn "$MODULE" "Cannot reach Celery Redis"
    fi

    # ─── Socket.IO Redis ───
    info_socket=$(redis_socket INFO 2>/dev/null || echo "")
    if [[ -n "$info_socket" ]]; then
        s_mem=$(get_redis_val "$info_socket" "used_memory")
        s_mem_mb=$(echo "scale=2; ${s_mem:-0} / 1048576" | bc 2>/dev/null || echo "0")
        s_mem_peak=$(get_redis_val "$info_socket" "used_memory_peak")
        s_mem_peak_mb=$(echo "scale=2; ${s_mem_peak:-0} / 1048576" | bc 2>/dev/null || echo "0")
        s_clients=$(get_redis_val "$info_socket" "connected_clients")
        s_blocked=$(get_redis_val "$info_socket" "blocked_clients")
        s_ops=$(get_redis_val "$info_socket" "instantaneous_ops_per_sec")
        s_db0=$(get_redis_val "$info_socket" "db0")
        s_keys=$(echo "$s_db0" | grep -oP 'keys=\K\d+' || echo "0")
        s_pubsub_ch=$(get_redis_val "$info_socket" "pubsub_channels")
        s_pubsub_pat=$(get_redis_val "$info_socket" "pubsub_patterns")

        echo "$ts,${s_mem_mb},${s_mem_peak_mb},${s_clients:-0},${s_blocked:-0},${s_ops:-0},${s_keys},${s_pubsub_ch:-0},${s_pubsub_pat:-0}" >> "$LOGFILE_SOCKET"

        if [[ "${s_clients:-0}" -gt 50 ]]; then
            log_warn "$MODULE" "Socket.IO Redis: ${s_clients} connected clients (many Socket.IO connections?)"
        fi
    fi

    # ─── Celery Queue depths ───
    extraction_len=$(redis_celery LLEN extraction 2>/dev/null || echo "0")
    training_len=$(redis_celery LLEN training 2>/dev/null || echo "0")
    # Reservations (tasks claimed by workers but not yet acknowledged)
    reservations=$(redis_celery KEYS "unacked*" 2>/dev/null | wc -l || echo "0")

    echo "$ts,$extraction_len,$training_len,$reservations" >> "$LOGFILE_QUEUES"

    if [[ "$extraction_len" -gt 5 ]]; then
        log_warn "$MODULE" "Extraction queue depth: ${extraction_len} (backlog building)"
    fi
    if [[ "$training_len" -gt 5 ]]; then
        log_warn "$MODULE" "Training queue depth: ${training_len} (backlog building)"
    fi

    # ─── Task result counts (sample up to 500 keys, lightweight) ───
    task_counts=$(redis_celery --no-auth-warning EVAL "
        local cursor = '0'
        local counts = {started=0, success=0, failure=0, pending=0, progress=0, revoked=0, retry=0}
        local scanned = 0
        repeat
            local result = redis.call('SCAN', cursor, 'MATCH', 'celery-task-meta-*', 'COUNT', 100)
            cursor = result[1]
            for _, key in ipairs(result[2]) do
                local val = redis.call('GET', key)
                if val then
                    if string.find(val, '\"STARTED\"') then counts.started = counts.started + 1
                    elseif string.find(val, '\"SUCCESS\"') then counts.success = counts.success + 1
                    elseif string.find(val, '\"FAILURE\"') then counts.failure = counts.failure + 1
                    elseif string.find(val, '\"PENDING\"') then counts.pending = counts.pending + 1
                    elseif string.find(val, '\"PROGRESS\"') then counts.progress = counts.progress + 1
                    elseif string.find(val, '\"REVOKED\"') then counts.revoked = counts.revoked + 1
                    elseif string.find(val, '\"RETRY\"') then counts.retry = counts.retry + 1
                    end
                end
                scanned = scanned + 1
                if scanned >= 500 then break end
            end
        until cursor == '0' or scanned >= 500
        return counts.started..','..counts.success..','..counts.failure..','..counts.pending..','..counts.progress..','..counts.revoked..','..counts.retry
    " 0 2>/dev/null || echo "0,0,0,0,0,0,0")

    echo "$ts,$task_counts" >> "$LOGFILE_TASKS"

    # Alert on stuck tasks
    started_count=$(echo "$task_counts" | cut -d, -f1)
    progress_count=$(echo "$task_counts" | cut -d, -f5)
    active_tasks=$((${started_count:-0} + ${progress_count:-0}))
    if [[ "$active_tasks" -gt 10 ]]; then
        log_warn "$MODULE" "${active_tasks} tasks in STARTED/PROGRESS state (possible stuck tasks?)"
    fi

    sleep "$SAMPLE_INTERVAL"
done

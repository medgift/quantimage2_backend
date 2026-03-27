#!/usr/bin/env bash
# ============================================================================
# Module 2: MySQL Database Monitor
# Tracks connections, slow queries, locks, thread states, table sizes,
# InnoDB buffer pool, and running queries
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="MYSQL"
LOGFILE="${LOG_DIR}/mysql_stats_${SESSION_ID}.csv"
LOGFILE_QUERIES="${LOG_DIR}/mysql_slow_queries_${SESSION_ID}.log"
LOGFILE_LOCKS="${LOG_DIR}/mysql_locks_${SESSION_ID}.log"

mysql_cmd() {
    docker exec "$QI2_DB" mysql -u root -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" \
        --connect-timeout=5 -N -B -e "$1" 2>/dev/null
}

mysql_cmd_raw() {
    docker exec "$QI2_DB" mysql -u root -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" \
        --connect-timeout=5 -e "$1" 2>/dev/null
}

# Check MySQL is reachable via Docker exec
if ! mysql_cmd "SELECT 1" > /dev/null 2>&1; then
    log_warn "$MODULE" "Cannot connect via Docker exec with root, trying user account..."

    # Fall back to regular user
    mysql_cmd() {
        docker exec "$QI2_DB" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" \
            --connect-timeout=5 -N -B -e "$1" 2>/dev/null
    }
    mysql_cmd_raw() {
        docker exec "$QI2_DB" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" \
            --connect-timeout=5 -e "$1" 2>/dev/null
    }

    if ! mysql_cmd "SELECT 1" > /dev/null 2>&1; then
        log_err "$MODULE" "Cannot connect to MySQL. Exiting."
        exit 1
    fi
    log_ok "$MODULE" "Connected to MySQL (limited privileges)"
fi

# CSV header
echo "timestamp,threads_connected,threads_running,max_connections,connections_pct,slow_queries_total,slow_queries_delta,questions_total,questions_delta,innodb_buffer_pool_hit_rate,innodb_buffer_pool_used_pct,innodb_row_lock_waits,innodb_row_lock_time_avg_ms,open_tables,table_locks_waited,aborted_connects,aborted_clients,created_tmp_tables,created_tmp_disk_tables" > "$LOGFILE"

log_header "$MODULE" "Starting MySQL monitor → $LOGFILE"
log_header "$MODULE" "Slow queries log → $LOGFILE_QUERIES"

prev_slow_queries=0
prev_questions=0
first_run=true

while true; do
    ts=$(csv_ts)

    # ── Global status (single query, pick what we need) ──
    status=$(mysql_cmd "
        SHOW GLOBAL STATUS WHERE Variable_name IN (
            'Threads_connected', 'Threads_running', 'Slow_queries',
            'Questions', 'Innodb_buffer_pool_read_requests',
            'Innodb_buffer_pool_reads', 'Innodb_buffer_pool_pages_total',
            'Innodb_buffer_pool_pages_free', 'Innodb_row_lock_waits',
            'Innodb_row_lock_time_avg', 'Open_tables',
            'Table_locks_waited', 'Aborted_connects', 'Aborted_clients',
            'Created_tmp_tables', 'Created_tmp_disk_tables'
        )
    " 2>/dev/null || echo "")

    if [[ -z "$status" ]]; then
        log_warn "$MODULE" "Failed to query MySQL status"
        sleep "$SAMPLE_INTERVAL"
        continue
    fi

    get_val() {
        echo "$status" | grep -i "^$1" | awk '{print $2}' | head -1
    }

    threads_conn=$(get_val "Threads_connected")
    threads_run=$(get_val "Threads_running")
    slow_queries=$(get_val "Slow_queries")
    questions=$(get_val "Questions")
    bp_read_requests=$(get_val "Innodb_buffer_pool_read_requests")
    bp_reads=$(get_val "Innodb_buffer_pool_reads")
    bp_pages_total=$(get_val "Innodb_buffer_pool_pages_total")
    bp_pages_free=$(get_val "Innodb_buffer_pool_pages_free")
    innodb_lock_waits=$(get_val "Innodb_row_lock_waits")
    innodb_lock_time_avg=$(get_val "Innodb_row_lock_time_avg")
    open_tables=$(get_val "Open_tables")
    table_locks_waited=$(get_val "Table_locks_waited")
    aborted_connects=$(get_val "Aborted_connects")
    aborted_clients=$(get_val "Aborted_clients")
    created_tmp=$(get_val "Created_tmp_tables")
    created_tmp_disk=$(get_val "Created_tmp_disk_tables")

    max_connections=$(mysql_cmd "SELECT @@max_connections" 2>/dev/null || echo "151")

    # Derived metrics
    conn_pct=$(echo "scale=1; ${threads_conn:-0} * 100 / ${max_connections:-151}" | bc 2>/dev/null || echo "0")

    # Buffer pool hit rate (higher is better)
    if [[ "${bp_read_requests:-0}" -gt 0 ]]; then
        bp_hit_rate=$(echo "scale=2; (1 - ${bp_reads:-0} / ${bp_read_requests:-1}) * 100" | bc 2>/dev/null || echo "0")
    else
        bp_hit_rate="100.00"
    fi

    # Buffer pool usage %
    if [[ "${bp_pages_total:-0}" -gt 0 ]]; then
        bp_used_pct=$(echo "scale=1; (${bp_pages_total:-0} - ${bp_pages_free:-0}) * 100 / ${bp_pages_total:-1}" | bc 2>/dev/null || echo "0")
    else
        bp_used_pct="0"
    fi

    # Delta calculations
    if $first_run; then
        slow_delta=0
        questions_delta=0
        first_run=false
    else
        slow_delta=$((${slow_queries:-0} - prev_slow_queries))
        questions_delta=$((${questions:-0} - prev_questions))
    fi
    prev_slow_queries=${slow_queries:-0}
    prev_questions=${questions:-0}

    echo "$ts,${threads_conn:-0},${threads_run:-0},${max_connections:-151},${conn_pct},${slow_queries:-0},${slow_delta},${questions:-0},${questions_delta},${bp_hit_rate},${bp_used_pct},${innodb_lock_waits:-0},${innodb_lock_time_avg:-0},${open_tables:-0},${table_locks_waited:-0},${aborted_connects:-0},${aborted_clients:-0},${created_tmp:-0},${created_tmp_disk:-0}" >> "$LOGFILE"

    # ── Alerts ──
    if (( $(echo "$conn_pct > 70" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "Connection pool at ${conn_pct}% (${threads_conn}/${max_connections})"
    fi
    if [[ "$slow_delta" -gt 0 ]]; then
        log_warn "$MODULE" "+${slow_delta} slow queries in last ${SAMPLE_INTERVAL}s"
    fi
    if (( $(echo "$bp_hit_rate < 95" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "InnoDB buffer pool hit rate: ${bp_hit_rate}% (should be >99%)"
    fi
    if [[ "${threads_run:-0}" -gt 10 ]]; then
        log_warn "$MODULE" "${threads_run} threads running (possible contention)"
    fi

    # ── Log active queries (only if something interesting is happening) ──
    if [[ "${threads_run:-0}" -gt 2 ]]; then
        {
            echo "=== [$ts] Active queries (threads_running=${threads_run}) ==="
            mysql_cmd_raw "
                SELECT ID, USER, HOST, DB, COMMAND, TIME, STATE,
                       LEFT(INFO, 200) as QUERY
                FROM information_schema.PROCESSLIST
                WHERE COMMAND != 'Sleep'
                  AND USER != 'system user'
                  AND INFO IS NOT NULL
                ORDER BY TIME DESC
                LIMIT 20
            " 2>/dev/null || echo "(query failed)"
        } >> "$LOGFILE_QUERIES"
    fi

    # ── Check for lock contention ──
    lock_waits=$(mysql_cmd "
        SELECT COUNT(*) FROM information_schema.INNODB_TRX
        WHERE trx_state = 'LOCK WAIT'
    " 2>/dev/null || echo "0")

    if [[ "${lock_waits:-0}" -gt 0 ]]; then
        log_warn "$MODULE" "${lock_waits} transactions waiting for locks!"
        {
            echo "=== [$ts] Lock contention detected (${lock_waits} waiters) ==="
            mysql_cmd_raw "
                SELECT trx_id, trx_state, trx_started,
                       TIMESTAMPDIFF(SECOND, trx_started, NOW()) as age_seconds,
                       trx_rows_locked, trx_rows_modified,
                       LEFT(trx_query, 200) as query
                FROM information_schema.INNODB_TRX
                ORDER BY trx_started
                LIMIT 20
            " 2>/dev/null || echo "(query failed)"
        } >> "$LOGFILE_LOCKS"
    fi

    sleep "$SAMPLE_INTERVAL"
done

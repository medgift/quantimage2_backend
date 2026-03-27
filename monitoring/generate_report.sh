#!/usr/bin/env bash
# ============================================================================
# Report Generator — Post-session analysis
# Reads all CSV logs from a session and produces a human-readable report
# with key findings, bottleneck identification, and recommendations
#
# Usage: ./generate_report.sh [SESSION_ID]
#        (if no SESSION_ID given, uses the latest session)
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Override SESSION_ID if provided as argument
if [[ -n "${1:-}" ]]; then
    SESSION_ID="$1"
else
    # Find latest session
    SESSION_ID=$(ls -1 "$LOG_DIR"/docker_resources_*.csv 2>/dev/null | sort | tail -1 | grep -oP '\d{8}_\d{6}' || echo "")
    if [[ -z "$SESSION_ID" ]]; then
        echo "No monitoring sessions found in $LOG_DIR"
        exit 1
    fi
fi

REPORT="${REPORT_DIR}/report_${SESSION_ID}.md"

log_header "REPORT" "Generating report for session $SESSION_ID → $REPORT"

cat > "$REPORT" << 'HEADER'
# QuantImage v2 — Performance Monitoring Report
HEADER

echo "**Session:** $SESSION_ID" >> "$REPORT"
echo "**Generated:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT"
echo "**Host:** $(hostname)" >> "$REPORT"
echo "**CPUs:** $(nproc) | **RAM:** $(free -h | awk '/Mem:/{print $2}')" >> "$REPORT"
echo "" >> "$REPORT"

# Helper: compute stats from a CSV column
csv_stats() {
    local file="$1"
    local col="$2"  # column number (1-based)
    if [[ ! -f "$file" ]]; then
        echo "N/A|N/A|N/A|N/A|0"
        return
    fi
    tail -n +2 "$file" | cut -d',' -f"$col" | grep -oP '[\d.]+' | awk '
    BEGIN {min=999999999; max=0; sum=0; n=0}
    {
        val=$1+0;
        sum+=val; n++;
        if(val<min) min=val;
        if(val>max) max=val;
    }
    END {
        if(n>0) printf "%.1f|%.1f|%.1f|%.1f|%d\n", min, max, sum/n, sum, n
        else printf "N/A|N/A|N/A|N/A|0\n"
    }'
}

csv_last() {
    local file="$1"
    local col="$2"
    if [[ ! -f "$file" ]]; then echo "N/A"; return; fi
    tail -1 "$file" | cut -d',' -f"$col"
}

csv_count_above() {
    local file="$1"
    local col="$2"
    local threshold="$3"
    if [[ ! -f "$file" ]]; then echo "0"; return; fi
    tail -n +2 "$file" | cut -d',' -f"$col" | awk -v t="$threshold" '{if($1+0 > t) c++} END {print c+0}'
}

duration_info() {
    local file="$1"
    if [[ ! -f "$file" ]]; then echo "N/A"; return; fi
    local first_ts=$(head -2 "$file" | tail -1 | cut -d',' -f1)
    local last_ts=$(tail -1 "$file" | cut -d',' -f1)
    echo "${first_ts} → ${last_ts}"
}

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 1. Monitoring Duration & Self-Resource Usage" >> "$REPORT"
echo "" >> "$REPORT"

SELF_FILE="${LOG_DIR}/self_resource_${SESSION_ID}.csv"
echo "**Duration:** $(duration_info "$SELF_FILE")" >> "$REPORT"

if [[ -f "$SELF_FILE" ]]; then
    IFS='|' read -r cpu_min cpu_max cpu_avg _ samples <<< "$(csv_stats "$SELF_FILE" 3)"
    IFS='|' read -r mem_min mem_max mem_avg _ _ <<< "$(csv_stats "$SELF_FILE" 4)"
    last_log_size=$(csv_last "$SELF_FILE" 6)

    echo "" >> "$REPORT"
    echo "| Metric | Min | Avg | Max |" >> "$REPORT"
    echo "|--------|-----|-----|-----|" >> "$REPORT"
    echo "| Monitoring CPU % | ${cpu_min}% | ${cpu_avg}% | ${cpu_max}% |" >> "$REPORT"
    echo "| Monitoring RAM (MB) | ${mem_min} | ${mem_avg} | ${mem_max} |" >> "$REPORT"
    echo "| Log directory size | — | — | ${last_log_size} MB |" >> "$REPORT"
    echo "| Samples collected | — | — | ${samples} |" >> "$REPORT"
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 2. Host System Resources" >> "$REPORT"
echo "" >> "$REPORT"

SYS_FILE="${LOG_DIR}/system_stats_${SESSION_ID}.csv"
if [[ -f "$SYS_FILE" ]]; then
    IFS='|' read -r _ cpu_user_max cpu_user_avg _ _ <<< "$(csv_stats "$SYS_FILE" 2)"
    IFS='|' read -r _ cpu_sys_max cpu_sys_avg _ _ <<< "$(csv_stats "$SYS_FILE" 3)"
    IFS='|' read -r _ iowait_max iowait_avg _ _ <<< "$(csv_stats "$SYS_FILE" 4)"
    IFS='|' read -r _ load1_max load1_avg _ _ <<< "$(csv_stats "$SYS_FILE" 6)"
    IFS='|' read -r _ mempct_max mempct_avg _ _ <<< "$(csv_stats "$SYS_FILE" 12)"
    high_iowait=$(csv_count_above "$SYS_FILE" 4 20)
    high_load=$(csv_count_above "$SYS_FILE" 6 "$(nproc)")

    echo "| Metric | Avg | Max | Alerts |" >> "$REPORT"
    echo "|--------|-----|-----|--------|" >> "$REPORT"
    echo "| CPU User % | ${cpu_user_avg}% | ${cpu_user_max}% | — |" >> "$REPORT"
    echo "| CPU System % | ${cpu_sys_avg}% | ${cpu_sys_max}% | — |" >> "$REPORT"
    echo "| CPU I/O Wait % | ${iowait_avg}% | ${iowait_max}% | ${high_iowait} samples >20% |" >> "$REPORT"
    echo "| Load Average (1m) | ${load1_avg} | ${load1_max} | ${high_load} samples >$(nproc) |" >> "$REPORT"
    echo "| Memory Usage % | ${mempct_avg}% | ${mempct_max}% | — |" >> "$REPORT"

    if (( $(echo "${iowait_max:-0} > 20" | bc -l 2>/dev/null || echo 0) )); then
        echo "" >> "$REPORT"
        echo "> **⚠ Finding:** High I/O wait detected (max ${iowait_max}%). Disk I/O may be a bottleneck during extraction tasks." >> "$REPORT"
    fi
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 3. Docker Container Resources" >> "$REPORT"
echo "" >> "$REPORT"

DOCKER_FILE="${LOG_DIR}/docker_resources_${SESSION_ID}.csv"
if [[ -f "$DOCKER_FILE" ]]; then
    echo "### Peak Resource Usage by Container" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Container | CPU Max % | CPU Avg % | Mem Max % | Mem Max MB | Alerts |" >> "$REPORT"
    echo "|-----------|-----------|-----------|-----------|------------|--------|" >> "$REPORT"

    for container in "${ALL_CONTAINERS[@]}"; do
        container_data=$(grep ",$container," "$DOCKER_FILE" || true)
        if [[ -n "$container_data" ]]; then
            cpu_max=$(echo "$container_data" | cut -d',' -f3 | sort -n | tail -1)
            cpu_avg=$(echo "$container_data" | cut -d',' -f3 | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}')
            mem_max_pct=$(echo "$container_data" | cut -d',' -f4 | sort -n | tail -1)
            mem_max_mb=$(echo "$container_data" | cut -d',' -f5 | sort -n | tail -1)
            alerts=""
            if (( $(echo "${cpu_max:-0} > 80" | bc -l 2>/dev/null || echo 0) )); then alerts+="HIGH CPU "; fi
            if (( $(echo "${mem_max_pct:-0} > 80" | bc -l 2>/dev/null || echo 0) )); then alerts+="HIGH MEM "; fi
            short_name=$(echo "$container" | sed 's/quantimage2-//;s/kheops-//')
            echo "| $short_name | ${cpu_max}% | ${cpu_avg}% | ${mem_max_pct}% | ${mem_max_mb} | ${alerts:-OK} |" >> "$REPORT"
        fi
    done
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 4. MySQL Database" >> "$REPORT"
echo "" >> "$REPORT"

MYSQL_FILE="${LOG_DIR}/mysql_stats_${SESSION_ID}.csv"
if [[ -f "$MYSQL_FILE" ]]; then
    IFS='|' read -r _ conn_max conn_avg _ _ <<< "$(csv_stats "$MYSQL_FILE" 2)"
    IFS='|' read -r _ threads_run_max threads_run_avg _ _ <<< "$(csv_stats "$MYSQL_FILE" 3)"
    IFS='|' read -r _ connpct_max connpct_avg _ _ <<< "$(csv_stats "$MYSQL_FILE" 5)"
    IFS='|' read -r _ _ _ slow_total _ <<< "$(csv_stats "$MYSQL_FILE" 7)"
    IFS='|' read -r _ bp_hit_min bp_hit_avg _ _ <<< "$(csv_stats "$MYSQL_FILE" 10)"
    IFS='|' read -r _ lockwaits_max _ _ _ <<< "$(csv_stats "$MYSQL_FILE" 12)"

    echo "| Metric | Avg | Max/Min | Alert |" >> "$REPORT"
    echo "|--------|-----|---------|-------|" >> "$REPORT"
    echo "| Connected Threads | ${conn_avg} | ${conn_max} | — |" >> "$REPORT"
    echo "| Running Threads | ${threads_run_avg} | ${threads_run_max} | $(if (( $(echo "${threads_run_max:-0} > 10" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ contention'; else echo 'OK'; fi) |" >> "$REPORT"
    echo "| Connection Pool % | ${connpct_avg}% | ${connpct_max}% | $(if (( $(echo "${connpct_max:-0} > 70" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ near limit'; else echo 'OK'; fi) |" >> "$REPORT"
    echo "| New Slow Queries | — | ${slow_total} total | $(if (( $(echo "${slow_total:-0} > 0" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ review needed'; else echo 'OK'; fi) |" >> "$REPORT"
    echo "| Buffer Pool Hit Rate | ${bp_hit_avg}% | ${bp_hit_min}% (min) | $(if (( $(echo "${bp_hit_min:-100} < 95" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ low cache hit'; else echo 'OK'; fi) |" >> "$REPORT"
    echo "| InnoDB Lock Waits (max) | — | ${lockwaits_max} | $(if (( $(echo "${lockwaits_max:-0} > 0" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ lock contention'; else echo 'OK'; fi) |" >> "$REPORT"

    lock_log="${LOG_DIR}/mysql_locks_${SESSION_ID}.log"
    if [[ -f "$lock_log" && -s "$lock_log" ]]; then
        echo "" >> "$REPORT"
        echo "> **⚠ Finding:** Lock contention was detected. See \`mysql_locks_${SESSION_ID}.log\` for details." >> "$REPORT"
    fi

    slow_log="${LOG_DIR}/mysql_slow_queries_${SESSION_ID}.log"
    if [[ -f "$slow_log" && -s "$slow_log" ]]; then
        echo "" >> "$REPORT"
        echo "> **⚠ Finding:** Active query snapshots captured. See \`mysql_slow_queries_${SESSION_ID}.log\`." >> "$REPORT"
    fi
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 5. Redis (Celery Broker & Socket.IO)" >> "$REPORT"
echo "" >> "$REPORT"

REDIS_C="${LOG_DIR}/redis_celery_${SESSION_ID}.csv"
REDIS_S="${LOG_DIR}/redis_socket_${SESSION_ID}.csv"
REDIS_Q="${LOG_DIR}/redis_queues_${SESSION_ID}.csv"

if [[ -f "$REDIS_C" ]]; then
    echo "### Celery Redis" >> "$REPORT"
    echo "" >> "$REPORT"
    IFS='|' read -r _ mem_max mem_avg _ _ <<< "$(csv_stats "$REDIS_C" 2)"
    IFS='|' read -r _ clients_max clients_avg _ _ <<< "$(csv_stats "$REDIS_C" 6)"
    IFS='|' read -r _ ops_max ops_avg _ _ <<< "$(csv_stats "$REDIS_C" 9)"
    IFS='|' read -r _ frag_max frag_avg _ _ <<< "$(csv_stats "$REDIS_C" 5)"
    IFS='|' read -r _ hit_min hit_avg _ _ <<< "$(csv_stats "$REDIS_C" 12)"
    last_keys=$(csv_last "$REDIS_C" 15)

    echo "| Metric | Avg | Max/Min |" >> "$REPORT"
    echo "|--------|-----|---------|" >> "$REPORT"
    echo "| Memory (MB) | ${mem_avg} | ${mem_max} |" >> "$REPORT"
    echo "| Connected Clients | ${clients_avg} | ${clients_max} |" >> "$REPORT"
    echo "| Ops/sec | ${ops_avg} | ${ops_max} |" >> "$REPORT"
    echo "| Fragmentation Ratio | ${frag_avg} | ${frag_max} |" >> "$REPORT"
    echo "| Hit Rate % | ${hit_avg}% | ${hit_min}% (min) |" >> "$REPORT"
    echo "| Total Keys (last) | — | ${last_keys} |" >> "$REPORT"
fi

if [[ -f "$REDIS_Q" ]]; then
    echo "" >> "$REPORT"
    echo "### Queue Depths" >> "$REPORT"
    echo "" >> "$REPORT"
    IFS='|' read -r _ ext_q_max ext_q_avg _ _ <<< "$(csv_stats "$REDIS_Q" 2)"
    IFS='|' read -r _ train_q_max train_q_avg _ _ <<< "$(csv_stats "$REDIS_Q" 3)"

    echo "| Queue | Avg Depth | Max Depth | Alert |" >> "$REPORT"
    echo "|-------|-----------|-----------|-------|" >> "$REPORT"
    echo "| Extraction | ${ext_q_avg} | ${ext_q_max} | $(if (( $(echo "${ext_q_max:-0} > 5" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ backlog'; else echo 'OK'; fi) |" >> "$REPORT"
    echo "| Training | ${train_q_avg} | ${train_q_max} | $(if (( $(echo "${train_q_max:-0} > 5" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ backlog'; else echo 'OK'; fi) |" >> "$REPORT"
fi

if [[ -f "$REDIS_S" ]]; then
    echo "" >> "$REPORT"
    echo "### Socket.IO Redis" >> "$REPORT"
    echo "" >> "$REPORT"
    IFS='|' read -r _ s_mem_max s_mem_avg _ _ <<< "$(csv_stats "$REDIS_S" 2)"
    IFS='|' read -r _ s_clients_max s_clients_avg _ _ <<< "$(csv_stats "$REDIS_S" 4)"
    IFS='|' read -r _ s_pubsub_max s_pubsub_avg _ _ <<< "$(csv_stats "$REDIS_S" 8)"

    echo "| Metric | Avg | Max |" >> "$REPORT"
    echo "|--------|-----|-----|" >> "$REPORT"
    echo "| Memory (MB) | ${s_mem_avg} | ${s_mem_max} |" >> "$REPORT"
    echo "| Connected Clients | ${s_clients_avg} | ${s_clients_max} |" >> "$REPORT"
    echo "| PubSub Channels | ${s_pubsub_avg} | ${s_pubsub_max} |" >> "$REPORT"
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 6. Celery Workers" >> "$REPORT"
echo "" >> "$REPORT"

CELERY_FILE="${LOG_DIR}/celery_workers_${SESSION_ID}.csv"
TASK_FILE="${LOG_DIR}/redis_task_counts_${SESSION_ID}.csv"

if [[ -f "$CELERY_FILE" ]]; then
    IFS='|' read -r _ ext_active_max ext_active_avg _ _ <<< "$(csv_stats "$CELERY_FILE" 4)"
    IFS='|' read -r _ train_active_max train_active_avg _ _ <<< "$(csv_stats "$CELERY_FILE" 6)"

    echo "| Metric | Avg | Max |" >> "$REPORT"
    echo "|--------|-----|-----|" >> "$REPORT"
    echo "| Extraction Active Tasks | ${ext_active_avg} | ${ext_active_max} |" >> "$REPORT"
    echo "| Training Active Tasks | ${train_active_avg} | ${train_active_max} |" >> "$REPORT"
fi

if [[ -f "$TASK_FILE" ]]; then
    last_line=$(tail -1 "$TASK_FILE")
    echo "" >> "$REPORT"
    echo "**Last task state snapshot:** $(echo "$last_line" | cut -d',' -f1)" >> "$REPORT"
    echo "- STARTED: $(echo "$last_line" | cut -d',' -f2)" >> "$REPORT"
    echo "- SUCCESS: $(echo "$last_line" | cut -d',' -f3)" >> "$REPORT"
    echo "- FAILURE: $(echo "$last_line" | cut -d',' -f4)" >> "$REPORT"
    echo "- PROGRESS: $(echo "$last_line" | cut -d',' -f6)" >> "$REPORT"
fi

celery_errors="${LOG_DIR}/celery_events_${SESSION_ID}.log"
if [[ -f "$celery_errors" && -s "$celery_errors" ]]; then
    echo "" >> "$REPORT"
    echo "> **⚠ Finding:** Celery worker errors detected. See \`celery_events_${SESSION_ID}.log\`." >> "$REPORT"
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 7. Flask Backend" >> "$REPORT"
echo "" >> "$REPORT"

FLASK_FILE="${LOG_DIR}/flask_api_${SESSION_ID}.csv"
if [[ -f "$FLASK_FILE" ]]; then
    IFS='|' read -r _ resp_max resp_avg _ _ <<< "$(csv_stats "$FLASK_FILE" 3)"
    IFS='|' read -r _ _ errors_total _ _ <<< "$(csv_stats "$FLASK_FILE" 5)"
    IFS='|' read -r _ fds_max fds_avg _ _ <<< "$(csv_stats "$FLASK_FILE" 9)"
    IFS='|' read -r _ sio_max sio_avg _ _ <<< "$(csv_stats "$FLASK_FILE" 7)"
    slow_count=$(csv_count_above "$FLASK_FILE" 3 2000)
    unreachable=$(csv_count_above "$FLASK_FILE" 2 0 | head -1)  # count where reachable=0

    echo "| Metric | Avg | Max | Alert |" >> "$REPORT"
    echo "|--------|-----|-----|-------|" >> "$REPORT"
    echo "| Response Time (ms) | ${resp_avg} | ${resp_max} | ${slow_count} samples >2s |" >> "$REPORT"
    echo "| Error Log Entries | — | ${errors_total} total | — |" >> "$REPORT"
    echo "| Open File Descriptors | ${fds_avg} | ${fds_max} | $(if (( $(echo "${fds_max:-0} > 500" | bc -l 2>/dev/null || echo 0) )); then echo '⚠ possible leak'; else echo 'OK'; fi) |" >> "$REPORT"
    echo "| Socket.IO Channels | ${sio_avg} | ${sio_max} | — |" >> "$REPORT"

    flask_errors="${LOG_DIR}/flask_errors_${SESSION_ID}.log"
    if [[ -f "$flask_errors" && -s "$flask_errors" ]]; then
        echo "" >> "$REPORT"
        echo "> **⚠ Finding:** Backend errors detected. See \`flask_errors_${SESSION_ID}.log\`." >> "$REPORT"
    fi
    flask_slow="${LOG_DIR}/flask_slow_requests_${SESSION_ID}.log"
    if [[ -f "$flask_slow" && -s "$flask_slow" ]]; then
        echo "" >> "$REPORT"
        echo "> **⚠ Finding:** Slow requests detected. See \`flask_slow_requests_${SESSION_ID}.log\`." >> "$REPORT"
    fi
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 8. Kheops (PACS)" >> "$REPORT"
echo "" >> "$REPORT"

KHEOPS_FILE="${LOG_DIR}/kheops_health_${SESSION_ID}.csv"
KHEOPS_PG="${LOG_DIR}/kheops_pacs_db_${SESSION_ID}.csv"

if [[ -f "$KHEOPS_FILE" ]]; then
    IFS='|' read -r _ rp_resp_max rp_resp_avg _ _ <<< "$(csv_stats "$KHEOPS_FILE" 3)"
    IFS='|' read -r _ api_resp_max api_resp_avg _ _ <<< "$(csv_stats "$KHEOPS_FILE" 11)"
    IFS='|' read -r _ pacs_resp_max pacs_resp_avg _ _ <<< "$(csv_stats "$KHEOPS_FILE" 10)"
    rp_down=$(csv_count_above "$KHEOPS_FILE" 2 0 | head -1)

    echo "| Endpoint | Avg (ms) | Max (ms) | Downtime Samples |" >> "$REPORT"
    echo "|----------|----------|----------|------------------|" >> "$REPORT"
    echo "| Reverse Proxy | ${rp_resp_avg} | ${rp_resp_max} | — |" >> "$REPORT"
    echo "| API /studies | ${api_resp_avg} | ${api_resp_max} | — |" >> "$REPORT"
    echo "| PACS DCM4CHEE | ${pacs_resp_avg} | ${pacs_resp_max} | — |" >> "$REPORT"
fi

if [[ -f "$KHEOPS_PG" ]]; then
    echo "" >> "$REPORT"
    echo "### Kheops Databases" >> "$REPORT"
    echo "" >> "$REPORT"
    IFS='|' read -r _ kh_conn_max kh_conn_avg _ _ <<< "$(csv_stats "$KHEOPS_PG" 2)"
    IFS='|' read -r _ pacs_conn_max pacs_conn_avg _ _ <<< "$(csv_stats "$KHEOPS_PG" 5)"
    IFS='|' read -r _ pacs_size_max _ _ _ <<< "$(csv_stats "$KHEOPS_PG" 7)"

    echo "| Database | Avg Active Conns | Max Active Conns | Size |" >> "$REPORT"
    echo "|----------|-----------------|-----------------|------|" >> "$REPORT"
    echo "| Kheops Auth PG | ${kh_conn_avg} | ${kh_conn_max} | — |" >> "$REPORT"
    echo "| PACS PG | ${pacs_conn_avg} | ${pacs_conn_max} | ${pacs_size_max} MB |" >> "$REPORT"
fi

kheops_errors="${LOG_DIR}/kheops_errors_${SESSION_ID}.log"
if [[ -f "$kheops_errors" && -s "$kheops_errors" ]]; then
    echo "" >> "$REPORT"
    echo "> **⚠ Finding:** Kheops errors detected. See \`kheops_errors_${SESSION_ID}.log\`." >> "$REPORT"
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 9. Disk I/O Summary" >> "$REPORT"
echo "" >> "$REPORT"

DISK_FILE="${LOG_DIR}/disk_io_${SESSION_ID}.csv"
if [[ -f "$DISK_FILE" ]]; then
    disks=$(tail -n +2 "$DISK_FILE" | cut -d',' -f2 | sort -u)
    echo "| Disk | Avg Reads/s | Max Reads/s | Avg Writes/s | Max Writes/s | Avg Util % | Max Util % |" >> "$REPORT"
    echo "|------|-------------|-------------|--------------|--------------|------------|------------|" >> "$REPORT"
    for disk in $disks; do
        disk_data=$(grep ",$disk," "$DISK_FILE")
        r_avg=$(echo "$disk_data" | cut -d',' -f3 | awk '{sum+=$1;n++} END {if(n) printf "%.1f",sum/n; else print 0}')
        r_max=$(echo "$disk_data" | cut -d',' -f3 | sort -n | tail -1)
        w_avg=$(echo "$disk_data" | cut -d',' -f4 | awk '{sum+=$1;n++} END {if(n) printf "%.1f",sum/n; else print 0}')
        w_max=$(echo "$disk_data" | cut -d',' -f4 | sort -n | tail -1)
        u_avg=$(echo "$disk_data" | cut -d',' -f7 | awk '{sum+=$1;n++} END {if(n) printf "%.1f",sum/n; else print 0}')
        u_max=$(echo "$disk_data" | cut -d',' -f7 | sort -n | tail -1)
        echo "| $disk | $r_avg | $r_max | $w_avg | $w_max | ${u_avg}% | ${u_max}% |" >> "$REPORT"
    done
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "## 10. Key Findings & Recommendations" >> "$REPORT"
echo "" >> "$REPORT"

findings=0

# Auto-detect bottlenecks
if [[ -f "$SYS_FILE" ]]; then
    iowait_max_val=$(tail -n +2 "$SYS_FILE" | cut -d',' -f4 | sort -n | tail -1)
    if (( $(echo "${iowait_max_val:-0} > 20" | bc -l 2>/dev/null || echo 0) )); then
        echo "### Disk I/O Bottleneck" >> "$REPORT"
        echo "High I/O wait (${iowait_max_val}%) detected. This commonly happens during DICOM extraction when studies are downloaded and unzipped. Consider:" >> "$REPORT"
        echo "- Using SSD/NVMe for \`/quantimage2-data\` volume" >> "$REPORT"
        echo "- Reducing extraction concurrency from 4 to 2" >> "$REPORT"
        echo "- Pre-caching frequently used studies" >> "$REPORT"
        echo "" >> "$REPORT"
        findings=$((findings + 1))
    fi
fi

if [[ -f "$MYSQL_FILE" ]]; then
    threads_max_val=$(tail -n +2 "$MYSQL_FILE" | cut -d',' -f3 | sort -n | tail -1)
    if (( $(echo "${threads_max_val:-0} > 10" | bc -l 2>/dev/null || echo 0) )); then
        echo "### Database Contention" >> "$REPORT"
        echo "High running thread count (${threads_max_val}) indicates query contention. Consider:" >> "$REPORT"
        echo "- Reviewing slow query log for optimization opportunities" >> "$REPORT"
        echo "- Increasing \`innodb_buffer_pool_size\`" >> "$REPORT"
        echo "- Adding indexes on frequently queried columns (feature_value, feature_extraction)" >> "$REPORT"
        echo "" >> "$REPORT"
        findings=$((findings + 1))
    fi
fi

if [[ -f "$REDIS_Q" ]]; then
    ext_max_val=$(tail -n +2 "$REDIS_Q" | cut -d',' -f2 | sort -n | tail -1)
    if (( $(echo "${ext_max_val:-0} > 10" | bc -l 2>/dev/null || echo 0) )); then
        echo "### Extraction Queue Backlog" >> "$REPORT"
        echo "Extraction queue reached ${ext_max_val} pending tasks. Consider:" >> "$REPORT"
        echo "- Increasing extraction worker concurrency" >> "$REPORT"
        echo "- Adding more extraction worker containers" >> "$REPORT"
        echo "" >> "$REPORT"
        findings=$((findings + 1))
    fi
fi

if [[ -f "$FLASK_FILE" ]]; then
    resp_max_val=$(tail -n +2 "$FLASK_FILE" | cut -d',' -f3 | sort -n | tail -1)
    if (( $(echo "${resp_max_val:-0} > 5000" | bc -l 2>/dev/null || echo 0) )); then
        echo "### Backend Response Time" >> "$REPORT"
        echo "Backend response times reached ${resp_max_val}ms. Possible causes:" >> "$REPORT"
        echo "- Eventlet thread starvation under concurrent requests" >> "$REPORT"
        echo "- Long-running DB queries blocking the event loop" >> "$REPORT"
        echo "- Consider gunicorn with multiple workers instead of single eventlet" >> "$REPORT"
        echo "" >> "$REPORT"
        findings=$((findings + 1))
    fi
fi

if [[ -f "$KHEOPS_FILE" ]]; then
    kheops_max_val=$(tail -n +2 "$KHEOPS_FILE" | cut -d',' -f11 | sort -n | tail -1)
    if (( $(echo "${kheops_max_val:-0} > 5000" | bc -l 2>/dev/null || echo 0) )); then
        echo "### Kheops Response Time" >> "$REPORT"
        echo "Kheops API response times reached ${kheops_max_val}ms. Consider:" >> "$REPORT"
        echo "- Checking DCM4CHEE PACS heap memory / JVM settings" >> "$REPORT"
        echo "- Reviewing PACS PostgreSQL slow queries" >> "$REPORT"
        echo "- Checking Kheops authorization service logs" >> "$REPORT"
        echo "" >> "$REPORT"
        findings=$((findings + 1))
    fi
fi

if [[ "$findings" -eq 0 ]]; then
    echo "No major bottlenecks detected during this monitoring session." >> "$REPORT"
    echo "" >> "$REPORT"
    echo "**General recommendations for multi-user sessions:**" >> "$REPORT"
    echo "- Monitor this report during peak usage for emerging patterns" >> "$REPORT"
    echo "- Consider increasing MySQL \`max_connections\` if usage grows" >> "$REPORT"
    echo "- Pre-warm the system by running a test extraction before users arrive" >> "$REPORT"
fi

# ============================================================================
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Log Files" >> "$REPORT"
echo "" >> "$REPORT"
echo "All raw data logs for this session:" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
ls -lh "${LOG_DIR}"/*"${SESSION_ID}"* 2>/dev/null | awk '{print $5, $9}' | sed "s|${LOG_DIR}/||" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"

echo "" >> "$REPORT"
echo "*Report generated by QuantImage v2 Monitoring Suite*" >> "$REPORT"

log_ok "REPORT" "Report generated: $REPORT"
echo ""
echo "View the report:"
echo "  cat $REPORT"
echo "  # or open in browser/VS Code for Markdown rendering"

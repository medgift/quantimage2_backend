#!/usr/bin/env bash
# ============================================================================
# Module 1: Docker Container Resource Monitor
# Tracks CPU%, Memory%, Memory usage, Net I/O, Block I/O for all containers
# Output: CSV log for graphing + real-time console alerts
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="DOCKER"
LOGFILE="${LOG_DIR}/docker_resources_${SESSION_ID}.csv"

# CSV header
echo "timestamp,container,cpu_pct,mem_pct,mem_usage_mb,mem_limit_mb,net_rx_mb,net_tx_mb,block_read_mb,block_write_mb,pids" > "$LOGFILE"

log_header "$MODULE" "Starting container resource monitor → $LOGFILE"
log_header "$MODULE" "Monitoring ${#ALL_CONTAINERS[@]} containers every ${SAMPLE_INTERVAL}s"

parse_size_to_mb() {
    local val="$1"
    # Handle formats like: 123.4MiB, 1.23GiB, 456KiB, 123B
    local num unit
    num=$(echo "$val" | grep -oP '[\d.]+' | head -1)
    unit=$(echo "$val" | grep -oP '[A-Za-z]+' | head -1)
    case "$unit" in
        GiB|GB) echo "$num * 1024" | bc 2>/dev/null || echo "0" ;;
        MiB|MB) echo "$num" ;;
        KiB|KB|kB) echo "scale=2; $num / 1024" | bc 2>/dev/null || echo "0" ;;
        B)   echo "scale=4; $num / 1048576" | bc 2>/dev/null || echo "0" ;;
        *)   echo "0" ;;
    esac
}

while true; do
    ts=$(csv_ts)

    # Use docker stats --no-stream for a single snapshot (lightweight)
    # Filter to only our containers to reduce overhead
    container_filter=""
    for c in "${ALL_CONTAINERS[@]}"; do
        container_filter+="$c "
    done

    # Capture stats in one call (much cheaper than per-container calls)
    stats_output=$(docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}|{{.PIDs}}' $container_filter 2>/dev/null || true)

    while IFS='|' read -r name cpu_pct mem_pct mem_usage net_io block_io pids; do
        [[ -z "$name" ]] && continue

        # Parse values
        cpu=$(echo "$cpu_pct" | tr -d '%')
        mem=$(echo "$mem_pct" | tr -d '%')

        # Memory: "123.4MiB / 1.5GiB"
        mem_used_raw=$(echo "$mem_usage" | cut -d'/' -f1 | xargs)
        mem_limit_raw=$(echo "$mem_usage" | cut -d'/' -f2 | xargs)
        mem_used_mb=$(parse_size_to_mb "$mem_used_raw")
        mem_limit_mb=$(parse_size_to_mb "$mem_limit_raw")

        # Network: "123MB / 456MB"
        net_rx_raw=$(echo "$net_io" | cut -d'/' -f1 | xargs)
        net_tx_raw=$(echo "$net_io" | cut -d'/' -f2 | xargs)
        net_rx_mb=$(parse_size_to_mb "$net_rx_raw")
        net_tx_mb=$(parse_size_to_mb "$net_tx_raw")

        # Block I/O: "123MB / 456MB"
        block_r_raw=$(echo "$block_io" | cut -d'/' -f1 | xargs)
        block_w_raw=$(echo "$block_io" | cut -d'/' -f2 | xargs)
        block_r_mb=$(parse_size_to_mb "$block_r_raw")
        block_w_mb=$(parse_size_to_mb "$block_w_raw")

        pids_clean=$(echo "$pids" | tr -d ' ')

        echo "$ts,$name,$cpu,$mem,$mem_used_mb,$mem_limit_mb,$net_rx_mb,$net_tx_mb,$block_r_mb,$block_w_mb,$pids_clean" >> "$LOGFILE"

        # Alerts for high resource usage
        if (( $(echo "$cpu > 80" | bc -l 2>/dev/null || echo 0) )); then
            log_warn "$MODULE" "HIGH CPU: $name at ${cpu}%"
        fi
        if (( $(echo "$mem > 80" | bc -l 2>/dev/null || echo 0) )); then
            log_warn "$MODULE" "HIGH MEMORY: $name at ${mem}% (${mem_used_mb}MB)"
        fi
    done <<< "$stats_output"

    sleep "$SAMPLE_INTERVAL"
done

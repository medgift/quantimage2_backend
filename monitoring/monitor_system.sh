#!/usr/bin/env bash
# ============================================================================
# Module 7: Host System & Self-Resource Monitor
# Tracks host-level CPU, memory, disk I/O, network, and the resource
# consumption of these monitoring scripts themselves
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODULE="SYSTEM"
LOGFILE="${LOG_DIR}/system_stats_${SESSION_ID}.csv"
LOGFILE_SELF="${LOG_DIR}/self_resource_${SESSION_ID}.csv"
LOGFILE_DISK="${LOG_DIR}/disk_io_${SESSION_ID}.csv"
LOGFILE_NET="${LOG_DIR}/network_${SESSION_ID}.csv"

# CSV headers
echo "timestamp,cpu_user_pct,cpu_sys_pct,cpu_iowait_pct,cpu_idle_pct,load_1m,load_5m,load_15m,mem_total_mb,mem_used_mb,mem_available_mb,mem_used_pct,swap_used_mb,swap_total_mb,procs_running,procs_blocked,context_switches,interrupts" > "$LOGFILE"
echo "timestamp,monitor_pid_count,monitor_cpu_pct,monitor_mem_mb,monitor_mem_pct,log_dir_size_mb" > "$LOGFILE_SELF"
echo "timestamp,disk,reads_per_sec,writes_per_sec,read_kb_per_sec,write_kb_per_sec,io_util_pct" > "$LOGFILE_DISK"
echo "timestamp,iface,rx_bytes_sec,tx_bytes_sec,rx_packets_sec,tx_packets_sec" > "$LOGFILE_NET"

log_header "$MODULE" "Starting system & self-resource monitor"
log_header "$MODULE" "Host: $(hostname), CPUs: $(nproc), Memory: $(free -h | awk '/Mem:/{print $2}')"

# Store PID of parent monitoring process
echo "$$" > "$MONITOR_PID_FILE"

prev_cpu_stats=""
prev_net_stats=""
prev_disk_stats=""

get_cpu_stats() {
    head -1 /proc/stat | awk '{print $2,$3,$4,$5,$6,$7,$8}'
}

get_net_stats() {
    # Returns interface rx_bytes tx_bytes rx_packets tx_packets for docker0 and eth0
    awk '/docker0|eth0|ens/ {gsub(/:/, "", $1); print $1, $2, $10, $3, $11}' /proc/net/dev
}

get_disk_stats() {
    # Major disk devices
    awk '$3 ~ /^(sd[a-z]|nvme[0-9]+n[0-9]+|vd[a-z])$/ {print $3, $4, $8, $6, $10, $13}' /proc/diskstats 2>/dev/null
}

# Initialize previous values
prev_cpu_stats=$(get_cpu_stats)
prev_net_stats=$(get_net_stats)
prev_disk_stats=$(get_disk_stats)
sleep 1  # Need a baseline

while true; do
    ts=$(csv_ts)

    # ── CPU utilization (from /proc/stat delta) ──
    curr_cpu=$(get_cpu_stats)
    if [[ -n "$prev_cpu_stats" && -n "$curr_cpu" ]]; then
        read -r p_user p_nice p_sys p_idle p_iowait p_irq p_softirq <<< "$prev_cpu_stats"
        read -r c_user c_nice c_sys c_idle c_iowait c_irq c_softirq <<< "$curr_cpu"

        d_user=$((c_user - p_user + c_nice - p_nice))
        d_sys=$((c_sys - p_sys + c_irq - p_irq + c_softirq - p_softirq))
        d_idle=$((c_idle - p_idle))
        d_iowait=$((c_iowait - p_iowait))
        d_total=$((d_user + d_sys + d_idle + d_iowait))

        if [[ $d_total -gt 0 ]]; then
            cpu_user=$(echo "scale=1; $d_user * 100 / $d_total" | bc)
            cpu_sys=$(echo "scale=1; $d_sys * 100 / $d_total" | bc)
            cpu_iowait=$(echo "scale=1; $d_iowait * 100 / $d_total" | bc)
            cpu_idle=$(echo "scale=1; $d_idle * 100 / $d_total" | bc)
        else
            cpu_user=0; cpu_sys=0; cpu_iowait=0; cpu_idle=100
        fi
    else
        cpu_user=0; cpu_sys=0; cpu_iowait=0; cpu_idle=100
    fi
    prev_cpu_stats="$curr_cpu"

    # Load averages
    read -r load1 load5 load15 _ < /proc/loadavg

    # Memory
    mem_info=$(awk '
        /MemTotal:/     {total=$2}
        /MemAvailable:/ {avail=$2}
        /SwapTotal:/    {swaptotal=$2}
        /SwapFree:/     {swapfree=$2}
        END {
            used = total - avail
            printf "%d %d %d %.1f %d %d\n",
                total/1024, used/1024, avail/1024,
                used*100/total,
                (swaptotal-swapfree)/1024, swaptotal/1024
        }
    ' /proc/meminfo)
    read -r mem_total mem_used mem_avail mem_pct swap_used swap_total <<< "$mem_info"

    # Processes
    procs_running=$(awk '/procs_running/ {print $2}' /proc/stat)
    procs_blocked=$(awk '/procs_blocked/ {print $2}' /proc/stat)

    # Context switches and interrupts
    ctxt=$(awk '/ctxt/ {print $2}' /proc/stat)
    intr=$(awk '/^intr/ {print $2}' /proc/stat)

    echo "$ts,$cpu_user,$cpu_sys,$cpu_iowait,$cpu_idle,$load1,$load5,$load15,$mem_total,$mem_used,$mem_avail,$mem_pct,$swap_used,$swap_total,$procs_running,$procs_blocked,$ctxt,$intr" >> "$LOGFILE"

    # Alerts
    if (( $(echo "$cpu_iowait > 20" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "High I/O wait: ${cpu_iowait}%"
    fi
    if (( $(echo "$mem_pct > 85" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "Memory usage: ${mem_pct}% (${mem_used}MB / ${mem_total}MB)"
    fi
    ncpus=$(nproc)
    if (( $(echo "$load1 > $ncpus * 2" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "High load average: $load1 / $load5 / $load15 (${ncpus} CPUs)"
    fi

    # ── Disk I/O ──
    curr_disk=$(get_disk_stats)
    if [[ -n "$prev_disk_stats" && -n "$curr_disk" ]]; then
        while IFS=' ' read -r disk reads writes read_sectors write_sectors io_ticks; do
            prev_line=$(echo "$prev_disk_stats" | grep "^$disk " || echo "")
            if [[ -n "$prev_line" ]]; then
                read -r _ p_reads p_writes p_rsect p_wsect p_io <<< "$prev_line"
                d_reads=$(( (reads - p_reads) ))
                d_writes=$(( (writes - p_writes) ))
                d_rsect=$(( (read_sectors - p_rsect) / 2 ))   # sectors → KB
                d_wsect=$(( (write_sectors - p_wsect) / 2 ))
                # IO utilization: ticks are in ms, interval is SAMPLE_INTERVAL seconds
                d_io=$(( io_ticks - p_io ))
                io_util=$(echo "scale=1; $d_io / (${SAMPLE_INTERVAL} * 10)" | bc 2>/dev/null || echo "0")
                [[ $(echo "$io_util > 100" | bc 2>/dev/null) == "1" ]] && io_util="100.0"

                r_per_s=$(echo "scale=1; $d_reads / ${SAMPLE_INTERVAL}" | bc 2>/dev/null || echo "0")
                w_per_s=$(echo "scale=1; $d_writes / ${SAMPLE_INTERVAL}" | bc 2>/dev/null || echo "0")
                rkb_per_s=$(echo "scale=1; $d_rsect / ${SAMPLE_INTERVAL}" | bc 2>/dev/null || echo "0")
                wkb_per_s=$(echo "scale=1; $d_wsect / ${SAMPLE_INTERVAL}" | bc 2>/dev/null || echo "0")

                echo "$ts,$disk,$r_per_s,$w_per_s,$rkb_per_s,$wkb_per_s,$io_util" >> "$LOGFILE_DISK"

                if (( $(echo "$io_util > 80" | bc -l 2>/dev/null || echo 0) )); then
                    log_warn "$MODULE" "Disk $disk I/O utilization: ${io_util}%"
                fi
            fi
        done <<< "$curr_disk"
    fi
    prev_disk_stats="$curr_disk"

    # ── Network I/O ──
    curr_net=$(get_net_stats)
    if [[ -n "$prev_net_stats" && -n "$curr_net" ]]; then
        while IFS=' ' read -r iface rx_bytes tx_bytes rx_pkt tx_pkt; do
            prev_line=$(echo "$prev_net_stats" | grep "^$iface " || echo "")
            if [[ -n "$prev_line" ]]; then
                read -r _ p_rx p_tx p_rxpkt p_txpkt <<< "$prev_line"
                rx_sec=$(( (rx_bytes - p_rx) / SAMPLE_INTERVAL ))
                tx_sec=$(( (tx_bytes - p_tx) / SAMPLE_INTERVAL ))
                rxpkt_sec=$(( (rx_pkt - p_rxpkt) / SAMPLE_INTERVAL ))
                txpkt_sec=$(( (tx_pkt - p_txpkt) / SAMPLE_INTERVAL ))
                echo "$ts,$iface,$rx_sec,$tx_sec,$rxpkt_sec,$txpkt_sec" >> "$LOGFILE_NET"
            fi
        done <<< "$curr_net"
    fi
    prev_net_stats="$curr_net"

    # ── Self-resource monitoring ──
    # Find all monitoring script PIDs (children of orchestrator or matching our script names)
    monitor_pids=$(pgrep -f "monitoring/monitor_" 2>/dev/null | tr '\n' ',' || echo "")
    monitor_pid_count=$(echo "$monitor_pids" | tr ',' '\n' | grep -c '[0-9]' || echo "0")

    if [[ -n "$monitor_pids" && "$monitor_pids" != "," ]]; then
        # Total CPU and memory of all monitoring processes
        monitor_cpu=$(ps -p "$(echo "$monitor_pids" | sed 's/,$//')" -o %cpu= 2>/dev/null | awk '{sum+=$1} END {printf "%.1f", sum}' || echo "0")
        monitor_mem=$(ps -p "$(echo "$monitor_pids" | sed 's/,$//')" -o rss= 2>/dev/null | awk '{sum+=$1} END {printf "%.1f", sum/1024}' || echo "0")
        monitor_mem_pct=$(echo "scale=2; ${monitor_mem:-0} * 100 / ${mem_total:-1}" | bc 2>/dev/null || echo "0")
    else
        monitor_cpu="0"
        monitor_mem="0"
        monitor_mem_pct="0"
    fi

    # Log directory size
    log_size_mb=$(du -sm "$LOG_DIR" 2>/dev/null | awk '{print $1}' || echo "0")

    echo "$ts,$monitor_pid_count,$monitor_cpu,$monitor_mem,$monitor_mem_pct,$log_size_mb" >> "$LOGFILE_SELF"

    if (( $(echo "$monitor_cpu > 5" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "Monitoring scripts using ${monitor_cpu}% CPU"
    fi
    if (( $(echo "$monitor_mem > 200" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "$MODULE" "Monitoring scripts using ${monitor_mem}MB RAM"
    fi
    if [[ "${log_size_mb:-0}" -gt 500 ]]; then
        log_warn "$MODULE" "Log directory is ${log_size_mb}MB (consider cleanup)"
    fi

    sleep "$SAMPLE_INTERVAL"
done

#!/usr/bin/env bash
# ============================================================================
# QuantImage v2 — Live Performance Monitoring Suite
# Master Orchestrator
#
# Launches all monitoring modules as background processes, displays a
# unified live dashboard, and provides clean shutdown with report generation.
#
# Usage:
#   ./start_monitoring.sh              # Start all monitors (default 10s interval)
#   SAMPLE_INTERVAL=30 ./start_monitoring.sh   # Custom interval
#   ./start_monitoring.sh --stop       # Stop all monitors
#   ./start_monitoring.sh --status     # Check running monitors
#   ./start_monitoring.sh --report     # Generate report from latest session
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export session ID so all modules share it
export SESSION_ID="${SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
export SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

source "${SCRIPT_DIR}/config.sh"

PID_FILE="${LOG_DIR}/orchestrator_${SESSION_ID}.pids"
DASHBOARD_INTERVAL=30  # Dashboard refresh interval (seconds)

# ============================================================================
# Command handlers
# ============================================================================

stop_monitoring() {
    echo -e "${BOLD}${RED}Stopping all monitoring processes...${NC}"

    # Find all PID files
    for pf in "${LOG_DIR}"/orchestrator_*.pids; do
        [[ -f "$pf" ]] || continue
        while IFS='|' read -r pid name; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && echo "  Stopped $name (PID $pid)" || true
                # Also kill child processes
                pkill -P "$pid" 2>/dev/null || true
            fi
        done < "$pf"
        rm -f "$pf"
    done

    # Cleanup any remaining monitor processes
    pkill -f "monitoring/monitor_" 2>/dev/null || true

    echo -e "${GREEN}All monitors stopped.${NC}"

    # Auto-generate report
    local latest_session
    latest_session=$(ls -1 "$LOG_DIR"/docker_resources_*.csv 2>/dev/null | sort | tail -1 | grep -oP '\d{8}_\d{6}' || echo "")
    if [[ -n "$latest_session" ]]; then
        echo ""
        echo -e "${BOLD}Generating final report...${NC}"
        bash "${SCRIPT_DIR}/generate_report.sh" "$latest_session"
    fi
}

show_status() {
    echo -e "${BOLD}${CYAN}=== Monitoring Status ===${NC}"
    echo ""

    local running=0
    local total=0

    for pf in "${LOG_DIR}"/orchestrator_*.pids; do
        [[ -f "$pf" ]] || continue
        local session
        session=$(echo "$pf" | grep -oP '\d{8}_\d{6}')
        echo -e "${BOLD}Session: $session${NC}"

        while IFS='|' read -r pid name; do
            total=$((total + 1))
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${GREEN}● $name${NC} (PID $pid)"
                running=$((running + 1))
            else
                echo -e "  ${RED}○ $name${NC} (PID $pid — stopped)"
            fi
        done < "$pf"
        echo ""
    done

    if [[ $total -eq 0 ]]; then
        echo "No monitoring sessions found."
    else
        echo "$running / $total monitors running"
    fi

    # Show log sizes
    echo ""
    echo -e "${BOLD}Log directory:${NC}"
    du -sh "$LOG_DIR" 2>/dev/null || echo "  (empty)"
    echo ""
    ls -lhS "$LOG_DIR"/*.csv 2>/dev/null | head -10 | awk '{print "  " $5 "  " $9}' | sed "s|${LOG_DIR}/||" || true
}

generate_report_cmd() {
    local session="${1:-}"
    if [[ -z "$session" ]]; then
        session=$(ls -1 "$LOG_DIR"/docker_resources_*.csv 2>/dev/null | sort | tail -1 | grep -oP '\d{8}_\d{6}' || echo "")
    fi
    if [[ -z "$session" ]]; then
        echo "No sessions found."
        exit 1
    fi
    bash "${SCRIPT_DIR}/generate_report.sh" "$session"
}

# ============================================================================
# Handle arguments
# ============================================================================

case "${1:-start}" in
    --stop|-s)
        stop_monitoring
        exit 0
        ;;
    --status|-t)
        show_status
        exit 0
        ;;
    --report|-r)
        generate_report_cmd "${2:-}"
        exit 0
        ;;
    --help|-h)
        cat << 'HELP'
QuantImage v2 — Live Performance Monitoring Suite

Usage:
  ./start_monitoring.sh              Start all monitors
  ./start_monitoring.sh --stop       Stop all monitors & generate report
  ./start_monitoring.sh --status     Show running monitor status
  ./start_monitoring.sh --report     Generate report from latest session

Environment variables:
  SAMPLE_INTERVAL=10    Sampling interval in seconds (default: 10)
  SESSION_ID=...        Override session ID (auto-generated if not set)

Monitors launched:
  1. Docker container resources (CPU, memory, net/block I/O)
  2. MySQL database (connections, slow queries, locks, buffer pool)
  3. Redis (memory, ops, queue depths, task counts)
  4. Celery workers (health, active tasks, errors)
  5. Flask backend (response time, errors, file descriptors)
  6. Kheops/PACS (endpoint health, DB connections, response times)
  7. Host system (CPU, memory, disk I/O, network, self-monitoring)

Output:
  logs/      — CSV time-series data + error logs per session
  reports/   — Markdown analysis reports
HELP
        exit 0
        ;;
    start|"")
        ;;  # Continue to main start logic
    *)
        echo "Unknown command: $1. Use --help for usage."
        exit 1
        ;;
esac

# ============================================================================
# Pre-flight checks
# ============================================================================

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   QuantImage v2 — Live Performance Monitoring Suite     ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Session:  ${BOLD}${SESSION_ID}${NC}"
echo -e "  Interval: ${BOLD}${SAMPLE_INTERVAL}s${NC}"
echo -e "  Logs:     ${BOLD}${LOG_DIR}/${NC}"
echo ""

# Check prerequisites
echo -e "${BOLD}Pre-flight checks:${NC}"

check_ok=true
for cmd in docker curl bc jq awk; do
    if command -v "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $cmd"
    else
        echo -e "  ${RED}✗${NC} $cmd (missing!)"
        check_ok=false
    fi
done

# Check Docker daemon
if docker info &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Docker daemon"
else
    echo -e "  ${RED}✗${NC} Docker daemon (not accessible)"
    check_ok=false
fi

# Check key containers
echo ""
echo -e "${BOLD}Container status:${NC}"
for container in "$QI2_BACKEND" "$QI2_DB" "$QI2_REDIS" "$QI2_CELERY_EXTRACTION" "$KHEOPS_REVERSE_PROXY" "$PACS_ARC"; do
    if container_running "$container"; then
        echo -e "  ${GREEN}●${NC} $container"
    else
        echo -e "  ${RED}○${NC} $container (not running)"
    fi
done

# Check MySQL connectivity
echo ""
echo -e "${BOLD}Service connectivity:${NC}"
if mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "SELECT 1" &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} MySQL ($MYSQL_HOST:$MYSQL_PORT)"
else
    if docker exec "$QI2_DB" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "SELECT 1" &>/dev/null; then
        echo -e "  ${YELLOW}~${NC} MySQL (via Docker exec only)"
    else
        echo -e "  ${RED}✗${NC} MySQL"
    fi
fi

# Check Redis
if docker exec "$QI2_REDIS" redis-cli PING 2>/dev/null | grep -q PONG; then
    echo -e "  ${GREEN}✓${NC} Redis (Celery)"
else
    echo -e "  ${RED}✗${NC} Redis (Celery)"
fi

if docker exec "$QI2_REDIS_SOCKET" redis-cli PING 2>/dev/null | grep -q PONG; then
    echo -e "  ${GREEN}✓${NC} Redis (Socket.IO)"
else
    echo -e "  ${RED}✗${NC} Redis (Socket.IO)"
fi

# Check Kheops
kheops_status=$(curl -sk -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 5 "$KHEOPS_URL" 2>/dev/null || echo "000")
if [[ "$kheops_status" =~ ^[2345] ]]; then
    echo -e "  ${GREEN}✓${NC} Kheops ($KHEOPS_URL) → HTTP $kheops_status"
else
    echo -e "  ${RED}✗${NC} Kheops ($KHEOPS_URL) → HTTP $kheops_status"
fi

echo ""

if [[ "$check_ok" != "true" ]]; then
    echo -e "${YELLOW}Some prerequisites are missing. Monitoring may be incomplete.${NC}"
fi

# ============================================================================
# Launch monitors
# ============================================================================

echo -e "${BOLD}Launching monitors...${NC}"
echo ""

> "$PID_FILE"

launch_module() {
    local script="$1"
    local name="$2"

    chmod +x "$script"
    bash "$script" >> "${LOG_DIR}/console_${name}_${SESSION_ID}.log" 2>&1 &
    local pid=$!
    echo "${pid}|${name}" >> "$PID_FILE"
    echo -e "  ${GREEN}▶${NC} ${name} (PID ${pid})"
}

launch_module "${SCRIPT_DIR}/monitor_docker.sh"  "docker-resources"
launch_module "${SCRIPT_DIR}/monitor_mysql.sh"    "mysql"
launch_module "${SCRIPT_DIR}/monitor_redis.sh"    "redis"
launch_module "${SCRIPT_DIR}/monitor_celery.sh"   "celery"
launch_module "${SCRIPT_DIR}/monitor_flask.sh"    "flask-api"
launch_module "${SCRIPT_DIR}/monitor_kheops.sh"   "kheops"
launch_module "${SCRIPT_DIR}/monitor_system.sh"   "system"

echo ""
echo -e "${GREEN}${BOLD}All 7 monitors launched!${NC}"
echo ""
echo -e "Commands:"
echo -e "  ${BOLD}./start_monitoring.sh --status${NC}   Check monitor status"
echo -e "  ${BOLD}./start_monitoring.sh --stop${NC}     Stop all & generate report"
echo -e "  ${BOLD}./start_monitoring.sh --report${NC}   Generate report now"
echo -e "  ${BOLD}tail -f ${LOG_DIR}/console_*_${SESSION_ID}.log${NC}  Watch live alerts"
echo ""

# ============================================================================
# Live dashboard (runs in foreground, Ctrl+C stops dashboard only)
# ============================================================================

echo -e "${BOLD}Starting live dashboard (Ctrl+C to detach, monitors keep running)${NC}"
echo -e "${YELLOW}Use './start_monitoring.sh --stop' to stop all monitors${NC}"
echo ""

trap 'echo ""; echo -e "${YELLOW}Dashboard detached. Monitors still running.${NC}"; echo "Use: ./start_monitoring.sh --stop"; exit 0' INT

while true; do
    sleep "$DASHBOARD_INTERVAL"

    clear 2>/dev/null || true
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║   QuantImage v2 — Live Dashboard   Session: ${SESSION_ID}  ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  $(date '+%Y-%m-%d %H:%M:%S') | Interval: ${SAMPLE_INTERVAL}s | Dashboard refresh: ${DASHBOARD_INTERVAL}s"
    echo ""

    # Quick snapshot of all monitors
    running_count=0
    total_count=0
    while IFS='|' read -r pid name; do
        total_count=$((total_count + 1))
        if kill -0 "$pid" 2>/dev/null; then
            running_count=$((running_count + 1))
        fi
    done < "$PID_FILE"
    echo -e "  Monitors: ${GREEN}${running_count}${NC}/${total_count} running"

    # System quick stats
    read -r load1 load5 load15 _ < /proc/loadavg
    mem_pct=$(awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} END{printf "%.0f", (t-a)*100/t}' /proc/meminfo)
    echo -e "  Host: Load ${load1}/${load5}/${load15} | Memory: ${mem_pct}%"

    # Log sizes
    log_total=$(du -sm "$LOG_DIR" 2>/dev/null | awk '{print $1}')
    echo -e "  Logs: ${log_total}MB total"
    echo ""

    # Latest data from key CSVs
    echo -e "  ${BOLD}--- Top Container CPU (last sample) ---${NC}"
    if [[ -f "${LOG_DIR}/docker_resources_${SESSION_ID}.csv" ]]; then
        # Get last sample for each container, sort by CPU
        tail -20 "${LOG_DIR}/docker_resources_${SESSION_ID}.csv" | \
            awk -F',' '{if(NR>1) data[$2]=$0} END {for(c in data) print data[c]}' | \
            sort -t',' -k3 -rn | head -5 | \
            awk -F',' '{printf "    %-35s CPU: %6s%%  Mem: %6s%%  (%sMB)\n", $2, $3, $4, $5}'
    fi

    echo ""
    echo -e "  ${BOLD}--- Queue Depths (last sample) ---${NC}"
    if [[ -f "${LOG_DIR}/redis_queues_${SESSION_ID}.csv" ]]; then
        last_q=$(tail -1 "${LOG_DIR}/redis_queues_${SESSION_ID}.csv")
        ext_q=$(echo "$last_q" | cut -d',' -f2)
        train_q=$(echo "$last_q" | cut -d',' -f3)
        echo -e "    Extraction: ${ext_q:-0}  |  Training: ${train_q:-0}"
    fi

    echo ""
    echo -e "  ${BOLD}--- Recent Alerts ---${NC}"
    # Show last 5 alerts from console logs
    for logfile in "${LOG_DIR}"/console_*_"${SESSION_ID}".log; do
        [[ -f "$logfile" ]] || continue
        grep -E "⚠|✗|HIGH|WARN|ERROR" "$logfile" 2>/dev/null | tail -3
    done | sort | tail -8 | sed 's/^/    /'

    echo ""
    echo -e "  ${YELLOW}Ctrl+C to detach dashboard (monitors keep running)${NC}"
done

#!/usr/bin/env bash
# ============================================================================
# Quick Snapshot — One-off system health check
# Run this anytime to get an instant overview of everything without starting
# the full monitoring suite. Not resource-intensive — runs once and exits.
#
# Usage: ./quick_snapshot.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   QuantImage v2 + Kheops — Quick Health Snapshot            ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "  $(date '+%Y-%m-%d %H:%M:%S') | $(hostname)"
echo ""

# ============================================================================
echo -e "${BOLD}══ 1. HOST SYSTEM ══${NC}"
echo ""
read -r load1 load5 load15 procs _ < /proc/loadavg
ncpus=$(nproc)
mem_info=$(free -m | awk '/Mem:/{printf "%dMB / %dMB (%.0f%%)", $3, $2, $3*100/$2}')
swap_info=$(free -m | awk '/Swap:/{if($2>0) printf "%dMB / %dMB (%.0f%%)", $3, $2, $3*100/$2; else print "none"}')
echo "  CPUs:        $ncpus"
echo "  Load:        $load1 / $load5 / $load15  (1m/5m/15m)"
echo "  Memory:      $mem_info"
echo "  Swap:        $swap_info"
echo "  Processes:   $procs"

# Disk usage for key mounts
echo "  Disk:"
df -h / /var/lib/docker 2>/dev/null | awk 'NR>1 {printf "    %-20s %s used of %s (%s)\n", $6, $3, $2, $5}' || true

echo ""

# ============================================================================
echo -e "${BOLD}══ 2. DOCKER CONTAINERS ══${NC}"
echo ""
printf "  %-38s %-10s %8s %8s\n" "CONTAINER" "STATUS" "CPU%" "MEM%"
printf "  %-38s %-10s %8s %8s\n" "$(printf '─%.0s' {1..38})" "$(printf '─%.0s' {1..10})" "$(printf '─%.0s' {1..8})" "$(printf '─%.0s' {1..8})"

# Get stats in one call
for container in "${ALL_CONTAINERS[@]}"; do
    if container_running "$container"; then
        stats=$(docker stats --no-stream --format '{{.CPUPerc}}|{{.MemPerc}}' "$container" 2>/dev/null || echo "?|?")
        cpu=$(echo "$stats" | cut -d'|' -f1)
        mem=$(echo "$stats" | cut -d'|' -f2)
        printf "  ${GREEN}●${NC} %-36s %-10s %8s %8s\n" "$container" "running" "$cpu" "$mem"
    else
        printf "  ${RED}○${NC} %-36s %-10s %8s %8s\n" "$container" "STOPPED" "—" "—"
    fi
done

echo ""

# ============================================================================
echo -e "${BOLD}══ 3. MYSQL DATABASE ══${NC}"
echo ""

mysql_q() {
    docker exec "$QI2_DB" mysql -u root -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" \
        --connect-timeout=3 -N -B -e "$1" 2>/dev/null
}

# Try Docker exec with root first, fall back to regular user
if ! mysql_q "SELECT 1" &>/dev/null; then
    mysql_q() {
        docker exec "$QI2_DB" mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" \
            --connect-timeout=3 -N -B -e "$1" 2>/dev/null
    }
fi

if mysql_q "SELECT 1" &>/dev/null; then
    threads_conn=$(mysql_q "SHOW GLOBAL STATUS WHERE Variable_name='Threads_connected'" | awk '{print $2}' || echo "?")
    threads_run=$(mysql_q "SHOW GLOBAL STATUS WHERE Variable_name='Threads_running'" | awk '{print $2}' || echo "?")
    max_conn=$(mysql_q "SELECT @@max_connections" || echo "?")
    slow_q=$(mysql_q "SHOW GLOBAL STATUS WHERE Variable_name='Slow_queries'" | awk '{print $2}' || echo "?")
    uptime=$(mysql_q "SHOW GLOBAL STATUS WHERE Variable_name='Uptime'" | awk '{print $2}' || echo "?")
    questions=$(mysql_q "SHOW GLOBAL STATUS WHERE Variable_name='Questions'" | awk '{print $2}' || echo "?")

    echo "  Connections: ${threads_conn}/${max_conn} (${threads_run} running)"
    echo "  Slow queries: ${slow_q} (total since restart)"
    echo "  Questions:    ${questions}"
    echo "  Uptime:       ${uptime}s"

    # Table sizes
    echo ""
    echo "  Top tables by size:"
    mysql_q "
        SELECT TABLE_NAME,
               ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 1) as size_mb,
               TABLE_ROWS
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA='$MYSQL_DATABASE'
        ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC
        LIMIT 8
    " 2>/dev/null | while IFS=$'\t' read -r tbl size rows; do
        printf "    %-35s %8s MB  (%s rows)\n" "$tbl" "$size" "$rows"
    done

    # Active queries
    active=$(mysql_q "SELECT COUNT(*) FROM information_schema.PROCESSLIST WHERE COMMAND != 'Sleep' AND USER != 'system user'" || echo "0")
    if [[ "${active:-0}" -gt 1 ]]; then
        echo ""
        echo "  Active queries ($active):"
        mysql_q "
            SELECT CONCAT('    [', TIME, 's] ', LEFT(INFO, 100))
            FROM information_schema.PROCESSLIST
            WHERE COMMAND != 'Sleep' AND USER != 'system user' AND INFO IS NOT NULL
            ORDER BY TIME DESC LIMIT 5
        " || true
    fi

    # Lock contention
    lock_waits=$(mysql_q "SELECT COUNT(*) FROM information_schema.INNODB_TRX WHERE trx_state = 'LOCK WAIT'" || echo "0")
    if [[ "${lock_waits:-0}" -gt 0 ]]; then
        echo -e "  ${YELLOW}⚠ ${lock_waits} transactions waiting for locks!${NC}"
    fi
else
    echo -e "  ${RED}Cannot connect to MySQL${NC}"
fi

echo ""

# ============================================================================
echo -e "${BOLD}══ 4. REDIS ══${NC}"
echo ""

for label_container in "Celery|$REDIS_CELERY_CONTAINER" "Socket.IO|$REDIS_SOCKET_CONTAINER"; do
    IFS='|' read -r label container <<< "$label_container"
    info=$(docker exec "$container" redis-cli INFO 2>/dev/null || echo "")
    if [[ -n "$info" ]]; then
        mem=$(echo "$info" | grep "^used_memory_human:" | cut -d: -f2 | tr -d $'\r')
        clients=$(echo "$info" | grep "^connected_clients:" | cut -d: -f2 | tr -d $'\r')
        ops=$(echo "$info" | grep "^instantaneous_ops_per_sec:" | cut -d: -f2 | tr -d $'\r')
        keys=$(echo "$info" | grep "^db0:" | grep -oP 'keys=\d+' | cut -d= -f2 || echo "0")
        echo "  ${label}:"
        echo "    Memory: ${mem} | Clients: ${clients} | Ops/s: ${ops} | Keys: ${keys}"
    else
        echo -e "  ${label}: ${RED}unreachable${NC}"
    fi
done

# Queue depths
ext_q=$(docker exec "$REDIS_CELERY_CONTAINER" redis-cli LLEN extraction 2>/dev/null || echo "?")
train_q=$(docker exec "$REDIS_CELERY_CONTAINER" redis-cli LLEN training 2>/dev/null || echo "?")
echo ""
echo "  Queue depths — Extraction: ${ext_q} | Training: ${train_q}"

echo ""

# ============================================================================
echo -e "${BOLD}══ 5. CELERY WORKERS ══${NC}"
echo ""

for wc in "$QI2_CELERY_EXTRACTION" "$QI2_CELERY_TRAINING"; do
    short=$(echo "$wc" | sed 's/quantimage2-//')
    if container_running "$wc"; then
        stats=$(docker stats --no-stream --format 'CPU:{{.CPUPerc}} Mem:{{.MemUsage}}' "$wc" 2>/dev/null || echo "?")
        echo -e "  ${GREEN}●${NC} ${short}: ${stats}"

        # Recent errors
        errors=$(docker logs --since "5m" "$wc" 2>&1 | grep -ciE "error|exception|traceback" || true)
        if [[ "$errors" -gt 0 ]]; then
            echo -e "    ${YELLOW}⚠ ${errors} errors in last 5 minutes${NC}"
        fi
    else
        echo -e "  ${RED}○${NC} ${short}: STOPPED"
    fi
done

echo ""

# ============================================================================
echo -e "${BOLD}══ 6. FLASK BACKEND ══${NC}"
echo ""

# Response time probe
for url in "$QI2_BACKEND_URL" "http://localhost:5000"; do
    result=$(curl -sk -o /dev/null -w "%{http_code}|%{time_total}" --connect-timeout 3 --max-time 10 "$url" 2>/dev/null || echo "000|0")
    status=$(echo "$result" | cut -d'|' -f1)
    time_ms=$(echo "$(echo "$result" | cut -d'|' -f2) * 1000" | bc 2>/dev/null || echo "0")

    if [[ "$status" =~ ^[23] ]]; then
        echo -e "  ${GREEN}●${NC} ${url} → HTTP ${status} (${time_ms}ms)"
    else
        echo -e "  ${RED}●${NC} ${url} → HTTP ${status} (${time_ms}ms)"
    fi
done

# Backend container logs health
be_errors=$(docker logs --since "5m" "$QI2_BACKEND" 2>&1 | grep -ciE "error|exception" || true)
be_warns=$(docker logs --since "5m" "$QI2_BACKEND" 2>&1 | grep -ciE "warning" || true)
echo "  Last 5min: ${be_errors} errors, ${be_warns} warnings"

# Open file descriptors
fds=$(docker exec "$QI2_BACKEND" sh -c 'ls /proc/1/fd 2>/dev/null | wc -l' 2>/dev/null || echo "?")
echo "  Open file descriptors: ${fds}"

echo ""

# ============================================================================
echo -e "${BOLD}══ 7. KHEOPS (PACS) ══${NC}"
echo ""

# Endpoint probes
for endpoint in "/" "/api/studies"; do
    result=$(curl -sk -o /dev/null -w "%{http_code}|%{time_total}" --connect-timeout 5 --max-time 15 "${KHEOPS_URL}${endpoint}" 2>/dev/null || echo "000|0")
    status=$(echo "$result" | cut -d'|' -f1)
    time_ms=$(echo "$(echo "$result" | cut -d'|' -f2) * 1000" | bc 2>/dev/null || echo "0")

    if [[ "$status" =~ ^[2345] ]]; then
        echo -e "  ${GREEN}●${NC} ${KHEOPS_URL}${endpoint} → HTTP ${status} (${time_ms}ms)"
    else
        echo -e "  ${RED}●${NC} ${KHEOPS_URL}${endpoint} → HTTP ${status} (${time_ms}ms)"
    fi
done

# Kheops DB connections
kh_conns=$(docker exec "$KHEOPS_POSTGRES" psql -U kheopsuser -d kheops -t -A -c "SELECT count(*) FROM pg_stat_activity" 2>/dev/null || echo "?")
pacs_conns=$(docker exec "$PACS_POSTGRES" psql -U kheops_pacs -d kheops_pacs -t -A -c "SELECT count(*) FROM pg_stat_activity" 2>/dev/null || echo "?")
echo "  Kheops PG connections: ${kh_conns} | PACS PG connections: ${pacs_conns}"

# PACS storage size
pacs_storage=$(docker exec "$PACS_ARC" sh -c 'du -sh /storage 2>/dev/null | cut -f1' 2>/dev/null || echo "?")
echo "  PACS storage: ${pacs_storage}"

# Recent errors in Kheops containers
kh_errors=0
for container in "$KHEOPS_AUTHORIZATION" "$KHEOPS_DICOMWEB_PROXY" "$PACS_ARC"; do
    e=$(docker logs --since "5m" "$container" 2>&1 | grep -ciE "error|exception|severe" || true)
    kh_errors=$((kh_errors + e))
done
echo "  Kheops errors (last 5min): ${kh_errors}"

echo ""

# ============================================================================
echo -e "${BOLD}══ 8. NETWORK CONNECTIONS ══${NC}"
echo ""

# Count established connections to key ports
echo "  Established connections:"
echo "    Backend (5000):  $(ss -tn state established '( dport = :5000 or sport = :5000 )' 2>/dev/null | tail -n +2 | wc -l)"
echo "    MySQL (3306):    $(ss -tn state established '( dport = :3306 or sport = :3306 )' 2>/dev/null | tail -n +2 | wc -l)"
echo "    MySQL (3307):    $(ss -tn state established '( dport = :3307 or sport = :3307 )' 2>/dev/null | tail -n +2 | wc -l)"
echo "    Redis (6379):    $(ss -tn state established '( dport = :6379 or sport = :6379 )' 2>/dev/null | tail -n +2 | wc -l)"
echo "    Redis (6380):    $(ss -tn state established '( dport = :6380 or sport = :6380 )' 2>/dev/null | tail -n +2 | wc -l)"
echo "    HTTPS (443):     $(ss -tn state established '( dport = :443 or sport = :443 )' 2>/dev/null | tail -n +2 | wc -l)"
echo "    Flower (3333):   $(ss -tn state established '( dport = :3333 or sport = :3333 )' 2>/dev/null | tail -n +2 | wc -l)"

# TIME_WAIT connections (can indicate connection churn)
tw=$(ss -tn state time-wait 2>/dev/null | tail -n +2 | wc -l)
if [[ "$tw" -gt 100 ]]; then
    echo -e "  ${YELLOW}⚠ TIME_WAIT connections: ${tw} (connection churn)${NC}"
else
    echo "  TIME_WAIT: ${tw}"
fi

echo ""
echo -e "${BOLD}${GREEN}Snapshot complete!${NC}"
echo ""

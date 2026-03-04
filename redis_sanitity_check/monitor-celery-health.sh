#!/bin/bash
# ==============================================================================
# monitor-celery-health.sh
# Quick daily health check for Celery workers and Redis
# Usage: ./monitor-celery-health.sh
# ==============================================================================

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

REDIS_CONTAINER="quantimage2-redis-1"
EXTRACTION_CONTAINER="quantimage2-celery_extraction-1"
TRAINING_CONTAINER="quantimage2-celery_training-1"
STUCK_THRESHOLD_HOURS=2

print_section() {
    echo ""
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
}

print_ok()   { echo -e "  ${GREEN}[OK]${NC}    $1"; }
print_warn() { echo -e "  ${YELLOW}[WARN]${NC}  $1"; }
print_fail() { echo -e "  ${RED}[FAIL]${NC}  $1"; }
print_info() { echo -e "  ${CYAN}[INFO]${NC}  $1"; }

echo ""
echo -e "${BOLD}Celery / Redis Health Check${NC}  —  $(date '+%Y-%m-%d %H:%M:%S')"

# ------------------------------------------------------------------------------
# 1. Redis health
# ------------------------------------------------------------------------------
print_section "1. Redis Health"

if docker exec "$REDIS_CONTAINER" redis-cli PING 2>/dev/null | grep -q "PONG"; then
    print_ok "Redis is responding (PONG)"
else
    print_fail "Redis is NOT responding — check container $REDIS_CONTAINER"
fi

REDIS_WARNINGS=$(docker logs "$REDIS_CONTAINER" 2>&1 | grep -i "warning\|error" | tail -5)
if [ -n "$REDIS_WARNINGS" ]; then
    print_warn "Recent Redis warnings/errors:"
    echo "$REDIS_WARNINGS" | sed 's/^/          /'
else
    print_ok "No recent warnings or errors in Redis logs"
fi

# ------------------------------------------------------------------------------
# 2. Active workers
# ------------------------------------------------------------------------------
print_section "2. Active Workers"

for CONTAINER in "$EXTRACTION_CONTAINER" "$TRAINING_CONTAINER"; do
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER" 2>/dev/null)
        if [ "$STATUS" = "running" ]; then
            print_ok "$CONTAINER is running"
        else
            print_fail "$CONTAINER exists but status is: $STATUS"
        fi
    else
        print_warn "$CONTAINER is not running (container not found)"
    fi
done

# ------------------------------------------------------------------------------
# 3. Running task count
# ------------------------------------------------------------------------------
print_section "3. Running Tasks"

TASK_KEYS=$(docker exec "$REDIS_CONTAINER" redis-cli --scan --pattern "celery-task-meta-*" 2>/dev/null | wc -l)
print_info "Total task metadata keys in Redis: $TASK_KEYS"

STARTED_COUNT=0
PROGRESS_COUNT=0
PENDING_COUNT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

while IFS= read -r key; do
    VALUE=$(docker exec "$REDIS_CONTAINER" redis-cli GET "$key" 2>/dev/null)
    if echo "$VALUE" | grep -q '"status": "STARTED"'; then
        STARTED_COUNT=$((STARTED_COUNT + 1))
    elif echo "$VALUE" | grep -q '"status": "PROGRESS"'; then
        PROGRESS_COUNT=$((PROGRESS_COUNT + 1))
    elif echo "$VALUE" | grep -q '"status": "PENDING"'; then
        PENDING_COUNT=$((PENDING_COUNT + 1))
    elif echo "$VALUE" | grep -q '"status": "SUCCESS"'; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    elif echo "$VALUE" | grep -q '"status": "FAILURE"'; then
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
done < <(docker exec "$REDIS_CONTAINER" redis-cli --scan --pattern "celery-task-meta-*" 2>/dev/null | head -500)

print_info "STARTED  tasks : $STARTED_COUNT"
print_info "PROGRESS tasks : $PROGRESS_COUNT"
print_info "PENDING  tasks : $PENDING_COUNT"
print_info "SUCCESS  tasks : $SUCCESS_COUNT"
print_info "FAILURE  tasks : $FAILURE_COUNT"

if [ "$FAILURE_COUNT" -gt 0 ]; then
    print_warn "$FAILURE_COUNT task(s) in FAILURE state — consider investigating"
fi

# ------------------------------------------------------------------------------
# 4. Tasks stuck > threshold
# ------------------------------------------------------------------------------
print_section "4. Tasks Stuck > ${STUCK_THRESHOLD_HOURS} Hours"

NOW_TS=$(date +%s)
THRESHOLD_SECONDS=$((STUCK_THRESHOLD_HOURS * 3600))
STUCK_COUNT=0

while IFS= read -r key; do
    VALUE=$(docker exec "$REDIS_CONTAINER" redis-cli GET "$key" 2>/dev/null)
    if echo "$VALUE" | grep -qE '"status": "(STARTED|PROGRESS)"'; then
        # Extract date_done or use TTL as a fallback proxy
        DATE_DONE=$(echo "$VALUE" | grep -oP '"date_done":\s*"\K[^"]+' 2>/dev/null || true)
        if [ -n "$DATE_DONE" ]; then
            TASK_TS=$(date -d "$DATE_DONE" +%s 2>/dev/null || echo 0)
            AGE=$((NOW_TS - TASK_TS))
            if [ "$AGE" -gt "$THRESHOLD_SECONDS" ]; then
                TASK_ID=$(echo "$key" | sed 's/celery-task-meta-//')
                AGE_H=$((AGE / 3600))
                AGE_M=$(( (AGE % 3600) / 60 ))
                print_warn "Stuck task: $TASK_ID  (age: ${AGE_H}h ${AGE_M}m)"
                STUCK_COUNT=$((STUCK_COUNT + 1))
            fi
        fi
    fi
done < <(docker exec "$REDIS_CONTAINER" redis-cli --scan --pattern "celery-task-meta-*" 2>/dev/null | head -500)

if [ "$STUCK_COUNT" -eq 0 ]; then
    print_ok "No tasks appear stuck beyond ${STUCK_THRESHOLD_HOURS} hours"
else
    print_warn "$STUCK_COUNT stuck task(s) found — run ./find-stuck-tasks.sh for details"
fi

# ------------------------------------------------------------------------------
# 5. Redis memory usage
# ------------------------------------------------------------------------------
print_section "5. Redis Memory Usage"

USED_MEM=$(docker exec "$REDIS_CONTAINER" redis-cli INFO memory 2>/dev/null | grep "used_memory_human" | awk -F: '{print $2}' | tr -d '\r')
PEAK_MEM=$(docker exec "$REDIS_CONTAINER" redis-cli INFO memory 2>/dev/null | grep "used_memory_peak_human" | awk -F: '{print $2}' | tr -d '\r')
MAX_MEM=$(docker exec "$REDIS_CONTAINER" redis-cli INFO memory 2>/dev/null | grep "maxmemory_human" | awk -F: '{print $2}' | tr -d '\r')

print_info "Used memory   : ${USED_MEM:-unknown}"
print_info "Peak memory   : ${PEAK_MEM:-unknown}"
print_info "Max memory    : ${MAX_MEM:-unknown (no limit set)}"

USED_BYTES=$(docker exec "$REDIS_CONTAINER" redis-cli INFO memory 2>/dev/null | grep "^used_memory:" | awk -F: '{print $2}' | tr -d '\r')
MAX_BYTES=$(docker exec "$REDIS_CONTAINER" redis-cli INFO memory 2>/dev/null | grep "^maxmemory:" | awk -F: '{print $2}' | tr -d '\r')

if [ -n "$MAX_BYTES" ] && [ "$MAX_BYTES" -gt 0 ] && [ -n "$USED_BYTES" ]; then
    PCT=$(( (USED_BYTES * 100) / MAX_BYTES ))
    if [ "$PCT" -ge 90 ]; then
        print_fail "Memory usage at ${PCT}% — CRITICAL"
    elif [ "$PCT" -ge 75 ]; then
        print_warn "Memory usage at ${PCT}% — approaching limit"
    else
        print_ok "Memory usage at ${PCT}%"
    fi
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
print_section "Summary"
echo -e "  Flower UI: ${CYAN}http://localhost:3333${NC}"
echo -e "  For stuck tasks: ${CYAN}./find-stuck-tasks.sh ${STUCK_THRESHOLD_HOURS}${NC}"
echo ""

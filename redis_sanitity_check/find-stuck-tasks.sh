#!/bin/bash
# ==============================================================================
# find-stuck-tasks.sh
# Identify Celery tasks that have been running beyond a given threshold
# Usage: ./find-stuck-tasks.sh [hours]
#   hours  - minimum age in hours to consider a task "stuck" (default: 2)
#            Fractional values are accepted, e.g. 0.5 = 30 minutes
# Examples:
#   ./find-stuck-tasks.sh          # tasks stuck > 2 hours
#   ./find-stuck-tasks.sh 1        # tasks stuck > 1 hour
#   ./find-stuck-tasks.sh 0.5      # tasks stuck > 30 minutes
# ==============================================================================

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

REDIS_CONTAINER="quantimage2-redis-1"
EXTRACTION_CONTAINER="quantimage2-celery_extraction-1"
TRAINING_CONTAINER="quantimage2-celery_training-1"

# Accept threshold as first argument (hours, supports decimals)
THRESHOLD_HOURS="${1:-2}"
# Convert to seconds using awk for float support
THRESHOLD_SECONDS=$(awk "BEGIN {printf \"%d\", $THRESHOLD_HOURS * 3600}")

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
}

echo ""
echo -e "${BOLD}Find Stuck Celery Tasks${NC}  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "Threshold: tasks running longer than ${BOLD}${THRESHOLD_HOURS} hour(s)${NC} (${THRESHOLD_SECONDS}s)"

# ------------------------------------------------------------------------------
# Verify Redis is reachable
# ------------------------------------------------------------------------------
if ! docker exec "$REDIS_CONTAINER" redis-cli PING 2>/dev/null | grep -q "PONG"; then
    echo -e "\n${RED}[ERROR]${NC} Cannot reach Redis container '$REDIS_CONTAINER'. Is it running?"
    exit 1
fi

NOW_TS=$(date +%s)
STUCK_TASKS=()
STUCK_IDS=()

# ------------------------------------------------------------------------------
# 1. Scan celery task metadata
# ------------------------------------------------------------------------------
print_header "Scanning Task Metadata (celery-task-meta-*)"

echo "  Fetching keys from Redis..."
ALL_KEYS=$(docker exec "$REDIS_CONTAINER" redis-cli --scan --pattern "celery-task-meta-*" 2>/dev/null)
TOTAL=$(echo "$ALL_KEYS" | grep -c . || true)
echo "  Found $TOTAL task metadata key(s)"

if [ "$TOTAL" -eq 0 ]; then
    echo -e "\n  ${GREEN}No task metadata found in Redis.${NC}"
fi

echo ""
printf "  %-38s  %-10s  %-12s  %s\n" "TASK ID" "STATUS" "AGE" "date_done"
printf "  %-38s  %-10s  %-12s  %s\n" "--------------------------------------" "----------" "------------" "---------"

while IFS= read -r key; do
    [ -z "$key" ] && continue

    VALUE=$(docker exec "$REDIS_CONTAINER" redis-cli GET "$key" 2>/dev/null)
    [ -z "$VALUE" ] && continue

    STATUS=$(echo "$VALUE" | grep -oP '"status":\s*"\K[^"]+' 2>/dev/null || true)

    # Only care about actively running / in-progress states
    if [[ "$STATUS" != "STARTED" && "$STATUS" != "PROGRESS" ]]; then
        continue
    fi

    TASK_ID=$(echo "$key" | sed 's/celery-task-meta-//')

    # Try to get date_done (timestamp set when task last updated)
    DATE_DONE=$(echo "$VALUE" | grep -oP '"date_done":\s*"\K[^"]+' 2>/dev/null || true)

    if [ -n "$DATE_DONE" ] && [ "$DATE_DONE" != "null" ]; then
        TASK_TS=$(date -d "$DATE_DONE" +%s 2>/dev/null || echo 0)
        AGE=$(( NOW_TS - TASK_TS ))
        AGE_DISPLAY="${AGE}s"
        if [ "$AGE" -ge 3600 ]; then
            AGE_H=$(( AGE / 3600 ))
            AGE_M=$(( (AGE % 3600) / 60 ))
            AGE_DISPLAY="${AGE_H}h ${AGE_M}m"
        elif [ "$AGE" -ge 60 ]; then
            AGE_M=$(( AGE / 60 ))
            AGE_DISPLAY="${AGE_M}m $(( AGE % 60 ))s"
        fi

        if [ "$AGE" -gt "$THRESHOLD_SECONDS" ]; then
            printf "  ${RED}%-38s  %-10s  %-12s  %s${NC}\n" "$TASK_ID" "$STATUS" "$AGE_DISPLAY" "$DATE_DONE"
            STUCK_TASKS+=("$TASK_ID|$STATUS|$AGE_DISPLAY|$DATE_DONE")
            STUCK_IDS+=("$TASK_ID")
        else
            printf "  ${GREEN}%-38s  %-10s  %-12s  %s${NC}\n" "$TASK_ID" "$STATUS" "$AGE_DISPLAY" "$DATE_DONE"
        fi
    else
        # No timestamp — report as unknown age if status is active
        printf "  ${YELLOW}%-38s  %-10s  %-12s  %s${NC}\n" "$TASK_ID" "$STATUS" "unknown" "(no timestamp)"
        STUCK_TASKS+=("$TASK_ID|$STATUS|unknown|no timestamp")
        STUCK_IDS+=("$TASK_ID")
    fi

done <<< "$ALL_KEYS"

# ------------------------------------------------------------------------------
# 2. Cross-check with live Celery workers
# ------------------------------------------------------------------------------
print_header "Active Tasks From Workers"

for CONTAINER in "$EXTRACTION_CONTAINER" "$TRAINING_CONTAINER"; do
    echo -e "  ${BOLD}$CONTAINER${NC}"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        ACTIVE=$(docker exec "$CONTAINER" celery -A tasks inspect active --timeout 10 2>/dev/null || echo "  (failed to query worker)")
        if [ -n "$ACTIVE" ]; then
            echo "$ACTIVE" | grep -E "id|name|time_start" | sed 's/^/    /'
        else
            echo "    (no active tasks or worker unreachable)"
        fi
    else
        echo "    Container not running"
    fi
    echo ""
done

# ------------------------------------------------------------------------------
# 3. Summary & cleanup commands
# ------------------------------------------------------------------------------
print_header "Summary"

STUCK_COUNT=${#STUCK_IDS[@]}

if [ "$STUCK_COUNT" -eq 0 ]; then
    echo -e "  ${GREEN}No stuck tasks found (threshold: ${THRESHOLD_HOURS}h).${NC}"
else
    echo -e "  ${RED}${BOLD}$STUCK_COUNT stuck task(s) found:${NC}"
    echo ""
    for ENTRY in "${STUCK_TASKS[@]}"; do
        IFS='|' read -r TID TSTAT TAGE TDATE <<< "$ENTRY"
        echo -e "    ${RED}•${NC} $TID  [${TSTAT}]  running ${TAGE}"
    done

    echo ""
    echo -e "  ${BOLD}Cleanup Commands:${NC}"
    echo ""
    echo "  # Revoke a specific task (soft — sends revoke signal):"
    for TID in "${STUCK_IDS[@]}"; do
        echo "  docker exec $EXTRACTION_CONTAINER celery -A tasks control revoke $TID --terminate"
    done

    echo ""
    echo "  # Remove task metadata from Redis directly:"
    for TID in "${STUCK_IDS[@]}"; do
        echo "  docker exec $REDIS_CONTAINER redis-cli DEL \"celery-task-meta-$TID\""
    done

    echo ""
    echo "  # Inspect a specific task's full metadata:"
    for TID in "${STUCK_IDS[@]}"; do
        echo "  docker exec $REDIS_CONTAINER redis-cli GET \"celery-task-meta-$TID\" | python3 -m json.tool"
    done

    echo ""
    echo -e "  ${YELLOW}[NOTE]${NC} After cleanup, verify with: ./monitor-celery-health.sh"
fi

# ------------------------------------------------------------------------------
# 4. Bonus: check for stuck extraction groups
# ------------------------------------------------------------------------------
print_header "Extraction Group Status (celery-taskset-meta-*)"

GROUP_KEYS=$(docker exec "$REDIS_CONTAINER" redis-cli --scan --pattern "celery-taskset-meta-*" 2>/dev/null)
GROUP_TOTAL=$(echo "$GROUP_KEYS" | grep -c . || true)
echo "  Found $GROUP_TOTAL group metadata key(s)"
echo ""

if [ "$GROUP_TOTAL" -gt 0 ]; then
    printf "  %-38s  %s\n" "GROUP ID" "RESULTS"
    printf "  %-38s  %s\n" "--------------------------------------" "-------"
    while IFS= read -r gkey; do
        [ -z "$gkey" ] && continue
        GID=$(echo "$gkey" | sed 's/celery-taskset-meta-//')
        GVAL=$(docker exec "$REDIS_CONTAINER" redis-cli GET "$gkey" 2>/dev/null)
        RESULT_COUNT=$(echo "$GVAL" | grep -oP '"result":\s*\[' | wc -l || echo "?")
        printf "  %-38s  %s\n" "$GID" "$GVAL" | head -c 120
        echo ""
    done <<< "$GROUP_KEYS"
fi

echo ""
echo -e "  Inspect a group: ${CYAN}docker exec $REDIS_CONTAINER redis-cli GET \"celery-taskset-meta-GROUP_ID\" | python3 -m json.tool${NC}"
echo ""

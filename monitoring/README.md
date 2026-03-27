# QuantImage v2 — Performance Monitoring Suite

Lightweight monitoring scripts for diagnosing performance bottlenecks when multiple users access the platform simultaneously. Designed to run during live sessions with minimal resource impact.

## Quick Start

```bash
cd /srv/quantimage-v2/quantimage2_backend/monitoring

# One-off health check (instant, no background processes):
./quick_snapshot.sh

# Start continuous monitoring (for live sessions):
./start_monitoring.sh

# Check status of running monitors:
./start_monitoring.sh --status

# Stop monitoring & auto-generate report:
./start_monitoring.sh --stop

# Generate report from specific session:
./start_monitoring.sh --report 20260323_140000
```

## What Gets Monitored

| Module | Script | Tracks |
|--------|--------|--------|
| **Docker** | `monitor_docker.sh` | CPU%, memory%, net I/O, block I/O for all 18 containers (QI2 + Kheops) |
| **MySQL** | `monitor_mysql.sh` | Connections, slow queries, InnoDB buffer pool, lock contention, active queries |
| **Redis** | `monitor_redis.sh` | Memory, ops/sec, queue depths (extraction/training), task state counts, fragmentation |
| **Celery** | `monitor_celery.sh` | Worker health, active/reserved tasks, error log scanning, memory per worker |
| **Flask** | `monitor_flask.sh` | Response times, HTTP errors, file descriptor count, Socket.IO channels, container restarts |
| **Kheops** | `monitor_kheops.sh` | Endpoint response times, PACS health, PostgreSQL connections, container errors |
| **System** | `monitor_system.sh` | Host CPU/memory/IO-wait, disk I/O, network throughput, **self-resource monitoring** |

## Resource Footprint

The scripts are designed to be lightweight:
- One `docker stats --no-stream` call per interval (not per container)
- SQL queries hit `information_schema` (metadata, no table scans)
- Redis `INFO` command is O(1)
- Kheops probed via simple HTTP HEAD/GET (reuses existing auth flow)
- Self-monitoring tracks CPU/memory used by the monitoring scripts themselves
- Default interval: **10 seconds** (configurable via `SAMPLE_INTERVAL`)

Typical overhead: **<1% CPU, <50MB RAM** for all 7 monitors combined.

## Output Structure

```
monitoring/
├── logs/                                    # Time-series CSV data
│   ├── docker_resources_YYYYMMDD_HHMMSS.csv
│   ├── mysql_stats_YYYYMMDD_HHMMSS.csv
│   ├── mysql_slow_queries_YYYYMMDD_HHMMSS.log
│   ├── mysql_locks_YYYYMMDD_HHMMSS.log
│   ├── redis_celery_YYYYMMDD_HHMMSS.csv
│   ├── redis_socket_YYYYMMDD_HHMMSS.csv
│   ├── redis_queues_YYYYMMDD_HHMMSS.csv
│   ├── redis_task_counts_YYYYMMDD_HHMMSS.csv
│   ├── celery_workers_YYYYMMDD_HHMMSS.csv
│   ├── celery_events_YYYYMMDD_HHMMSS.log
│   ├── flask_api_YYYYMMDD_HHMMSS.csv
│   ├── flask_errors_YYYYMMDD_HHMMSS.log
│   ├── flask_slow_requests_YYYYMMDD_HHMMSS.log
│   ├── kheops_health_YYYYMMDD_HHMMSS.csv
│   ├── kheops_pacs_db_YYYYMMDD_HHMMSS.csv
│   ├── kheops_errors_YYYYMMDD_HHMMSS.log
│   ├── system_stats_YYYYMMDD_HHMMSS.csv
│   ├── self_resource_YYYYMMDD_HHMMSS.csv
│   ├── disk_io_YYYYMMDD_HHMMSS.csv
│   ├── network_YYYYMMDD_HHMMSS.csv
│   └── console_*_YYYYMMDD_HHMMSS.log       # Per-module alert logs
├── reports/
│   └── report_YYYYMMDD_HHMMSS.md            # Generated analysis report
├── config.sh                                # Shared configuration
├── start_monitoring.sh                      # Master orchestrator
├── quick_snapshot.sh                        # One-off health check
├── generate_report.sh                       # Report generator
└── monitor_*.sh                             # Individual modules
```

## Configuration

Edit `config.sh` to adjust:
- `SAMPLE_INTERVAL` — sampling frequency (default: 10s)
- Container names — if your Docker Compose project name differs
- MySQL credentials — if changed
- Kheops URL — if not using the default

Or override at runtime:
```bash
SAMPLE_INTERVAL=30 ./start_monitoring.sh  # Sample every 30 seconds
```

## Analyzing Results

### Generated Report
After stopping monitoring, a Markdown report is auto-generated with:
- Tabular summaries of all metrics (min/avg/max)
- Automated bottleneck detection with recommendations
- Pointers to detailed log files for each finding

### CSV Data
All CSV files can be opened in Excel, Google Sheets, or plotted with Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Docker container resources over time
df = pd.read_csv('logs/docker_resources_YYYYMMDD_HHMMSS.csv')
backend = df[df['container'] == 'quantimage2-backend-1']
backend.plot(x='timestamp', y=['cpu_pct', 'mem_pct'], figsize=(15,5))
plt.title('Backend Resource Usage')
plt.show()
```

## Alerts

Real-time alerts are printed to console logs and shown on the live dashboard:
- **HIGH CPU** — Container >80% CPU
- **HIGH MEMORY** — Container >80% memory
- **Slow queries** — New MySQL slow queries detected
- **Lock contention** — InnoDB transactions waiting for locks
- **Queue backlog** — >5 tasks waiting in extraction/training queue
- **Slow response** — Backend >2s response time
- **I/O wait** — Host >20% CPU in I/O wait
- **Service down** — Container not running or endpoint unreachable

## Containers Monitored

### QuantImage v2
- `quantimage2-backend-1` — Flask API + Socket.IO
- `quantimage2-celery_extraction-1` — Feature extraction workers
- `quantimage2-celery_training-1` — ML training workers
- `quantimage2-flower-1` — Celery monitoring
- `quantimage2-db-1` — MySQL 5.7
- `quantimage2-redis-1` — Celery broker
- `quantimage2-redis-socket-1` — Socket.IO message queue
- `quantimage2-ohif-1` — DICOM viewer

### Kheops (PACS)
- `kheopsreverseproxy` — Nginx reverse proxy
- `kheopsauthorization` — Authorization service
- `kheopsdicomwebproxy` — DICOMweb proxy
- `kheopszipper` — ZIP download service
- `kheopspostgres` — Kheops PostgreSQL
- `kheopsui` — Web UI
- `pacsarc` — DCM4CHEE PACS archive
- `pacspostgres` — PACS PostgreSQL
- `pacsauthorizationproxy` — PACS auth proxy
- `ldap` — LDAP directory

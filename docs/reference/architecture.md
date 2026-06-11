# Architecture, Repository Structure & Tech Stack

## Project Overview

QuantImage v2 is a **radiomics research platform** for medical imaging. This repository is the **backend**, written in Python, running across multiple Docker containers. It provides:

- A **Flask REST API** + **Socket.IO** server for the frontend SPA
- **Celery workers** for asynchronous feature extraction (PyRadiomics/Okapy) and ML model training
- **MySQL** database for features, labels, models, and metadata
- **Redis** for Celery broker/result backend and Socket.IO message queue
- Integration with **Kheops** (PACS/DICOMweb) for medical image storage and **Keycloak** for OIDC authentication

### Data Flow

```
Frontend (React SPA)
  → Backend API (Flask, port 5000)
    → Kheops (DICOMweb API): download DICOM studies
    → Celery Workers:
        extraction queue → PyRadiomics/Okapy → feature values stored in MySQL
        training queue   → scikit-learn / scikit-survival → model files saved to disk
    → Socket.IO: real-time progress updates to frontend
```

---

## Repository Structure

```
quantimage2_backend/
├── shared/                              # Shared library (quantimage2_backend_common)
│   ├── setup.py
│   └── quantimage2_backend_common/
│       ├── const.py                     # Enums, constants, feature ID regex
│       ├── feature_storage.py           # Store/retrieve features from MySQL
│       ├── flask_init.py                # Flask app factory, DB connection
│       ├── kheops_utils.py              # Kheops API client, DICOM field mappings
│       ├── modeling_utils.py            # Survival CV splitter, c-index scorer
│       ├── models.py                    # SQLAlchemy ORM models (1670 lines)
│       └── utils.py                     # Error classes, formatters, Socket.IO helpers
├── webapp/                              # Flask web application
│   ├── app.py                           # App entry point, blueprint registration
│   ├── config.py                        # Keycloak client, Flask config
│   ├── populate.py                      # Seed default feature presets
│   ├── routes/                          # Flask Blueprints (REST endpoints)
│   │   ├── utils.py                     # Auth decorators (decode_token, validate_decorate)
│   │   ├── features.py                  # /extractions/*, /extract/*
│   │   ├── models.py                    # /models/*
│   │   ├── albums.py                    # /albums/*
│   │   ├── tasks.py                     # /tasks/*
│   │   ├── labels.py                    # /labels/*
│   │   ├── charts.py                    # /charts/*
│   │   ├── feature_presets.py           # /feature-presets/*
│   │   ├── feature_collections.py       # /feature-collections/*
│   │   ├── clinical_features.py         # /clinical-features/*
│   │   └── navigation_history.py        # Navigation tracking
│   ├── modeling/                         # ML pipeline
│   │   ├── modeling.py                  # Abstract Modeling base class
│   │   ├── classification.py            # Classification (LR, SVM, RF)
│   │   ├── survival.py                  # Survival analysis (CoxPH, CoxNet, IPC)
│   │   └── utils.py                     # Normalization (StandardScaler, MinMaxScaler)
│   ├── service/                         # Business logic layer
│   │   ├── feature_extraction.py        # Orchestrates extraction (Celery chord)
│   │   └── machine_learning.py          # Orchestrates ML training
│   ├── presets_default/                 # Default PyRadiomics YAML configs
│   ├── alembic/                         # Database migrations
│   └── scripts/                         # Data import/parsing scripts
├── workers/                             # Celery worker processes
│   ├── tasks.py                         # Task definitions (extract, train)
│   ├── utils.py                         # Worker utilities
│   ├── celeryconfig.py                  # Celery broker/backend config
│   └── config_workers.py               # Worker-specific Flask config
├── env_files/                           # Environment variable files
├── secrets/                             # Docker secrets (passwords)
├── docker-compose.yml                   # Base Docker Compose
├── docker-compose.override.yml          # Development overrides
├── docker-compose.local.yml             # Local deployment
└── docker-compose.prod.yml             # Production (Traefik, backups)
```

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| **Web Framework** | Flask | 1.1.2 |
| **ORM** | SQLAlchemy (via Flask-SQLAlchemy) | 1.3.24 |
| **Database** | MySQL | 5.7 |
| **Migrations** | Alembic | 1.14.1 |
| **Task Queue** | Celery | 5.5.0 |
| **Message Broker** | Redis | 7.4.1 |
| **WebSocket** | Flask-SocketIO (eventlet) | 4.3.2 |
| **Auth** | Keycloak (python-keycloak) | 3.6.1 |
| **DICOM** | pydicom, SimpleITK | 2.4.4, 2.4.1 |
| **Radiomics** | PyRadiomics (via Okapy) | 3.1.0 |
| **ML Classification** | scikit-learn | 1.3.2 |
| **ML Survival** | scikit-survival | 0.22.2 |
| **Data** | pandas, numpy | 1.5.3, 1.23.5 |
| **Model Persistence** | joblib | (bundled with sklearn) |
| **Code Formatter** | Black | 24.8.0 |

---


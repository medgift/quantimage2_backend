# Database & ORM Patterns

## Database & ORM Patterns

### Model Definition

All models inherit from `BaseModel` (provides `id`, `created_at`, `updated_at`, CRUD methods):

```python
class MyModel(BaseModel, db.Model):
    def __init__(self, name: str, user_id: str):
        self.name = name
        self.user_id = user_id

    name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255), nullable=False)

    # Foreign keys
    extraction_id = db.Column(db.Integer, db.ForeignKey("feature_extraction.id"))

    # Methods
    @classmethod
    def find_by_user(cls, user_id: str) -> List["MyModel"]:
        return db.session.query(cls).filter_by(user_id=user_id).all()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
        }
```

### Key BaseModel Methods

| Method | Purpose |
|---|---|
| `find_by_id(id)` | Query by primary key |
| `delete_by_id(id)` | Delete + commit |
| `find_all()` | Query all records |
| `get_or_create(criteria, defaults)` | Upsert pattern with row-level locking |
| `save_to_db()` | Add + commit |
| `flush_to_db()` | Add + flush (no commit, for use within transactions) |
| `update(**kwargs)` | Update attributes + commit |

### Key ORM Models

| Model | Table | Purpose |
|---|---|---|
| `FeatureExtraction` | `feature_extraction` | One extraction run per album |
| `FeatureExtractionTask` | `feature_extraction_task` | One task per study in an extraction |
| `FeatureValue` | `feature_value` | Individual feature measurement |
| `FeatureDefinition` | `feature_definition` | Feature name registry |
| `Modality` | `modality` | CT, PET, MR, etc. |
| `ROI` | `roi` | GTV_T, GTV_N, etc. |
| `FeatureCollection` | `feature_collection` | User-defined feature subset |
| `Model` | `model` | Trained ML model record |
| `LabelCategory` | `label_category` | Outcome definition (Classification/Survival) |
| `Label` | `label` | Patient outcome values |
| `ClinicalFeatureFile` | `clinical_feature_file` | One uploaded clinical-features CSV per (user, album); owns its definitions (unique on user+album+name) |
| `ClinicalFeatureDefinition` | `clinical_feature_definition` | Clinical feature schema (FK → `clinical_feature_file`, `ON DELETE CASCADE`) |
| `ClinicalFeatureValue` | `clinical_feature_value` | Clinical feature data (FK → `clinical_feature_definition`, `ON DELETE CASCADE`) |

Clinical features support multiple CSVs per album: feature IDs are namespaced
`{clinical_feature_file_id}::{name}` (`CLINICAL_FEATURE_ID_SEPARATOR` in `const.py`) so the same
column name in two files maps to two distinct columns. Deleting a file cascades to its
definitions and their values via the DB foreign keys.

### Database Connection

```python
# Connection string pattern:
mysql://{DB_USER}:{secret}@db/{DB_DATABASE}

# Pool settings (flask_init.py):
SQLALCHEMY_POOL_SIZE = 10
SQLALCHEMY_MAX_OVERFLOW = 20
SQLALCHEMY_POOL_TIMEOUT = 30
SQLALCHEMY_POOL_PRE_PING = True  # Reconnect stale connections
SQLALCHEMY_POOL_RECYCLE = 1800   # Recycle every 30 minutes
```

### Migrations

```bash
# Generate a new migration (run inside the backend container, where alembic.ini lives):
docker compose exec backend alembic revision --autogenerate -m "describe change"

# Apply migrations:
docker compose exec backend alembic upgrade head
```

On startup the backend entrypoint runs **only** `alembic upgrade head` when `DB_AUTOMIGRATE=1`
(set for dev/local). It does **not** auto-generate migrations — generate them by hand after a
model change, review them (autogenerate emits schema DDL only; add data backfills yourself),
then apply. Migration files under `webapp/alembic/versions/` are **git-ignored**, and prod does
**not** set `DB_AUTOMIGRATE` (migrations are applied there manually). See the "Migration
workflow" note in [CLAUDE.md](../../CLAUDE.md) for the authoritative workflow and the
prod / git-ignore caveats.

---


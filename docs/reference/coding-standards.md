# Coding Standards, Formatting & Testing

## Coding Standards

### Python Style

- **PEP 8** compliance, enforced by **Black** formatter (line length: Black default 88 chars).
- Use **type hints** for all new function signatures (parameters and return types).
- Use **f-strings** for string formatting, never `%` or `.format()`.
- Use **`pathlib.Path`** for file system operations in new code (existing code uses `os.path`).
- Prefer **list comprehensions** over `map()`/`filter()` for readability.
- Use **`logging`** module (not `print()`) for all new diagnostic output. Existing code uses `print()` — do not refactor unless explicitly asked.
- Follow existing naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

### Type Hinting

```python
from typing import List, Dict, Optional, Tuple, Any

def process_features(
    extraction_id: int,
    feature_ids: List[str],
    normalize: bool = True,
) -> pd.DataFrame:
    ...
```

- Use `Optional[X]` for parameters that can be `None`.
- Use `-> None` for functions that return nothing.
- For SQLAlchemy models, type hint query results as the model class.

### Error Handling

```python
from quantimage2_backend_common.utils import InvalidUsage, ComputationError

# For client errors (bad request, missing data):
raise InvalidUsage("Patient labels are missing", status_code=400)

# For server-side computation errors:
raise ComputationError("Feature extraction failed for study {study_uid}", status_code=500)
```

- **Never swallow exceptions silently.** At minimum, log the error.
- Use specific exception types — avoid bare `except:`.
- For Celery tasks, catch `SoftTimeLimitExceeded` to handle timeout gracefully.
- Log all errors with `logging.error()` or `logging.exception()` (includes traceback).
- Medical data processing must fail loudly — never return partial results without explicit warning.

### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Starting extraction {extraction_id} for album {album_id}")
logger.warning(f"Missing ROI {roi_name} in study {study_uid}")
logger.error(f"Extraction failed: {e}", exc_info=True)
```

- **NEVER log** patient names, JWT tokens, passwords, or Docker secrets.
- DICOM UIDs (StudyInstanceUID, SeriesInstanceUID) may be logged for debugging.
- Use structured context in log messages: include `extraction_id`, `study_uid`, `album_id` where applicable.

### Imports

```python
# 1. Standard library
import os
import logging
from typing import List, Dict

# 2. Third-party packages
import pandas as pd
import numpy as np
from flask import request, jsonify, g
from celery import Celery

# 3. Local/shared packages
from quantimage2_backend_common.models import FeatureExtraction, FeatureValue
from quantimage2_backend_common.utils import InvalidUsage
```

- Group imports in three sections: stdlib, third-party, local.
- Use absolute imports for the shared package: `from quantimage2_backend_common.X import Y`.
- Within webapp or workers, use relative imports for sibling modules: `from routes.utils import validate_decorate`.

---


## Code Formatting

After every code edit, ensure modified Python files conform to **Black** formatting:

```bash
black <modified_files>
```

Verify:
```bash
black --check <modified_files>
```

Format only the files you changed — do not reformat the entire codebase.

---

## Testing & Validation

### Running the Stack

```bash
# Development (with source code mounts):
docker-compose up

# Local deployment (built images):
docker-compose -f docker-compose.yml -f docker-compose.local.yml up

# Production:
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Validate Docker Compose

```bash
docker-compose config --quiet && echo "Config OK"
```

### Database Access

```bash
# Via phpMyAdmin:
http://localhost:8888

# Via MySQL CLI:
docker-compose exec db mysql -u quantimage2 -p quantimage2
```

### Celery Monitoring

```bash
# Flower dashboard:
http://localhost:3333

# Check worker status:
docker-compose exec celery_extraction celery -A tasks inspect active
```

---


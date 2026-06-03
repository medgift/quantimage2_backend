# Feature Extraction Pipeline

## Feature Extraction Pipeline

### Flow

1. **Route**: POST `/extract/album/<album_id>` → `routes/features.py`
2. **Service**: `service/feature_extraction.py` → `run_feature_extraction()`
   - Creates `FeatureExtraction` + `FeatureExtractionTask` DB records
   - Saves YAML config to `/quantimage2-data/extractions/configs/{user_id}/{album_id}/`
   - Dispatches Celery chord to `extraction` queue
3. **Worker**: `workers/tasks.py` → `run_extraction()`
   - Downloads DICOM study from Kheops as ZIP
   - Extracts to temp directory
   - Runs `okapy.dicomconverter.converter.ExtractorConverter` with YAML config
   - Calls `store_features()` to persist to MySQL
   - Emits Socket.IO progress events
4. **Finalization**: Updates extraction status, emits final Socket.IO event

### Configuration Format

YAML files defining PyRadiomics extraction parameters. Default presets in `webapp/presets_default/`:
- `MRI.yaml`, `MRI_CT.yaml`, `MRI_PET.yaml`, `PETCT_all.yaml`

### Feature Storage

Features are stored in **MySQL** via `feature_storage.py`:
- DataFrame columns: `patient`, `modality`, `VOI` (ROI), `feature_name`, `feature_value`
- Creates/updates `Modality`, `ROI`, `FeatureDefinition` records
- Bulk-inserts `FeatureValue` records via `bulk_insert_mappings`

**HDF5 caching** for retrieval performance:
- Path: `/quantimage2-data/features-cache/extraction-{id}/features.h5`
- Format: pandas HDF5 `fixed` format
- Cache is invalidated/rebuilt when features change

### Feature ID Format

```
{modality}‑{roi}‑{feature_name}
```

**CRITICAL:** The separator is a **non-breaking hyphen** (`‑`, U+2011), NOT a regular hyphen (`-`). ROI names can contain regular hyphens (e.g., `GTV-T`). The regex `featureIDMatcher` in `const.py` parses this format.

### Feature Prefixes (by extractor)

| Source | Prefixes | Status |
|---|---|---|
| PyRadiomics | `original`, `log`, `wavelet`, `gradient`, `square`, `squareroot`, `exponential`, `logarithm` | Active |
| Riesz | `tex` | Legacy — parsed for existing data, no longer computed (MATLAB support removed) |
| ZRAD | `zrad` | Legacy — parsed for existing data, no longer computed (ZRAD build support removed) |

The prefix lists live in `const.py` (`prefixes`, consumed by `featureIDMatcher`). Keep the legacy
prefixes so existing feature IDs in the DB still parse — removing one makes the matcher silently
drop or misclassify those rows.

### Rules for Feature Code

- Always use `FEATURE_ID_SEPARATOR` (U+2011) when constructing or parsing feature IDs.
- Store features via `store_features()` / `FeatureValue.save_features_batch()` — never insert directly.
- Clean up temporary DICOM files after extraction (use `try/finally` or context managers).
- Respect the YAML config format — validate configs before dispatching extraction.
- Feature names must start with a known prefix in `const.py` (PyRadiomics for new extractions; Riesz/ZRAD are legacy, parse-only).

---


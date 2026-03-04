# Radiomics Feature Engineering — PyRadiomics/Okapy Extraction

Use this prompt when creating or modifying feature extraction scripts, YAML configs, or the extraction pipeline.

## Context

@workspace The radiomics feature extraction pipeline works as follows:

1. **Trigger**: POST `/extract/album/<album_id>` → `webapp/routes/features.py`
2. **Orchestration**: `webapp/service/feature_extraction.py` → `run_feature_extraction()`
   - Creates `FeatureExtraction` + `FeatureExtractionTask` DB records
   - Saves YAML config to `/quantimage2-data/extractions/configs/{user_id}/{album_id}/`
   - Dispatches a Celery **chord** — one task per study, final callback to finalize
3. **Worker**: `workers/tasks.py` → `run_extraction()`
   - Downloads DICOM from Kheops as ZIP via `download_study()`
   - Extracts to temp dir, runs `okapy.dicomconverter.converter.ExtractorConverter`
   - Calls `store_features()` from `shared/quantimage2_backend_common/feature_storage.py`
   - Emits Socket.IO progress events
4. **Storage**: MySQL via `FeatureValue.save_features_batch()` (bulk insert)
5. **Cache**: HDF5 at `/quantimage2-data/features-cache/extraction-{id}/features.h5`

## Critical Rules

- **Feature ID format**: `{modality}‑{roi}‑{feature_name}` where `‑` is U+2011 (non-breaking hyphen), NOT `-`.
- Always use `FEATURE_ID_SEPARATOR` from `quantimage2_backend_common.const` when constructing/parsing feature IDs.
- Feature names must start with a known prefix: `original`, `log`, `wavelet`, `gradient`, `square`, `squareroot`, `exponential`, `logarithm` (PyRadiomics), `tex` (Riesz), or `zrad` (ZRAD).
- Store features via `store_features()` — never insert `FeatureValue` rows directly.
- Clean up temp DICOM files in `try/finally` blocks.
- YAML configs define PyRadiomics extraction parameters — validate before dispatching.
- Default presets are in `webapp/presets_default/` (MRI.yaml, MRI_CT.yaml, MRI_PET.yaml, PETCT_all.yaml).

## Key Files

- `workers/tasks.py` — Celery task definitions (`run_extraction`, `finalize_extraction_task`, `finalize_extraction`)
- `webapp/service/feature_extraction.py` — Orchestration (chord dispatch)
- `shared/quantimage2_backend_common/feature_storage.py` — `store_features()`, feature retrieval
- `shared/quantimage2_backend_common/const.py` — `FEATURE_ID_SEPARATOR`, `featureIDMatcher`, feature prefix lists
- `shared/quantimage2_backend_common/models.py` — `FeatureExtraction`, `FeatureExtractionTask`, `FeatureValue`, `FeatureDefinition`, `Modality`, `ROI`
- `webapp/routes/features.py` — REST endpoints for extractions

## When Modifying Extraction Code

1. Check `workers/tasks.py` for the current `run_extraction()` flow.
2. Check `shared/quantimage2_backend_common/feature_storage.py` for `store_features()`.
3. Verify feature ID format uses `FEATURE_ID_SEPARATOR` (U+2011).
4. Ensure Socket.IO progress events are emitted via `send_extraction_status_message()`.
5. Handle `SoftTimeLimitExceeded` (soft limit: 6600s, hard limit: 7200s).
6. Delete temp DICOM dirs in `finally` blocks.
7. Run `black` on modified files.

## When Adding New Feature Types

1. Add prefix to the appropriate list in `const.py` (`PYRADIOMICS_FEATURE_PREFIXES`, `RIESZ_FEATURE_PREFIXES`, or `ZRAD_FEATURE_PREFIXES`).
2. Update `featureIDMatcher` regex if the new prefix pattern differs.
3. Ensure `store_features()` handles the new feature DataFrame format.
4. Test that feature IDs round-trip correctly through `featureIDMatcher`.

## When Modifying YAML Configs

1. Validate YAML syntax before saving to disk.
2. Follow PyRadiomics YAML config schema (see existing presets in `webapp/presets_default/`).
3. Ensure the config is compatible with `ExtractorConverter.from_params()`.

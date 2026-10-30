"""Overview of a feature extraction: how it was configured, what was extracted
and on which dataset.

The configuration is the legacy okapy YAML stored by ``save_config`` at
extraction time (``general`` / ``volume_preprocessing`` / ``feature_extraction``).
Everything else comes from the database only: no Kheops or Celery calls, and no
patient identifiers in the output.
"""

import logging
import math
from pathlib import Path

import yaml

import config
from quantimage2_backend_common.const import (
    PYRADIOMICS_FEATURE_PREFIXES,
    RIESZ_FEATURE_PREFIXES,
    ZRAD_FEATURE_PREFIXES,
)
from quantimage2_backend_common.models import FeatureExtractionTask

logger = logging.getLogger(__name__)

CONFIG_STATUS_OK = "ok"
CONFIG_STATUS_MISSING = "missing"
CONFIG_STATUS_INVALID = "invalid"
CONFIG_STATUS_UNRECOGNIZED = "unrecognized"

# Sections that are not modalities (okapy's legacy format)
COMMON_SECTION = "common"
DEFAULT_SECTION = "default"
SHARED_SECTIONS = (COMMON_SECTION, DEFAULT_SECTION)

RESAMPLER_KEY = "bspline_resampler"
NORMALIZER_KEYS = ("standardizer", "masked_standardizer", "combined_standardizer")
PYRADIOMICS_BACKEND = "pyradiomics"
PYRADIOMICS_SETTING_KEYS = (
    "binWidth",
    "binCount",
    "voxelArrayShift",
    "resampledPixelSpacing",
    "normalize",
    "normalizeScale",
)

# B-spline order -> interpolation name
INTERPOLATION_LABELS = {0: "nearest", 1: "linear", 3: "cubic"}

RIESZ_FAMILY = "riesz"
ZRAD_FAMILY = "zrad"
OTHER_FAMILY = "other"

# Limits on what is read from the (user-editable) config and echoed back
MAX_CONFIG_BYTES = 1_000_000
MAX_VALUE_DEPTH = 3
MAX_VALUE_ITEMS = 50
MAX_STRING_LENGTH = 200


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _json_value(value, depth=0):
    """Copy a YAML value into something JSON can encode, bounded in size.

    YAML can hold values JSON cannot (sets, binary, NaN, recursive aliases), and
    aliases can make a small file expand into a huge structure.
    """
    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]

    if isinstance(value, (list, tuple)):
        if depth >= MAX_VALUE_DEPTH:
            return None
        return [_json_value(v, depth + 1) for v in value[:MAX_VALUE_ITEMS]]

    if isinstance(value, dict):
        if depth >= MAX_VALUE_DEPTH:
            return None
        return {
            str(k)[:MAX_STRING_LENGTH]: _json_value(v, depth + 1)
            for k, v in list(value.items())[:MAX_VALUE_ITEMS]
        }

    return str(value)[:MAX_STRING_LENGTH]


def _interpolation_label(order):
    if isinstance(order, bool) or not isinstance(order, int):
        return None

    return INTERPOLATION_LABELS.get(order, f"bspline order {order}")


def _select_section(section, key):
    """Section okapy applies to a modality key: the key itself, then its
    modality prefix (``MR_T1Pre`` -> ``MR``), then ``default``.

    A key that is present with an empty value still wins over ``default``.
    """
    if key in section:
        return _as_dict(section[key])

    modality = str(key).split("_", maxsplit=1)[0]
    if modality in section:
        return _as_dict(section[modality])

    return _as_dict(section.get(DEFAULT_SECTION))


def _parse_resampling(common, selected):
    """The selected section's resampler is merged over the common one."""
    if RESAMPLER_KEY not in common and RESAMPLER_KEY not in selected:
        return None

    resampler = {
        **_as_dict(common.get(RESAMPLER_KEY)),
        **_as_dict(selected.get(RESAMPLER_KEY)),
    }
    order = resampler.get("order")

    return {
        "spacing": _json_value(resampler.get("resampling_spacing")),
        "order": _json_value(order),
        "interpolation": _interpolation_label(order),
    }


def _parse_normalization(common, selected):
    """okapy runs the common processors first, then the selected ones, so the
    selected section's normalizer is the one that ends up applied."""
    for section in (selected, common):
        for normalizer in NORMALIZER_KEYS:
            if normalizer in section:
                return {
                    "type": normalizer,
                    "params": _json_value(_as_dict(section[normalizer])),
                }

    return None


def _parse_pyradiomics_extractor(name, extractor):
    extractor = _as_dict(extractor)
    settings = _as_dict(extractor.get("setting"))

    return {
        "backend": PYRADIOMICS_BACKEND,
        "name": _json_value(name),
        "image_types": _json_value(list(_as_dict(extractor.get("imageType")))),
        "feature_classes": _json_value(list(_as_dict(extractor.get("featureClass")))),
        "settings": {
            key: _json_value(value)
            for key, value in settings.items()
            if key in PYRADIOMICS_SETTING_KEYS
        },
    }


def _parse_extractors(modality_extraction):
    extractors = []
    for backend, body in _as_dict(modality_extraction).items():
        if backend == PYRADIOMICS_BACKEND:
            extractors.extend(
                _parse_pyradiomics_extractor(name, extractor)
                for name, extractor in _as_dict(body).items()
            )
        else:
            # Only pyradiomics settings are interpreted, other backends are listed
            extractors.append(
                {
                    "backend": _json_value(backend),
                    "name": _json_value(backend),
                    "image_types": [],
                    "feature_classes": [],
                    "settings": {},
                }
            )

    return extractors


def _not_ok_config(status):
    return {"status": status, "general": None, "modalities": []}


def parse_extraction_config(config_dict, modalities=None):
    """Summarize a parsed (legacy okapy) extraction config; never raises.

    Settings are resolved per modality the way okapy applies them. When
    ``modalities`` (the extracted ones) is given, one row is reported for each.
    """
    if not isinstance(config_dict, dict):
        return _not_ok_config(CONFIG_STATUS_INVALID)

    if (
        "volume_preprocessing" not in config_dict
        and "feature_extraction" not in config_dict
    ):
        return _not_ok_config(CONFIG_STATUS_UNRECOGNIZED)

    general = config_dict.get("general")
    if isinstance(general, dict):
        general = {
            "padding": _json_value(general.get("padding")),
            "submodalities": _json_value(general.get("submodalities")),
            "combine_segmentation": _json_value(general.get("combine_segmentation")),
        }
    else:
        general = None

    volume_preprocessing = _as_dict(config_dict.get("volume_preprocessing"))
    feature_extraction = _as_dict(config_dict.get("feature_extraction"))

    # A legacy section is a mapping (or empty); scalar entries belong to other
    # config formats (e.g. deep features)
    sections = [
        (key, value)
        for section in (volume_preprocessing, feature_extraction)
        for key, value in section.items()
        if value is None or isinstance(value, dict)
    ]
    if not sections:
        return _not_ok_config(CONFIG_STATUS_UNRECOGNIZED)

    # Rows: the modalities actually extracted, else the configured ones.
    # Display name -> key to look up (YAML keys are not always strings)
    if modalities:
        row_keys = {str(name): str(name) for name in modalities}
    else:
        row_keys = {}
        for key, _ in sections:
            if str(key) not in SHARED_SECTIONS:
                row_keys.setdefault(str(key), key)

    common_preprocessing = _as_dict(volume_preprocessing.get(COMMON_SECTION))
    common_extractors = _parse_extractors(feature_extraction.get(COMMON_SECTION))

    rows = []
    for name in sorted(row_keys):
        key = row_keys[name]
        selected_preprocessing = _select_section(volume_preprocessing, key)
        rows.append(
            {
                "modality": name[:MAX_STRING_LENGTH],
                "resampling": _parse_resampling(
                    common_preprocessing, selected_preprocessing
                ),
                "normalization": _parse_normalization(
                    common_preprocessing, selected_preprocessing
                ),
                "extractors": common_extractors
                + _parse_extractors(_select_section(feature_extraction, key)),
            }
        )

    return {"status": CONFIG_STATUS_OK, "general": general, "modalities": rows}


def _feature_family_and_filter(name):
    if name.startswith(tuple(RIESZ_FEATURE_PREFIXES)):
        return RIESZ_FAMILY, None

    if name.startswith(tuple(ZRAD_FEATURE_PREFIXES)):
        return ZRAD_FAMILY, None

    # PyRadiomics: <filter>[-<filter params>]_<family>_<feature>
    tokens = name.split("_")
    image_filter = tokens[0].split("-")[0]
    if image_filter in PYRADIOMICS_FEATURE_PREFIXES and len(tokens) >= 3:
        return tokens[1], image_filter

    return OTHER_FAMILY, None


def summarize_feature_names(names):
    """Count unique feature names by family and by pyradiomics image filter."""
    unique_names = list(dict.fromkeys(names))

    by_family = {}
    filters = {}
    for name in unique_names:
        family, image_filter = _feature_family_and_filter(name)
        by_family[family] = by_family.get(family, 0) + 1
        if image_filter is not None:
            filters[image_filter] = filters.get(image_filter, 0) + 1

    return {"total": len(unique_names), "by_family": by_family, "filters": filters}


def load_extraction_config(extraction):
    """Read and parse the extraction's YAML config.

    ``config_file`` is not trusted: only a file under the owner's config
    directory is read. Returns ``(status, parsed_yaml)``.
    """
    if not extraction.config_file:
        return CONFIG_STATUS_MISSING, None

    configs_root = Path(config.EXTRACTIONS_BASE_DIR, config.CONFIGS_SUBDIR).resolve()
    allowed_dir = (configs_root / str(extraction.user_id)).resolve()
    config_path = Path(extraction.config_file).resolve()

    # The user's directory must be exactly one level below the configs root
    # (rules out user IDs such as "", "." or "..")
    if (
        allowed_dir.parent != configs_root
        or not config_path.is_relative_to(allowed_dir)
        or not config_path.is_file()
    ):
        logger.warning(
            f"Config file of extraction {extraction.id} is missing or outside the user's config directory"
        )
        return CONFIG_STATUS_MISSING, None

    try:
        if config_path.stat().st_size > MAX_CONFIG_BYTES:
            return CONFIG_STATUS_INVALID, None
        return CONFIG_STATUS_OK, yaml.safe_load(config_path.read_text("utf-8"))
    except OSError:
        return CONFIG_STATUS_MISSING, None
    except (yaml.YAMLError, UnicodeDecodeError, RecursionError, ValueError):
        return CONFIG_STATUS_INVALID, None


def build_extraction_summary(extraction):
    status, config_dict = load_extraction_config(extraction)
    extracted_modalities = sorted(m.name for m in extraction.modalities)
    summary_config = (
        parse_extraction_config(config_dict, modalities=extracted_modalities)
        if status == CONFIG_STATUS_OK
        else _not_ok_config(status)
    )

    features = summarize_feature_names(
        [definition.name for definition in extraction.feature_definitions]
    )
    features["modalities"] = extracted_modalities
    features["rois"] = sorted(r.name for r in extraction.rois)

    training_patients = extraction.training_patients
    test_patients = extraction.test_patients

    return {
        "extraction_id": extraction.id,
        "created_at": extraction._serialize_dt(extraction.created_at),
        "config": summary_config,
        "features": features,
        "dataset": {
            "studies": FeatureExtractionTask.query.filter_by(
                feature_extraction_id=extraction.id
            ).count(),
            "data_splitting_type": extraction.data_splitting_type,
            "train_test_split_type": extraction.train_test_split_type,
            "training_patients": (
                len(training_patients) if training_patients is not None else None
            ),
            "test_patients": len(test_patients) if test_patients is not None else None,
        },
    }

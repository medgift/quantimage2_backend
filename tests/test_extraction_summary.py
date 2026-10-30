"""
Tests for the extraction overview summary: GET /extractions/<id>/summary and
the pure helpers behind it (service.extraction_summary).

The endpoint reports how the features of an extraction were produced (the
stored YAML configuration), what was extracted (feature families, filters,
modalities, ROIs) and the dataset counts (studies, split sizes). It must be
owner-only, read-only, never touch Kheops or Celery, never return patient IDs
or the config file path, and only read config files under the owner's config
directory.
"""

import datetime
import importlib
import json
import os
from unittest.mock import MagicMock, patch

import pytest

USER = "test-user-uuid-1234"
OTHER_USER = "other-user-uuid-5678"
ALBUM = "alb-summary"
NOT_FOUND = {"error": "Feature extraction not found"}

# Identifiers that must never leak into the summary response
SECRET_PATIENTS_TRAIN = ["PATIENT-SECRET-001", "PATIENT-SECRET-002", "PAT-S-003"]
SECRET_PATIENTS_TEST = ["PATIENT-SECRET-104", "PATIENT-SECRET-105"]
SECRET_STUDIES = [
    "1.2.840.99999.STUDY-SECRET-1",
    "1.2.840.99999.STUDY-SECRET-2",
    "1.2.840.99999.STUDY-SECRET-3",
]


# ---------------------------------------------------------------------------
# YAML configs modelled on webapp/presets_default
# ---------------------------------------------------------------------------

PETCT_YAML = """\
general:
  padding: 5
  submodalities: False
  result_format: "long"
  combine_segmentation: True
  additional_dicom_tags:
    - "SeriesInstanceUID"
    - "SeriesDescription"

volume_preprocessing:
  common:
    bspline_resampler:
      resampling_spacing: [1.0, 1.0, 1.0]
      order: 1
  default:

feature_extraction:
  CT:
    pyradiomics:
      original:
        imageType:
          Original: {}
        featureClass:
          shape:
          firstorder:
          glcm:
          glszm:
          glrlm:
          ngtdm:
          gldm:

        setting:
          binWidth: 20
          voxelArrayShift: 1000
          label: 1

  PT:
    pyradiomics:
      original:
        imageType:
          Original: {}
        featureClass:
          shape:
          firstorder:
          glcm:
          glszm:
          glrlm:
          ngtdm:
          gldm:

        setting:
          binWidth: 0.2
          voxelArrayShift: 0
          label: 1
"""

MRI_YAML = """\
general:
  padding: 5
  submodalities: True
  result_format: "long"
  combine_segmentation: False
  additional_dicom_tags:
    - "SeriesInstanceUID"
    - "SeriesDescription"

volume_preprocessing:
  common:
    bspline_resampler:
      resampling_spacing: [1.0, 1.0, 1.0]
      order: 1
    standardizer:
      threshold: 0.001
  default:

feature_extraction:
  MR_T1:
    pyradiomics:
      original:
        imageType:
          Original: {}
        featureClass:
          shape:
          firstorder:
          glcm:
          glszm:
          glrlm:
          ngtdm:
          gldm:

        setting:
          binWidth: 0.3
          voxelArrayShift: 10
          label: 1

  MR_T2:
    pyradiomics:
      original:
        imageType:
          Original: {}
        featureClass:
          shape:
          firstorder:
          glcm:
          glszm:
          glrlm:
          ngtdm:
          gldm:

        setting:
          binWidth: 0.3
          voxelArrayShift: 10
          label: 1
"""

PYRADIOMICS_CLASSES = ["shape", "firstorder", "glcm", "glszm", "glrlm", "ngtdm", "gldm"]


def _pyrad_extractor(settings, image_types=("Original",), classes=None):
    return {
        "backend": "pyradiomics",
        "name": "original",
        "image_types": list(image_types),
        "feature_classes": list(classes or PYRADIOMICS_CLASSES),
        "settings": settings,
    }


LINEAR_1MM = {"spacing": [1.0, 1.0, 1.0], "order": 1, "interpolation": "linear"}

EXPECTED_PETCT_CONFIG = {
    "status": "ok",
    "general": {"padding": 5, "submodalities": False, "combine_segmentation": True},
    "modalities": [
        {
            "modality": "CT",
            "resampling": LINEAR_1MM,
            "normalization": None,
            "extractors": [_pyrad_extractor({"binWidth": 20, "voxelArrayShift": 1000})],
        },
        {
            "modality": "PT",
            "resampling": LINEAR_1MM,
            "normalization": None,
            "extractors": [_pyrad_extractor({"binWidth": 0.2, "voxelArrayShift": 0})],
        },
    ],
}

NOT_OK_CONFIG_SHAPE = {"general": None, "modalities": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svc():
    """Import the service under test lazily, so route tests still run (and fail
    with a 404) while the module does not exist yet."""
    return importlib.import_module("service.extraction_summary")


def _yaml(text):
    import yaml

    return yaml.safe_load(text)


def _patch_decode_token(user=USER):
    return patch(
        "routes.utils.decode_token",
        return_value={
            "sub": user,
            "preferred_username": "testuser",
            "resource_access": {"quantimage2-frontend": {"roles": ["admin"]}},
        },
    )


def _get_summary(client, extraction_id, user=USER):
    with _patch_decode_token(user):
        return client.get(f"/extractions/{extraction_id}/summary")


def _make_extraction(
    user_id=USER,
    album_id=ALBUM,
    config_file=None,
    feature_names=(),
    modalities=(),
    rois=(),
    study_uids=(),
    training_patients=None,
    test_patients=None,
    data_splitting_type=None,
    train_test_split_type=None,
    created_at=None,
    result_id=None,
):
    """Create a FeatureExtraction with its associations; return its id."""
    from quantimage2_backend_common.models import (
        ROI,
        FeatureDefinition,
        FeatureExtraction,
        FeatureExtractionTask,
        Modality,
        db,
    )

    extraction = FeatureExtraction(user_id=user_id, album_id=album_id)
    extraction.config_file = config_file
    extraction.result_id = result_id
    extraction.training_patients = training_patients
    extraction.test_patients = test_patients
    if data_splitting_type is not None:
        extraction.data_splitting_type = data_splitting_type
    if train_test_split_type is not None:
        extraction.train_test_split_type = train_test_split_type
    if created_at is not None:
        extraction.created_at = created_at
    extraction.save_to_db()

    for name in modalities:
        modality = Modality.query.filter_by(name=name).one_or_none() or Modality(name)
        extraction.modalities.append(modality)
    for name in rois:
        roi = ROI.query.filter_by(name=name).one_or_none() or ROI(name)
        extraction.rois.append(roi)
    for name in feature_names:
        extraction.feature_definitions.append(FeatureDefinition(name))
    db.session.commit()

    for i, study_uid in enumerate(study_uids):
        FeatureExtractionTask(extraction.id, study_uid, f"celery-task-{i}").save_to_db()

    return extraction.id


@pytest.fixture
def configs_base(tmp_path, monkeypatch):
    """Point the webapp config at a temporary extractions base dir; return the
    configs root (<base>/<CONFIGS_SUBDIR>)."""
    import config

    base = tmp_path / "extractions"
    monkeypatch.setattr(config, "EXTRACTIONS_BASE_DIR", str(base))
    configs_root = base / config.CONFIGS_SUBDIR
    configs_root.mkdir(parents=True)
    return configs_root


def _write_config(configs_root, user_id, text, name="extraction-1-config.yaml"):
    folder = configs_root / user_id / ALBUM
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text)
    return str(path)


def _assert_no_leaks(response, *paths):
    """The response must not carry patient/study IDs or any file path."""
    text = response.get_data(as_text=True)
    for secret in SECRET_PATIENTS_TRAIN + SECRET_PATIENTS_TEST + SECRET_STUDIES:
        assert secret not in text
    for path in paths:
        if path:
            assert str(path) not in text
            assert os.path.basename(str(path)) not in text
    assert "config_file" not in text
    assert ".yaml" not in text


# ===========================================================================
# Pure helper: parse_extraction_config
# ===========================================================================


@pytest.mark.unit
class TestParseExtractionConfigPresets:
    def test_petct_preset_uses_common_resampler_and_no_normalization(self):
        result = _svc().parse_extraction_config(_yaml(PETCT_YAML))

        assert result == EXPECTED_PETCT_CONFIG

    def test_mri_preset_uses_common_resampler_and_standardizer(self):
        result = _svc().parse_extraction_config(_yaml(MRI_YAML))

        mr_extractor = _pyrad_extractor({"binWidth": 0.3, "voxelArrayShift": 10})
        normalization = {"type": "standardizer", "params": {"threshold": 0.001}}
        assert result == {
            "status": "ok",
            "general": {
                "padding": 5,
                "submodalities": True,
                "combine_segmentation": False,
            },
            "modalities": [
                {
                    "modality": "MR_T1",
                    "resampling": LINEAR_1MM,
                    "normalization": normalization,
                    "extractors": [mr_extractor],
                },
                {
                    "modality": "MR_T2",
                    "resampling": LINEAR_1MM,
                    "normalization": normalization,
                    "extractors": [mr_extractor],
                },
            ],
        }

    def test_settings_only_report_the_listed_pyradiomics_keys_that_are_present(self):
        config = _yaml(PETCT_YAML)
        config["feature_extraction"]["CT"]["pyradiomics"]["original"]["setting"] = {
            "binCount": 64,
            "voxelArrayShift": 300,
            "resampledPixelSpacing": [2, 2, 2],
            "normalize": True,
            "normalizeScale": 100,
            "label": 1,
            "interpolator": "sitkBSpline",
            "geometryTolerance": 0.0001,
        }

        result = _svc().parse_extraction_config(config)

        ct = result["modalities"][0]
        assert ct["modality"] == "CT"
        assert ct["extractors"][0]["settings"] == {
            "binCount": 64,
            "voxelArrayShift": 300,
            "resampledPixelSpacing": [2, 2, 2],
            "normalize": True,
            "normalizeScale": 100,
        }

    def test_image_types_and_feature_classes_are_the_mapping_keys_in_order(self):
        config = _yaml(PETCT_YAML)
        original = config["feature_extraction"]["CT"]["pyradiomics"]["original"]
        original["imageType"] = {
            "Original": {},
            "Wavelet": {},
            "LoG": {"sigma": [1.0, 2.0]},
        }
        original["featureClass"] = {"firstorder": None, "glcm": ["Contrast"]}

        result = _svc().parse_extraction_config(config)

        extractor = result["modalities"][0]["extractors"][0]
        assert extractor["image_types"] == ["Original", "Wavelet", "LoG"]
        assert extractor["feature_classes"] == ["firstorder", "glcm"]

    def test_several_pyradiomics_extractors_per_modality_are_all_listed(self):
        config = _yaml(PETCT_YAML)
        ct = config["feature_extraction"]["CT"]["pyradiomics"]
        ct["wavelet"] = {
            "imageType": {"Wavelet": {}},
            "featureClass": {"glcm": None},
            "setting": {"binCount": 32},
        }

        result = _svc().parse_extraction_config(config)

        extractors = result["modalities"][0]["extractors"]
        assert [(e["backend"], e["name"]) for e in extractors] == [
            ("pyradiomics", "original"),
            ("pyradiomics", "wavelet"),
        ]
        assert extractors[1]["image_types"] == ["Wavelet"]
        assert extractors[1]["feature_classes"] == ["glcm"]
        assert extractors[1]["settings"] == {"binCount": 32}


@pytest.mark.unit
class TestParseExtractionConfigPrecedence:
    def test_resampler_is_own_section_else_default_merged_over_common(self):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {
                "common": {
                    "bspline_resampler": {
                        "resampling_spacing": [2.0, 2.0, 2.0],
                        "order": 3,
                    }
                },
                "default": {
                    "bspline_resampler": {
                        "resampling_spacing": [4.0, 4.0, 4.0],
                        "order": 0,
                    }
                },
                "CT": {
                    "bspline_resampler": {
                        "resampling_spacing": [0.5, 0.5, 1.5],
                        "order": 1,
                    }
                },
            },
            "feature_extraction": {"CT": None, "PT": None},
        }

        result = _svc().parse_extraction_config(config)

        by_modality = {m["modality"]: m for m in result["modalities"]}
        assert by_modality["CT"]["resampling"] == {
            "spacing": [0.5, 0.5, 1.5],
            "order": 1,
            "interpolation": "linear",
        }
        # PT has no section of its own -> default, merged over common
        assert by_modality["PT"]["resampling"] == {
            "spacing": [4.0, 4.0, 4.0],
            "order": 0,
            "interpolation": "nearest",
        }

    def test_default_resampler_is_used_when_neither_modality_nor_common_has_one(self):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {
                "common": {"standardizer": {"threshold": 0.01}},
                "default": {
                    "bspline_resampler": {
                        "resampling_spacing": [4.0, 4.0, 4.0],
                        "order": 0,
                    }
                },
            },
            "feature_extraction": {"MR": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["modalities"][0]["modality"] == "MR"
        assert result["modalities"][0]["resampling"] == {
            "spacing": [4.0, 4.0, 4.0],
            "order": 0,
            "interpolation": "nearest",
        }

    def test_no_resampler_anywhere_gives_null_resampling(self):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {"common": None, "default": None},
            "feature_extraction": {"CT": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["modalities"][0]["resampling"] is None
        assert result["modalities"][0]["normalization"] is None

    def test_normalization_is_selected_sections_else_common(self):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {
                "common": {"masked_standardizer": {"threshold": 0.5}},
                "default": {"combined_standardizer": {"threshold": 0.9}},
                "CT": {"standardizer": {"threshold": 0.001}},
                "PT": {"combined_standardizer": {"threshold": 0.2}},
            },
            "feature_extraction": {"CT": None, "PT": None, "MR": None},
        }

        result = _svc().parse_extraction_config(config)

        by_modality = {m["modality"]: m for m in result["modalities"]}
        assert by_modality["CT"]["normalization"] == {
            "type": "standardizer",
            "params": {"threshold": 0.001},
        }
        # The modality's own normalizer wins over common's, whatever its type
        assert by_modality["PT"]["normalization"] == {
            "type": "combined_standardizer",
            "params": {"threshold": 0.2},
        }
        # No MR section -> default is the selected section, and its
        # normalizer wins over common's
        assert by_modality["MR"]["normalization"] == {
            "type": "combined_standardizer",
            "params": {"threshold": 0.9},
        }

    @pytest.mark.parametrize(
        "level_config, expected_type",
        [
            (
                {
                    "combined_standardizer": {"threshold": 0.3},
                    "masked_standardizer": {"threshold": 0.2},
                    "standardizer": {"threshold": 0.1},
                },
                "standardizer",
            ),
            (
                {
                    "combined_standardizer": {"threshold": 0.3},
                    "masked_standardizer": {"threshold": 0.2},
                },
                "masked_standardizer",
            ),
        ],
    )
    @pytest.mark.parametrize("level", ["CT", "common", "default"])
    def test_normalization_type_order_within_one_level(
        self, level, level_config, expected_type
    ):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {level: level_config},
            "feature_extraction": {"CT": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["modalities"][0]["normalization"] == {
            "type": expected_type,
            "params": level_config[expected_type],
        }

    def test_normalization_falls_back_to_default(self):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {
                "default": {"combined_standardizer": {"threshold": 0.9}},
            },
            "feature_extraction": {"MR": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["modalities"][0]["normalization"] == {
            "type": "combined_standardizer",
            "params": {"threshold": 0.9},
        }

    def test_configured_modalities_are_union_of_both_sections_sorted(self):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {
                "common": None,
                "default": None,
                "PT": {
                    "bspline_resampler": {
                        "resampling_spacing": [2.0, 2.0, 2.0],
                        "order": 1,
                    }
                },
                "MR_T2": None,
            },
            "feature_extraction": {"CT": None, "PT": None},
        }

        result = _svc().parse_extraction_config(config)

        assert [m["modality"] for m in result["modalities"]] == ["CT", "MR_T2", "PT"]


def _resampler(spacing=None, order=None):
    params = {}
    if spacing is not None:
        params["resampling_spacing"] = spacing
    if order is not None:
        params["order"] = order
    return {"bspline_resampler": params}


def _pyrad_section(name, feature_class, bin_width):
    return {
        "pyradiomics": {
            name: {
                "imageType": {"Original": {}},
                "featureClass": {feature_class: None},
                "setting": {"binWidth": bin_width},
            }
        }
    }


def _pyrad_row_extractor(name, feature_class, bin_width):
    return {
        "backend": "pyradiomics",
        "name": name,
        "image_types": ["Original"],
        "feature_classes": [feature_class],
        "settings": {"binWidth": bin_width},
    }


@pytest.mark.unit
class TestOkapyResolution:
    """Clarification 10: section selection is key > modality prefix > default,
    common is merged/prepended, rows come from the extraction's modalities."""

    def test_modality_prefix_falls_back_to_prefix_section(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {
                "default": {
                    **_resampler([4.0, 4.0, 4.0], 0),
                    "combined_standardizer": {"threshold": 0.9},
                },
                "MR": {
                    **_resampler([0.8, 0.8, 3.0], 1),
                    "standardizer": {"threshold": 0.01},
                },
                "MR_T2": _resampler([0.5, 0.5, 0.5], 3),
            },
            "feature_extraction": {
                "default": _pyrad_section("fallback", "shape", 1),
                "MR": _pyrad_section("mr", "glcm", 5),
            },
        }

        result = _svc().parse_extraction_config(
            config, modalities=["MR_T2", "MR_T1Pre"]
        )

        assert result["status"] == "ok"
        assert result["modalities"] == [
            {
                "modality": "MR_T1Pre",
                "resampling": {
                    "spacing": [0.8, 0.8, 3.0],
                    "order": 1,
                    "interpolation": "linear",
                },
                "normalization": {
                    "type": "standardizer",
                    "params": {"threshold": 0.01},
                },
                "extractors": [_pyrad_row_extractor("mr", "glcm", 5)],
            },
            {
                # Exact key beats the prefix for volume_preprocessing; no
                # MR_T2 feature_extraction key -> MR prefix section
                "modality": "MR_T2",
                "resampling": {
                    "spacing": [0.5, 0.5, 0.5],
                    "order": 3,
                    "interpolation": "cubic",
                },
                "normalization": None,
                "extractors": [_pyrad_row_extractor("mr", "glcm", 5)],
            },
        ]

    def test_key_with_null_value_suppresses_default(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {
                "CT": None,
                "default": {
                    **_resampler([4.0, 4.0, 4.0], 0),
                    "combined_standardizer": {"threshold": 0.9},
                },
            },
            "feature_extraction": {
                "CT": None,
                "default": _pyrad_section("fallback", "shape", 1),
            },
        }

        result = _svc().parse_extraction_config(config, modalities=["CT"])

        assert result["modalities"] == [
            {
                "modality": "CT",
                "resampling": None,
                "normalization": None,
                "extractors": [],
            }
        ]

    def test_prefix_key_with_null_value_suppresses_default(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {
                "MR": None,
                "default": _resampler([4.0, 4.0, 4.0], 0),
            },
            "feature_extraction": {
                "MR": None,
                "default": _pyrad_section("fallback", "shape", 1),
            },
        }

        result = _svc().parse_extraction_config(config, modalities=["MR_T1"])

        [row] = result["modalities"]
        assert row["modality"] == "MR_T1"
        assert row["resampling"] is None
        assert row["extractors"] == []

    def test_resampler_is_deep_merged_over_common(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {
                "common": _resampler([1.0, 1.0, 1.0], 3),
                "CT": _resampler(spacing=[2.0, 2.0, 2.0]),
                "PT": _resampler(order=0),
            },
            "feature_extraction": {"CT": None, "PT": None},
        }

        result = _svc().parse_extraction_config(config, modalities=["CT", "PT"])

        by_modality = {m["modality"]: m for m in result["modalities"]}
        assert by_modality["CT"]["resampling"] == {
            "spacing": [2.0, 2.0, 2.0],
            "order": 3,
            "interpolation": "cubic",
        }
        assert by_modality["PT"]["resampling"] == {
            "spacing": [1.0, 1.0, 1.0],
            "order": 0,
            "interpolation": "nearest",
        }

    def test_resampler_only_in_selected_section_is_used_as_is(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {
                "common": {"standardizer": {"threshold": 0.1}},
                "CT": _resampler([3.0, 3.0, 3.0], 1),
            },
            "feature_extraction": {"CT": None},
        }

        result = _svc().parse_extraction_config(config, modalities=["CT"])

        assert result["modalities"][0]["resampling"] == {
            "spacing": [3.0, 3.0, 3.0],
            "order": 1,
            "interpolation": "linear",
        }

    def test_selected_normalizer_wins_over_common_else_common_is_used(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {
                "common": {"standardizer": {"threshold": 0.001}},
                "CT": {"masked_standardizer": {"threshold": 0.5}},
                "PT": _resampler([2.0, 2.0, 2.0], 1),
            },
            "feature_extraction": {"CT": None, "PT": None},
        }

        result = _svc().parse_extraction_config(config, modalities=["CT", "PT", "MR"])

        by_modality = {m["modality"]: m for m in result["modalities"]}
        assert by_modality["CT"]["normalization"] == {
            "type": "masked_standardizer",
            "params": {"threshold": 0.5},
        }
        # PT's section has no normalizer -> common's
        assert by_modality["PT"]["normalization"] == {
            "type": "standardizer",
            "params": {"threshold": 0.001},
        }
        # MR: no section, no default -> common's
        assert by_modality["MR"]["normalization"] == {
            "type": "standardizer",
            "params": {"threshold": 0.001},
        }

    def test_common_extractors_come_first_then_selected_sections(self):
        config = {
            "general": {"padding": 5},
            "feature_extraction": {
                "common": _pyrad_section("shared", "shape", 1),
                "CT": _pyrad_section("ct", "glcm", 25),
                "default": _pyrad_section("fallback", "firstorder", 3),
            },
        }

        result = _svc().parse_extraction_config(config, modalities=["CT", "PT"])

        by_modality = {m["modality"]: m for m in result["modalities"]}
        assert by_modality["CT"]["extractors"] == [
            _pyrad_row_extractor("shared", "shape", 1),
            _pyrad_row_extractor("ct", "glcm", 25),
        ]
        assert by_modality["PT"]["extractors"] == [
            _pyrad_row_extractor("shared", "shape", 1),
            _pyrad_row_extractor("fallback", "firstorder", 3),
        ]

    def test_modality_without_feature_section_gets_default_extractors(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {"common": _resampler([1.0, 1.0, 1.0], 1)},
            "feature_extraction": {
                "CT": _pyrad_section("ct", "glcm", 25),
                "default": _pyrad_section("fallback", "firstorder", 3),
            },
        }

        result = _svc().parse_extraction_config(config, modalities=["PT"])

        assert result["modalities"] == [
            {
                "modality": "PT",
                "resampling": LINEAR_1MM,
                "normalization": None,
                "extractors": [_pyrad_row_extractor("fallback", "firstorder", 3)],
            }
        ]

    def test_rows_come_from_the_modalities_argument(self):
        result = _svc().parse_extraction_config(_yaml(PETCT_YAML), modalities=["CT"])

        assert result == {
            "status": "ok",
            "general": EXPECTED_PETCT_CONFIG["general"],
            "modalities": [EXPECTED_PETCT_CONFIG["modalities"][0]],
        }

    def test_rows_from_modalities_argument_are_sorted(self):
        result = _svc().parse_extraction_config(
            _yaml(PETCT_YAML), modalities=["PT", "CT"]
        )

        assert result == EXPECTED_PETCT_CONFIG

    def test_extracted_modality_missing_from_yaml_still_gets_a_row(self):
        result = _svc().parse_extraction_config(
            _yaml(PETCT_YAML), modalities=["MR", "CT"]
        )

        assert [m["modality"] for m in result["modalities"]] == ["CT", "MR"]
        mr = result["modalities"][1]
        # default is null in the preset -> only common's resampler, no
        # extractors
        assert mr == {
            "modality": "MR",
            "resampling": LINEAR_1MM,
            "normalization": None,
            "extractors": [],
        }

    @pytest.mark.parametrize("modalities", [None, []])
    def test_no_modalities_falls_back_to_yaml_keys(self, modalities):
        result = _svc().parse_extraction_config(
            _yaml(PETCT_YAML), modalities=modalities
        )

        assert result == EXPECTED_PETCT_CONFIG

    def test_modalities_argument_does_not_change_non_ok_shape(self):
        assert _svc().parse_extraction_config(None, modalities=["CT"]) == {
            "status": "invalid",
            **NOT_OK_CONFIG_SHAPE,
        }
        assert _svc().parse_extraction_config(
            {"general": {"padding": 5}}, modalities=["CT"]
        ) == {"status": "unrecognized", **NOT_OK_CONFIG_SHAPE}

    @pytest.mark.parametrize(
        "config",
        [
            {"volume_preprocessing": {"common": _resampler([1.0, 1.0, 1.0], 1)}},
            {"volume_preprocessing": {"default": {}}},
            {"feature_extraction": {"default": None}},
            {"feature_extraction": {"common": _pyrad_section("s", "shape", 1)}},
        ],
    )
    def test_only_common_or_default_sections_is_ok(self, config):
        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["modalities"] == []

    def test_only_common_section_with_modalities_gives_rows(self):
        config = {"volume_preprocessing": {"common": _resampler([1.0, 1.0, 1.0], 1)}}

        result = _svc().parse_extraction_config(config, modalities=["CT"])

        assert result["status"] == "ok"
        assert result["modalities"] == [
            {
                "modality": "CT",
                "resampling": LINEAR_1MM,
                "normalization": None,
                "extractors": [],
            }
        ]

    @pytest.mark.parametrize(
        "config",
        [
            {"volume_preprocessing": {}, "feature_extraction": {}},
            {"volume_preprocessing": None, "feature_extraction": None},
            {"volume_preprocessing": {"CT": "resampler"}},
            {"feature_extraction": {"CT": ["pyradiomics"], "common": 3}},
        ],
    )
    def test_sections_without_mapping_or_null_values_are_unrecognized(self, config):
        result = _svc().parse_extraction_config(config)

        assert result == {"status": "unrecognized", **NOT_OK_CONFIG_SHAPE}


@pytest.mark.unit
class TestInterpolationLabel:
    @pytest.mark.parametrize(
        "order, label",
        [
            (0, "nearest"),
            (1, "linear"),
            (3, "cubic"),
            (2, "bspline order 2"),
            (5, "bspline order 5"),
        ],
    )
    def test_order_maps_to_interpolation_label(self, order, label):
        config = {
            "general": {"padding": 10},
            "volume_preprocessing": {
                "common": {
                    "bspline_resampler": {
                        "resampling_spacing": [1.0, 1.0, 1.0],
                        "order": order,
                    }
                }
            },
            "feature_extraction": {"CT": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["modalities"][0]["resampling"] == {
            "spacing": [1.0, 1.0, 1.0],
            "order": order,
            "interpolation": label,
        }


@pytest.mark.unit
class TestParseExtractionConfigRobustness:
    @pytest.mark.parametrize("value", [None, [], ["a", "b"], "just a string", 42])
    def test_non_mapping_is_invalid(self, value):
        result = _svc().parse_extraction_config(value)

        assert result == {"status": "invalid", **NOT_OK_CONFIG_SHAPE}

    @pytest.mark.parametrize(
        "value",
        [
            {},
            {"general": {"padding": 5}},
            {
                "general": {"padding": 5},
                "geometry_preprocessing": {"resampling": {"spacing": [1, 1, 1]}},
                "extraction": {"CT": {}},
            },
        ],
    )
    def test_mapping_without_known_sections_is_unrecognized(self, value):
        result = _svc().parse_extraction_config(value)

        assert result == {"status": "unrecognized", **NOT_OK_CONFIG_SHAPE}

    def test_padding_whole_image_is_reported_verbatim(self):
        config = _yaml(PETCT_YAML)
        config["general"]["padding"] = "whole_image"

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["general"] == {
            "padding": "whole_image",
            "submodalities": False,
            "combine_segmentation": True,
        }

    def test_null_sections_do_not_crash(self):
        config = {
            "general": None,
            "volume_preprocessing": None,
            "feature_extraction": {"CT": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["general"] is None
        assert result["modalities"] == [
            {
                "modality": "CT",
                "resampling": None,
                "normalization": None,
                "extractors": [],
            }
        ]

    def test_missing_general_keys_are_null(self):
        config = _yaml(PETCT_YAML)
        config["general"] = {"padding": 3}

        result = _svc().parse_extraction_config(config)

        assert result["general"] == {
            "padding": 3,
            "submodalities": None,
            "combine_segmentation": None,
        }

    def test_null_pyradiomics_subsections_give_empty_lists(self):
        config = {
            "general": {"padding": 5},
            "feature_extraction": {
                "CT": {
                    "pyradiomics": {
                        "original": {
                            "imageType": None,
                            "featureClass": None,
                            "setting": None,
                        }
                    }
                }
            },
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["modalities"][0]["extractors"] == [
            {
                "backend": "pyradiomics",
                "name": "original",
                "image_types": [],
                "feature_classes": [],
                "settings": {},
            }
        ]

    def test_null_pyradiomics_extractor_body_does_not_crash(self):
        config = {
            "general": {"padding": 5},
            "feature_extraction": {"CT": {"pyradiomics": {"original": None}}},
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["modalities"][0]["modality"] == "CT"

    def test_resampler_without_spacing_or_order_does_not_crash(self):
        config = {
            "general": {"padding": 5},
            "volume_preprocessing": {"common": {"bspline_resampler": None}},
            "feature_extraction": {"CT": None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert result["modalities"][0]["modality"] == "CT"

    def test_non_pyradiomics_extractor_is_listed_with_empty_settings(self):
        config = _yaml(PETCT_YAML)
        config["feature_extraction"]["CT"]["riesz"] = {
            "extractor0": {"RieszOrder": 1, "RieszScales": 4}
        }

        result = _svc().parse_extraction_config(config)

        ct = result["modalities"][0]
        assert ct["modality"] == "CT"
        backends = [e["backend"] for e in ct["extractors"]]
        assert sorted(backends) == ["pyradiomics", "riesz"]
        riesz = next(e for e in ct["extractors"] if e["backend"] == "riesz")
        assert riesz == {
            "backend": "riesz",
            "name": "riesz",
            "image_types": [],
            "feature_classes": [],
            "settings": {},
        }
        # The pyradiomics extractor is still fully reported
        pyrad = next(e for e in ct["extractors"] if e["backend"] == "pyradiomics")
        assert pyrad["settings"] == {"binWidth": 20, "voxelArrayShift": 1000}

    @pytest.mark.parametrize(
        "config",
        [
            {"volume_preprocessing": ["not", "a", "mapping"]},
            {"feature_extraction": "CT"},
            {"feature_extraction": {"CT": ["pyradiomics"]}},
            {"feature_extraction": {"CT": {"pyradiomics": ["original"]}}},
            {"volume_preprocessing": {"CT": "bspline_resampler"}},
            {
                "volume_preprocessing": {
                    "CT": {"bspline_resampler": {"order": "one"}},
                },
            },
            {"general": "padding", "feature_extraction": {"CT": None}},
            {"feature_extraction": {"CT": {"pyradiomics": {"o": {"setting": [1]}}}}},
        ],
    )
    def test_malformed_sections_never_raise(self, config):
        result = _svc().parse_extraction_config(config)

        assert isinstance(result, dict)
        assert result["status"] in ("ok", "invalid", "unrecognized")
        assert isinstance(result["modalities"], list)


# ===========================================================================
# Pure helper: summarize_feature_names
# ===========================================================================


@pytest.mark.unit
class TestSummarizeFeatureNames:
    def test_families_from_pyradiomics_riesz_zrad_and_other(self):
        names = [
            "original_shape_Elongation",
            "original_shape_Sphericity",
            "original_firstorder_Mean",
            "log-sigma-1-0-mm-3D_firstorder_Mean",
            "wavelet-LLH_glcm_Contrast",
            "wavelet-HHH_glcm_Contrast",
            "original_glcm_Idm",
            "tex_sig1_energy",
            "tex_sig2_energy",
            "zrad_glcm_Contrast",
            "PatientAge",
        ]

        result = _svc().summarize_feature_names(names)

        assert result["total"] == 11
        assert result["by_family"] == {
            "shape": 2,
            "firstorder": 2,
            "glcm": 3,
            "riesz": 2,
            "zrad": 1,
            "other": 1,
        }
        assert result["filters"] == {"original": 4, "log": 1, "wavelet": 2}
        assert "tex" not in result["filters"]
        assert "zrad" not in result["filters"]

    def test_filters_cut_first_token_at_first_hyphen(self):
        names = [
            "original_shape_Elongation",
            "original_firstorder_Mean",
            "wavelet-LLH_glcm_Contrast",
            "wavelet-HHH_glcm_Contrast",
            "wavelet-LLL_firstorder_Mean",
            "log-sigma-1-0-mm-3D_firstorder_Mean",
            "log-sigma-3-0-mm-3D_glcm_Idm",
            "gradient_glrlm_RunEntropy",
            "squareroot_firstorder_Mean",
            "logarithm_firstorder_Mean",
        ]

        result = _svc().summarize_feature_names(names)

        assert result["total"] == 10
        assert result["filters"] == {
            "original": 2,
            "wavelet": 3,
            "log": 2,
            "gradient": 1,
            "squareroot": 1,
            "logarithm": 1,
        }

    def test_filters_ignore_names_without_known_prefix(self):
        names = [
            "original_firstorder_Mean",
            "PatientAge",
            "lbp-3D-k_firstorder_Mean",
            "custom_glcm_Contrast",
        ]

        result = _svc().summarize_feature_names(names)

        assert result["filters"] == {"original": 1}
        assert result["total"] == 4
        assert result["by_family"] == {"firstorder": 1, "other": 3}

    def test_total_counts_unique_names(self):
        names = [
            "original_firstorder_Mean",
            "original_firstorder_Mean",
            "original_shape_Elongation",
        ]

        result = _svc().summarize_feature_names(names)

        assert result["total"] == 2
        assert result["by_family"] == {"firstorder": 1, "shape": 1}
        assert result["filters"] == {"original": 2}

    def test_duplicates_count_once_in_families_and_filters(self):
        names = [
            "wavelet-LLH_glcm_Contrast",
            "wavelet-LLH_glcm_Contrast",
            "wavelet-LLH_glcm_Contrast",
            "tex_sig1_energy",
            "tex_sig1_energy",
            "PatientAge",
            "PatientAge",
        ]

        result = _svc().summarize_feature_names(names)

        assert result == {
            "total": 3,
            "by_family": {"glcm": 1, "riesz": 1, "other": 1},
            "filters": {"wavelet": 1},
        }

    def test_tex_and_zrad_are_never_filters(self):
        names = ["tex_sig1_energy", "zrad_glcm_Contrast", "zrad-x_firstorder_Mean"]

        result = _svc().summarize_feature_names(names)

        assert result["filters"] == {}
        assert result["by_family"] == {"riesz": 1, "zrad": 2}

    def test_empty_list(self):
        result = _svc().summarize_feature_names([])

        assert result == {"total": 0, "by_family": {}, "filters": {}}


# ===========================================================================
# Route: GET /extractions/<id>/summary
# ===========================================================================

FEATURE_NAMES = [
    "original_shape_Elongation",
    "original_shape_Sphericity",
    "original_firstorder_Mean",
    "original_firstorder_Entropy",
    "original_firstorder_Skewness",
    "wavelet-LLH_glcm_Contrast",
    "log-sigma-1-0-mm-3D_firstorder_Mean",
    "tex_sig1_energy",
]


@pytest.mark.integration
class TestSummaryRouteHappyPath:
    def test_full_summary_for_owned_extraction(
        self, client, app, db_session, configs_base
    ):
        config_path = _write_config(configs_base, USER, PETCT_YAML)
        with app.app_context():
            extraction_id = _make_extraction(
                config_file=config_path,
                feature_names=FEATURE_NAMES,
                modalities=["PT", "CT"],
                rois=["GTV_T", "GTV_L", "GTV-N"],
                study_uids=SECRET_STUDIES,
                training_patients=SECRET_PATIENTS_TRAIN,
                test_patients=SECRET_PATIENTS_TEST,
                data_splitting_type="traintest",
                train_test_split_type="manual",
                created_at=datetime.datetime(2026, 10, 1, 10, 0, 0),
            )

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        body = response.get_json()
        assert set(body) == {
            "extraction_id",
            "created_at",
            "config",
            "features",
            "dataset",
        }
        assert body["extraction_id"] == extraction_id
        assert body["created_at"] == "2026-10-01T10:00:00"
        assert body["config"] == EXPECTED_PETCT_CONFIG
        assert body["features"] == {
            "total": 8,
            "by_family": {"shape": 2, "firstorder": 4, "glcm": 1, "riesz": 1},
            "filters": {"original": 5, "wavelet": 1, "log": 1},
            "modalities": ["CT", "PT"],
            "rois": ["GTV-N", "GTV_L", "GTV_T"],
        }
        assert body["dataset"] == {
            "studies": 3,
            "data_splitting_type": "traintest",
            "train_test_split_type": "manual",
            "training_patients": 3,
            "test_patients": 2,
        }
        _assert_no_leaks(response, config_path, configs_base)

    def test_mri_config_through_route(self, client, app, db_session, configs_base):
        config_path = _write_config(configs_base, USER, MRI_YAML)
        with app.app_context():
            extraction_id = _make_extraction(config_file=config_path)

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        config = response.get_json()["config"]
        assert config["status"] == "ok"
        assert [m["modality"] for m in config["modalities"]] == ["MR_T1", "MR_T2"]
        assert config["modalities"][0]["normalization"] == {
            "type": "standardizer",
            "params": {"threshold": 0.001},
        }

    def test_config_dir_is_read_at_call_time(
        self, client, app, db_session, tmp_path, monkeypatch
    ):
        """Changing config.EXTRACTIONS_BASE_DIR after import must be honoured."""
        import config

        base = tmp_path / "other-base"
        folder = base / config.CONFIGS_SUBDIR / USER / ALBUM
        folder.mkdir(parents=True)
        path = folder / "extraction-7-config.yaml"
        path.write_text(PETCT_YAML)
        monkeypatch.setattr(config, "EXTRACTIONS_BASE_DIR", str(base))
        with app.app_context():
            extraction_id = _make_extraction(config_file=str(path))

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        assert response.get_json()["config"]["status"] == "ok"

    def test_empty_extraction_reports_zero_counts(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction()

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        body = response.get_json()
        assert body["features"] == {
            "total": 0,
            "by_family": {},
            "filters": {},
            "modalities": [],
            "rois": [],
        }
        assert body["dataset"]["studies"] == 0

    def test_counts_only_this_extractions_rows(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            _make_extraction(
                feature_names=["original_glszm_ZoneEntropy", "original_ngtdm_Busyness"],
                modalities=["MR"],
                rois=["Liver"],
                study_uids=["1.2.3.other-1", "1.2.3.other-2"],
            )
            extraction_id = _make_extraction(
                feature_names=["original_shape_Elongation"],
                modalities=["CT"],
                rois=["GTV_T"],
                study_uids=["1.2.3.mine"],
            )

        response = _get_summary(client, extraction_id)

        body = response.get_json()
        assert response.status_code == 200
        assert body["features"]["total"] == 1
        assert body["features"]["by_family"] == {"shape": 1}
        assert body["features"]["modalities"] == ["CT"]
        assert body["features"]["rois"] == ["GTV_T"]
        assert body["dataset"]["studies"] == 1


@pytest.mark.integration
class TestSummaryRouteDatasetSplit:
    def test_unsaved_split_gives_null_counts(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction(study_uids=SECRET_STUDIES[:2])

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        assert response.get_json()["dataset"] == {
            "studies": 2,
            "data_splitting_type": "traintest",
            "train_test_split_type": "automatic",
            "training_patients": None,
            "test_patients": None,
        }

    def test_saved_split_gives_counts_without_ids(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction(
                training_patients=SECRET_PATIENTS_TRAIN,
                test_patients=SECRET_PATIENTS_TEST,
            )

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        dataset = response.get_json()["dataset"]
        assert dataset["training_patients"] == 3
        assert dataset["test_patients"] == 2
        _assert_no_leaks(response)

    def test_empty_split_lists_count_as_saved_with_zero(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction(training_patients=[], test_patients=[])

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        dataset = response.get_json()["dataset"]
        assert dataset["training_patients"] == 0
        assert dataset["test_patients"] == 0

    def test_full_dataset_splitting_type_is_reported(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction(
                data_splitting_type="fulldataset",
                training_patients=SECRET_PATIENTS_TRAIN + SECRET_PATIENTS_TEST,
            )

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        dataset = response.get_json()["dataset"]
        assert dataset["data_splitting_type"] == "fulldataset"
        assert dataset["training_patients"] == 5
        assert dataset["test_patients"] is None
        _assert_no_leaks(response)


@pytest.mark.integration
class TestSummaryRouteConfigStatus:
    def _summary_config(self, client, app, config_file):
        with app.app_context():
            extraction_id = _make_extraction(config_file=config_file)
        response = _get_summary(client, extraction_id)
        assert response.status_code == 200
        return response, response.get_json()["config"]

    def test_null_config_file_is_missing(self, client, app, db_session, configs_base):
        _, config = self._summary_config(client, app, None)

        assert config == {"status": "missing", **NOT_OK_CONFIG_SHAPE}

    def test_nonexistent_file_is_missing(self, client, app, db_session, configs_base):
        path = configs_base / USER / ALBUM / "extraction-404-config.yaml"

        response, config = self._summary_config(client, app, str(path))

        assert config == {"status": "missing", **NOT_OK_CONFIG_SHAPE}
        _assert_no_leaks(response, path)

    def test_config_file_pointing_to_a_directory_is_missing(
        self, client, app, db_session, configs_base
    ):
        folder = configs_base / USER / ALBUM
        folder.mkdir(parents=True)

        response, config = self._summary_config(client, app, str(folder))

        assert config == {"status": "missing", **NOT_OK_CONFIG_SHAPE}

    def test_invalid_yaml_is_invalid(self, client, app, db_session, configs_base):
        path = _write_config(
            configs_base, USER, "general: [unclosed\n  padding: : 5\n\t- x"
        )

        response, config = self._summary_config(client, app, path)

        assert config == {"status": "invalid", **NOT_OK_CONFIG_SHAPE}
        _assert_no_leaks(response, path)

    def test_yaml_list_is_invalid(self, client, app, db_session, configs_base):
        path = _write_config(configs_base, USER, "- general\n- feature_extraction\n")

        _, config = self._summary_config(client, app, path)

        assert config == {"status": "invalid", **NOT_OK_CONFIG_SHAPE}

    def test_empty_yaml_is_invalid(self, client, app, db_session, configs_base):
        path = _write_config(configs_base, USER, "")

        _, config = self._summary_config(client, app, path)

        assert config == {"status": "invalid", **NOT_OK_CONFIG_SHAPE}

    def test_unsafe_yaml_tag_is_not_executed(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(
            configs_base,
            USER,
            "general: !!python/object/apply:os.system ['echo pwned']\n",
        )

        _, config = self._summary_config(client, app, path)

        assert config == {"status": "invalid", **NOT_OK_CONFIG_SHAPE}

    def test_new_okapy_schema_is_unrecognized(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(
            configs_base,
            USER,
            "general:\n  padding: 5\n"
            "geometry_preprocessing:\n  resampling:\n    spacing: [1, 1, 1]\n",
        )

        _, config = self._summary_config(client, app, path)

        assert config == {"status": "unrecognized", **NOT_OK_CONFIG_SHAPE}

    def test_whole_image_padding_through_route(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(
            configs_base,
            USER,
            PETCT_YAML.replace("padding: 5", 'padding: "whole_image"'),
        )

        _, config = self._summary_config(client, app, path)

        assert config["status"] == "ok"
        assert config["general"]["padding"] == "whole_image"


@pytest.mark.integration
class TestSummaryRouteConfigPathSafety:
    """config_file is untrusted: only files under the owner's config dir."""

    def _assert_missing(self, client, app, config_file, *forbidden):
        with app.app_context():
            extraction_id = _make_extraction(config_file=config_file)
        response = _get_summary(client, extraction_id)
        assert response.status_code == 200
        assert response.get_json()["config"] == {
            "status": "missing",
            **NOT_OK_CONFIG_SHAPE,
        }
        text = response.get_data(as_text=True)
        for snippet in forbidden:
            assert snippet not in text
        _assert_no_leaks(response, config_file)

    def test_other_users_config_dir_is_refused(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(configs_base, OTHER_USER, PETCT_YAML)

        self._assert_missing(client, app, path, "voxelArrayShift", "binWidth")

    def test_etc_passwd_is_refused(self, client, app, db_session, configs_base):
        self._assert_missing(client, app, "/etc/passwd", "root:", "/bin/")

    def test_dotdot_traversal_is_refused(self, client, app, db_session, configs_base):
        _write_config(configs_base, OTHER_USER, PETCT_YAML)
        traversal = os.path.join(
            str(configs_base),
            USER,
            "..",
            OTHER_USER,
            ALBUM,
            "extraction-1-config.yaml",
        )

        self._assert_missing(client, app, traversal, "voxelArrayShift", "binWidth")

    def test_dotdot_traversal_to_etc_is_refused(
        self, client, app, db_session, configs_base
    ):
        traversal = os.path.join(
            str(configs_base), USER, *([".."] * 40), "etc", "passwd"
        )

        self._assert_missing(client, app, traversal, "root:")

    def test_sibling_dir_sharing_the_user_prefix_is_refused(
        self, client, app, db_session, configs_base
    ):
        # <configs>/<user>-evil/ starts with the string <configs>/<user>
        path = _write_config(configs_base, USER + "-evil", PETCT_YAML)

        self._assert_missing(client, app, path, "voxelArrayShift", "binWidth")

    def test_symlink_inside_user_dir_pointing_outside_is_refused(
        self, client, app, db_session, configs_base, tmp_path
    ):
        outside = tmp_path / "outside.yaml"
        outside.write_text(PETCT_YAML)
        folder = configs_base / USER / ALBUM
        folder.mkdir(parents=True)
        link = folder / "extraction-9-config.yaml"
        link.symlink_to(outside)

        self._assert_missing(client, app, str(link), "voxelArrayShift", "binWidth")

    def test_file_outside_base_dir_is_refused(
        self, client, app, db_session, configs_base, tmp_path
    ):
        outside = tmp_path / "elsewhere" / USER / "extraction-1-config.yaml"
        outside.parent.mkdir(parents=True)
        outside.write_text(PETCT_YAML)

        self._assert_missing(client, app, str(outside), "voxelArrayShift")

    def test_config_in_extractions_base_but_not_configs_subdir_is_refused(
        self, client, app, db_session, configs_base
    ):
        # <base>/<user>/... is under EXTRACTIONS_BASE_DIR but not under CONFIGS_SUBDIR
        wrong = configs_base.parent / USER / ALBUM / "extraction-1-config.yaml"
        wrong.parent.mkdir(parents=True)
        wrong.write_text(PETCT_YAML)

        self._assert_missing(client, app, str(wrong), "voxelArrayShift")

    def test_owner_config_is_still_read_via_normalised_path(
        self, client, app, db_session, configs_base
    ):
        """A `..` that stays inside the owner's dir is fine once resolved."""
        _write_config(configs_base, USER, PETCT_YAML)
        path = os.path.join(
            str(configs_base), USER, ALBUM, "..", ALBUM, "extraction-1-config.yaml"
        )
        with app.app_context():
            extraction_id = _make_extraction(config_file=path)

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        assert response.get_json()["config"] == EXPECTED_PETCT_CONFIG


@pytest.mark.integration
class TestSummaryRouteAccess:
    def test_unknown_id_is_404(self, client, app, db_session, configs_base):
        response = _get_summary(client, 987654)

        assert response.status_code == 404
        assert response.get_json() == NOT_FOUND

    def test_other_users_extraction_is_404(self, client, app, db_session, configs_base):
        path = _write_config(configs_base, OTHER_USER, PETCT_YAML)
        with app.app_context():
            extraction_id = _make_extraction(
                user_id=OTHER_USER,
                config_file=path,
                feature_names=FEATURE_NAMES,
                study_uids=SECRET_STUDIES,
                training_patients=SECRET_PATIENTS_TRAIN,
            )

        response = _get_summary(client, extraction_id, user=USER)

        assert response.status_code == 404
        assert response.get_json() == NOT_FOUND

    def test_owner_sees_extraction_that_other_user_cannot(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction(user_id=OTHER_USER)

        assert _get_summary(client, extraction_id, user=OTHER_USER).status_code == 200
        assert _get_summary(client, extraction_id, user=USER).status_code == 404

    @pytest.mark.parametrize("bad_id", ["abc", "1.5", "-1", "1;DROP"])
    def test_non_integer_id_is_404(self, client, app, db_session, configs_base, bad_id):
        response = _get_summary(client, bad_id)

        assert response.status_code == 404

    def test_missing_token_is_401(self, app, db_session, configs_base):
        with app.app_context():
            extraction_id = _make_extraction()

        with app.test_client() as anonymous:
            response = anonymous.get(f"/extractions/{extraction_id}/summary")

        assert response.status_code == 401

    def test_invalid_token_is_401(self, client, app, db_session, configs_base):
        from jose import JWTError

        with app.app_context():
            extraction_id = _make_extraction()

        with patch("routes.utils.decode_token", side_effect=JWTError("bad token")):
            response = client.get(f"/extractions/{extraction_id}/summary")

        assert response.status_code == 401


@pytest.mark.integration
class TestSummaryRouteSideEffects:
    def _seed(self, app, configs_base):
        path = _write_config(configs_base, USER, PETCT_YAML)
        with app.app_context():
            return _make_extraction(
                config_file=path,
                feature_names=FEATURE_NAMES,
                modalities=["CT", "PT"],
                rois=["GTV_T"],
                study_uids=SECRET_STUDIES,
                training_patients=SECRET_PATIENTS_TRAIN,
                test_patients=SECRET_PATIENTS_TEST,
                result_id="celery-group-result-id",
            )

    def _row_counts(self, app):
        from sqlalchemy import func, select

        from quantimage2_backend_common.models import db

        with app.app_context():
            return {
                table.name: db.session.execute(
                    select(func.count()).select_from(table)
                ).scalar()
                for table in db.metadata.sorted_tables
            }

    def test_no_rows_created_or_modified(self, client, app, db_session, configs_base):
        from quantimage2_backend_common.models import Album, FeatureExtraction

        extraction_id = self._seed(app, configs_base)
        before = self._row_counts(app)
        with app.app_context():
            updated_before = FeatureExtraction.find_by_id(extraction_id).updated_at

        with patch.object(
            Album, "find_by_album_id", wraps=Album.find_by_album_id
        ) as find_album:
            response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        after = self._row_counts(app)
        assert after == before
        find_album.assert_not_called()
        with app.app_context():
            db_session.expire_all()
            extraction = FeatureExtraction.find_by_id(extraction_id)
            assert extraction.updated_at == updated_before
            assert extraction.training_patients == SECRET_PATIENTS_TRAIN
            assert extraction.test_patients == SECRET_PATIENTS_TEST

    def test_no_kheops_or_celery_calls(self, client, app, db_session, configs_base):
        import requests

        import quantimage2_backend_common.utils as common_utils

        extraction_id = self._seed(app, configs_base)
        fake_celery = MagicMock()
        original_app_celery = app.my_celery
        app.my_celery = MagicMock()
        try:
            with patch.object(requests, "get") as http_get, patch.object(
                requests, "post"
            ) as http_post, patch.object(
                requests, "request"
            ) as http_request, patch.object(
                requests.Session, "request"
            ) as session_request, patch.object(
                common_utils, "celery", fake_celery
            ), patch(
                "routes.features.get_studies_from_album"
            ) as studies, patch(
                "routes.features.get_album_details"
            ) as album_details, patch(
                "routes.features.fetch_extraction_result"
            ) as fetch_result:
                response = _get_summary(client, extraction_id)

            assert response.status_code == 200
            http_get.assert_not_called()
            http_post.assert_not_called()
            http_request.assert_not_called()
            session_request.assert_not_called()
            studies.assert_not_called()
            album_details.assert_not_called()
            fetch_result.assert_not_called()
            assert fake_celery.mock_calls == []
            assert app.my_celery.mock_calls == []
        finally:
            app.my_celery = original_app_celery

    def test_response_has_no_patient_or_study_ids_or_paths(
        self, client, app, db_session, configs_base
    ):
        extraction_id = self._seed(app, configs_base)

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        _assert_no_leaks(response, configs_base)
        body = response.get_json()
        assert "config_file" not in json.dumps(body)
        assert "config_file" not in body
        assert "config_file" not in body["config"]


@pytest.mark.integration
class TestExistingExtractionRoutesRegression:
    def test_get_extraction_by_id_still_responds(
        self, client, app, db_session, configs_base
    ):
        with app.app_context():
            extraction_id = _make_extraction(
                feature_names=["original_shape_Elongation"], modalities=["CT"]
            )

        with _patch_decode_token():
            response = client.get(f"/extractions/{extraction_id}")

        assert response.status_code == 200
        body = response.get_json()
        assert body["id"] == extraction_id
        assert body["feature_definitions"] == ["original_shape_Elongation"]
        assert body["modalities"] == ["CT"]

    def test_download_configuration_still_returns_the_yaml(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(configs_base, USER, PETCT_YAML)
        with app.app_context():
            extraction_id = _make_extraction(config_file=path)

        with _patch_decode_token():
            response = client.get(
                f"/extractions/{extraction_id}/download-configuration"
            )

        assert response.status_code == 200
        assert response.get_data(as_text=True) == PETCT_YAML


@pytest.mark.integration
class TestSummaryRouteOkapyRows:
    def test_ct_only_extraction_with_petct_config_has_single_ct_row(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(configs_base, USER, PETCT_YAML)
        with app.app_context():
            extraction_id = _make_extraction(
                config_file=path,
                feature_names=["original_firstorder_Mean"],
                modalities=["CT"],
                rois=["GTV_T"],
            )

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        assert response.get_json()["config"] == {
            "status": "ok",
            "general": EXPECTED_PETCT_CONFIG["general"],
            "modalities": [EXPECTED_PETCT_CONFIG["modalities"][0]],
        }

    def test_route_rows_use_prefix_fallback_for_extracted_modalities(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(configs_base, USER, MRI_YAML)
        with app.app_context():
            extraction_id = _make_extraction(
                config_file=path, modalities=["MR_T1", "MR_T1Pre"]
            )

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        rows = response.get_json()["config"]["modalities"]
        assert [r["modality"] for r in rows] == ["MR_T1", "MR_T1Pre"]
        # MR_T1Pre: no own key, no "MR" key, default is null -> no extractors,
        # common's resampler and standardizer
        assert rows[1] == {
            "modality": "MR_T1Pre",
            "resampling": LINEAR_1MM,
            "normalization": {"type": "standardizer", "params": {"threshold": 0.001}},
            "extractors": [],
        }
        assert rows[0]["extractors"] == [
            _pyrad_extractor({"binWidth": 0.3, "voxelArrayShift": 10})
        ]

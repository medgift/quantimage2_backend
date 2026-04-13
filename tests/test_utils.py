"""
Tests for the shared utility functions and constants.

Covers exception classes, Socket.IO body formatters, feature ID parsing,
config reading, and datetime helpers.
"""

import datetime
from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Exception Classes
# ---------------------------------------------------------------------------


class TestCustomException:
    def test_default_status_code(self):
        from quantimage2_backend_common.utils import CustomException

        exc = CustomException("something broke")
        assert exc.status_code == 500
        assert exc.message == "something broke"
        assert str(exc) == "something broke"

    def test_custom_status_code(self):
        from quantimage2_backend_common.utils import CustomException

        exc = CustomException("bad request", status_code=400)
        assert exc.status_code == 400

    def test_to_dict(self):
        from quantimage2_backend_common.utils import CustomException

        exc = CustomException("test", payload={"detail": "info"})
        d = exc.to_dict()
        assert d["message"] == "test"
        assert d["detail"] == "info"

    def test_to_dict_no_payload(self):
        from quantimage2_backend_common.utils import CustomException

        exc = CustomException("test")
        d = exc.to_dict()
        assert d["message"] == "test"


class TestInvalidUsage:
    def test_status_code_400(self):
        from quantimage2_backend_common.utils import InvalidUsage

        exc = InvalidUsage("bad input")
        assert exc.status_code == 400
        assert exc.message == "bad input"


class TestComputationError:
    def test_status_code_500(self):
        from quantimage2_backend_common.utils import ComputationError

        exc = ComputationError("extraction failed")
        assert exc.status_code == 500
        assert exc.message == "extraction failed"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_feature_id_separator(self):
        from quantimage2_backend_common.const import FEATURE_ID_SEPARATOR

        # Must be non-breaking hyphen (U+2011)
        assert FEATURE_ID_SEPARATOR == "\u2011"
        assert FEATURE_ID_SEPARATOR != "-"  # Not a regular hyphen

    def test_feature_id_matcher(self):
        from quantimage2_backend_common.const import (
            FEATURE_ID_SEPARATOR,
            featureIDMatcher,
        )

        sep = FEATURE_ID_SEPARATOR
        feature_id = f"CT{sep}GTV-T{sep}original_shape_Elongation"
        match = featureIDMatcher.match(feature_id)
        assert match is not None
        assert match.group("modality") == "CT"
        assert match.group("roi") == "GTV-T"
        assert match.group("feature") == "original_shape_Elongation"

    def test_feature_id_matcher_with_regular_hyphens_in_roi(self):
        from quantimage2_backend_common.const import (
            FEATURE_ID_SEPARATOR,
            featureIDMatcher,
        )

        sep = FEATURE_ID_SEPARATOR
        # ROI names can contain regular hyphens
        feature_id = f"PET{sep}GTV-T-Large{sep}log_glcm_JointAverage"
        match = featureIDMatcher.match(feature_id)
        assert match is not None
        assert match.group("roi") == "GTV-T-Large"
        assert match.group("feature") == "log_glcm_JointAverage"

    def test_feature_id_matcher_no_match(self):
        from quantimage2_backend_common.const import featureIDMatcher

        # Plain text without non-breaking hyphens should not match
        assert featureIDMatcher.match("some_random_text") is None

    def test_model_types_enum(self):
        from quantimage2_backend_common.const import MODEL_TYPES

        assert MODEL_TYPES.CLASSIFICATION.value == "Classification"
        assert MODEL_TYPES.SURVIVAL.value == "Survival"

    def test_data_splitting_types_enum(self):
        from quantimage2_backend_common.const import DATA_SPLITTING_TYPES

        assert DATA_SPLITTING_TYPES.FULLDATASET.value == "fulldataset"
        assert DATA_SPLITTING_TYPES.TRAINTESTSPLIT.value == "traintest"

    def test_estimator_step_enum(self):
        from quantimage2_backend_common.const import ESTIMATOR_STEP

        assert ESTIMATOR_STEP.CLASSIFICATION.value == "classifier"
        assert ESTIMATOR_STEP.SURVIVAL.value == "analyzer"

    def test_queue_constants(self):
        from quantimage2_backend_common.const import QUEUE_EXTRACTION, QUEUE_TRAINING

        assert QUEUE_EXTRACTION == "extraction"
        assert QUEUE_TRAINING == "training"

    def test_all_feature_prefixes_present(self):
        from quantimage2_backend_common.const import (
            PYRADIOMICS_FEATURE_PREFIXES,
            RIESZ_FEATURE_PREFIXES,
            ZRAD_FEATURE_PREFIXES,
        )

        assert "original" in PYRADIOMICS_FEATURE_PREFIXES
        assert "wavelet" in PYRADIOMICS_FEATURE_PREFIXES
        assert "tex" in RIESZ_FEATURE_PREFIXES
        assert "zrad" in ZRAD_FEATURE_PREFIXES

    def test_feature_id_matcher_all_prefixes(self):
        """Every known prefix should be matchable by the regex."""
        from quantimage2_backend_common.const import (
            FEATURE_ID_SEPARATOR,
            featureIDMatcher,
            prefixes,
        )

        sep = FEATURE_ID_SEPARATOR
        for prefix in prefixes:
            fid = f"CT{sep}GTV{sep}{prefix}_someFeature"
            match = featureIDMatcher.match(fid)
            assert match is not None, f"Prefix '{prefix}' did not match"


# ---------------------------------------------------------------------------
# read_config_file
# ---------------------------------------------------------------------------


class TestReadConfigFile:
    def test_read_existing_file(self, tmp_path):
        from quantimage2_backend_common.utils import read_config_file

        config_file = tmp_path / "config.yaml"
        config_file.write_text("setting: value\n")

        content = read_config_file(str(config_file))
        assert "setting: value" in content

    def test_read_nonexistent_file(self):
        from quantimage2_backend_common.utils import read_config_file

        result = read_config_file("/nonexistent/path.yaml")
        assert result is None


# ---------------------------------------------------------------------------
# MessageType enum
# ---------------------------------------------------------------------------


class TestMessageType:
    def test_values(self):
        from quantimage2_backend_common.utils import MessageType

        assert MessageType.FEATURE_TASK_STATUS.value == "feature-status"
        assert MessageType.EXTRACTION_STATUS.value == "extraction-status"
        assert MessageType.TRAINING_STATUS.value == "training-status"


# ---------------------------------------------------------------------------
# CV constants
# ---------------------------------------------------------------------------


class TestCVConstants:
    def test_cv_defaults(self):
        from quantimage2_backend_common.utils import CV_SPLITS, CV_REPEATS

        assert CV_SPLITS == 5
        assert CV_REPEATS == 1

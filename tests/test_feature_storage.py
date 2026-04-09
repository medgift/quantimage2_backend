"""
Tests for feature storage functions.

Covers store_features(), store_extraction_associations(), and feature
retrieval via FeatureValue model methods.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# store_features
# ---------------------------------------------------------------------------


class TestStoreFeatures:
    def _create_extraction(self, db_session):
        """Create a FeatureExtraction + Task for testing."""
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureExtractionTask,
        )

        extraction = FeatureExtraction("user-store", "album-store")
        extraction.save_to_db()

        task = FeatureExtractionTask(extraction.id, "study-store-1", "celery-123")
        task.save_to_db()

        return extraction, task

    def test_store_features_basic(self, app, db_session):
        from quantimage2_backend_common.feature_storage import store_features
        from quantimage2_backend_common.models import FeatureValue

        extraction, task = self._create_extraction(db_session)

        features_df = pd.DataFrame(
            {
                "patient": ["P1", "P1", "P2", "P2"],
                "modality": ["CT", "CT", "CT", "CT"],
                "VOI": ["GTV", "GTV", "GTV", "GTV"],
                "feature_name": ["feat_a", "feat_b", "feat_a", "feat_b"],
                "feature_value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = store_features(task.id, extraction.id, features_df)
        assert len(result) == 4

    def test_store_features_creates_modality(self, app, db_session):
        from quantimage2_backend_common.feature_storage import store_features
        from quantimage2_backend_common.models import Modality

        extraction, task = self._create_extraction(db_session)

        features_df = pd.DataFrame(
            {
                "patient": ["P1"],
                "modality": ["MR_STORE_TEST"],
                "VOI": ["GTV_STORE"],
                "feature_name": ["original_test_feature"],
                "feature_value": [42.0],
            }
        )

        store_features(task.id, extraction.id, features_df)

        modality, created = Modality.get_or_create(
            criteria={"name": "MR_STORE_TEST"},
            defaults={"name": "MR_STORE_TEST"},
        )
        assert created is False  # Already exists

    def test_store_features_creates_roi(self, app, db_session):
        from quantimage2_backend_common.feature_storage import store_features
        from quantimage2_backend_common.models import ROI

        extraction, task = self._create_extraction(db_session)

        features_df = pd.DataFrame(
            {
                "patient": ["P1"],
                "modality": ["CT"],
                "VOI": ["GTV-NEW"],
                "feature_name": ["original_test"],
                "feature_value": [1.0],
            }
        )

        store_features(task.id, extraction.id, features_df)

        roi, created = ROI.get_or_create(
            criteria={"name": "GTV-NEW"},
            defaults={"name": "GTV-NEW"},
        )
        assert created is False

    def test_store_features_creates_feature_definitions(self, app, db_session):
        from quantimage2_backend_common.feature_storage import store_features
        from quantimage2_backend_common.models import FeatureDefinition

        extraction, task = self._create_extraction(db_session)

        features_df = pd.DataFrame(
            {
                "patient": ["P1", "P1"],
                "modality": ["CT", "CT"],
                "VOI": ["GTV", "GTV"],
                "feature_name": ["unique_feat_alpha", "unique_feat_beta"],
                "feature_value": [1.0, 2.0],
            }
        )

        store_features(task.id, extraction.id, features_df)

        found = FeatureDefinition.find_by_name(
            ["unique_feat_alpha", "unique_feat_beta"]
        )
        names = {f.name for f in found}
        assert "unique_feat_alpha" in names
        assert "unique_feat_beta" in names

    def test_store_features_cancelled_extraction(self, app, db_session):
        """If extraction was deleted, store_features should return None."""
        from quantimage2_backend_common.feature_storage import store_features

        # Non-existent extraction ID — positional args match function signature
        result = store_features(1, 99999, pd.DataFrame())
        assert result is None

    def test_store_features_associates_modalities_with_extraction(
        self, app, db_session
    ):
        from quantimage2_backend_common.feature_storage import store_features
        from quantimage2_backend_common.models import FeatureExtraction

        extraction, task = self._create_extraction(db_session)

        features_df = pd.DataFrame(
            {
                "patient": ["P1", "P1"],
                "modality": ["CT_ASSOC", "PT_ASSOC"],
                "VOI": ["GTV", "GTV"],
                "feature_name": ["original_f1", "original_f1"],
                "feature_value": [1.0, 2.0],
            }
        )

        store_features(task.id, extraction.id, features_df)

        # Reload extraction
        found = FeatureExtraction.find_by_id(extraction.id)
        modality_names = {m.name for m in found.modalities}
        assert "CT_ASSOC" in modality_names
        assert "PT_ASSOC" in modality_names


# ---------------------------------------------------------------------------
# store_extraction_associations
# ---------------------------------------------------------------------------


class TestStoreExtractionAssociations:
    def test_creates_new_associations(self, app, db_session):
        from quantimage2_backend_common.feature_storage import (
            store_extraction_associations,
        )
        from quantimage2_backend_common.models import Modality

        instances = store_extraction_associations(
            Modality, ["NewMod1", "NewMod2"], existing=[]
        )
        assert len(instances) == 2
        names = {i.name for i in instances}
        assert "NewMod1" in names
        assert "NewMod2" in names

    def test_reuses_existing_associations(self, app, db_session):
        from quantimage2_backend_common.feature_storage import (
            store_extraction_associations,
        )
        from quantimage2_backend_common.models import Modality

        existing_mod = Modality("ExistingMod")
        existing_mod.save_to_db()

        instances = store_extraction_associations(
            Modality, ["ExistingMod", "BrandNew"], existing=[existing_mod]
        )
        assert len(instances) == 2
        # The existing modality should be the same object
        existing_instance = next(i for i in instances if i.name == "ExistingMod")
        assert existing_instance.id == existing_mod.id

    def test_mixed_new_and_existing(self, app, db_session):
        from quantimage2_backend_common.feature_storage import (
            store_extraction_associations,
        )
        from quantimage2_backend_common.models import ROI

        roi1 = ROI("ROI_EXIST")
        roi1.save_to_db()

        instances = store_extraction_associations(
            ROI, ["ROI_EXIST", "ROI_NEW"], existing=[roi1]
        )
        assert len(instances) == 2

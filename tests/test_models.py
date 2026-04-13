"""
Tests for SQLAlchemy ORM models.

Covers BaseModel CRUD operations, model relationships, to_dict() serialization,
and model-specific class methods.
"""

import datetime
from datetime import timezone

import pytest

# ---------------------------------------------------------------------------
# BaseModel / FeaturePreset (simplest concrete model)
# ---------------------------------------------------------------------------


class TestBaseModelCRUD:
    """Test the generic CRUD methods inherited from BaseModel."""

    def test_save_and_find_by_id(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        preset = FeaturePreset("TestPreset", "/tmp/test.yaml")
        preset.save_to_db()

        assert preset.id is not None
        found = FeaturePreset.find_by_id(preset.id)
        assert found is not None
        assert found.name == "TestPreset"

    def test_find_all(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        FeaturePreset("P1", "/tmp/p1.yaml").save_to_db()
        FeaturePreset("P2", "/tmp/p2.yaml").save_to_db()

        all_presets = FeaturePreset.find_all()
        assert len(all_presets) >= 2

    def test_delete_by_id(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        preset = FeaturePreset("ToDelete", "/tmp/delete.yaml")
        preset.save_to_db()
        pid = preset.id

        deleted = FeaturePreset.delete_by_id(pid)
        assert deleted is not None
        assert FeaturePreset.find_by_id(pid) is None

    def test_delete_by_id_nonexistent(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        result = FeaturePreset.delete_by_id(99999)
        assert result is None

    def test_update(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        preset = FeaturePreset("OldName", "/tmp/old.yaml")
        preset.save_to_db()

        preset.update(name="NewName")
        found = FeaturePreset.find_by_id(preset.id)
        assert found.name == "NewName"

    def test_created_at_and_updated_at(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        preset = FeaturePreset("Timestamps", "/tmp/ts.yaml")
        preset.save_to_db()

        assert preset.created_at is not None
        assert preset.updated_at is not None
        assert isinstance(preset.created_at, datetime.datetime)

    def test_flush_to_db(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        preset = FeaturePreset("Flushed", "/tmp/flushed.yaml")
        preset.flush_to_db()

        # Should have an ID after flush (before commit)
        assert preset.id is not None

    def test_get_or_create_creates(self, app, db_session):
        from quantimage2_backend_common.models import Modality

        instance, created = Modality.get_or_create(
            criteria={"name": "CT_TEST"},
            defaults={"name": "CT_TEST"},
        )
        assert created is True
        assert instance.name == "CT_TEST"

    def test_get_or_create_gets_existing(self, app, db_session):
        from quantimage2_backend_common.models import Modality

        Modality.get_or_create(
            criteria={"name": "PET_TEST"},
            defaults={"name": "PET_TEST"},
        )
        instance, created = Modality.get_or_create(
            criteria={"name": "PET_TEST"},
            defaults={"name": "PET_TEST"},
        )
        assert created is False
        assert instance.name == "PET_TEST"


# ---------------------------------------------------------------------------
# FeaturePreset
# ---------------------------------------------------------------------------


class TestFeaturePreset:
    def test_find_by_name(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        FeaturePreset("FindMe", "/tmp/findme.yaml").save_to_db()
        found = FeaturePreset.find_by_name("FindMe")
        assert found is not None
        assert found.config_path == "/tmp/findme.yaml"

    def test_find_by_name_not_found(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        assert FeaturePreset.find_by_name("Nonexistent") is None

    def test_to_dict(self, app, db_session):
        from quantimage2_backend_common.models import FeaturePreset

        preset = FeaturePreset("DictTest", "/tmp/dict.yaml")
        preset.save_to_db()
        d = preset.to_dict()
        assert d["name"] == "DictTest"
        assert d["config_path"] == "/tmp/dict.yaml"
        assert "id" in d
        assert "created_at" in d
        assert "updated_at" in d


# ---------------------------------------------------------------------------
# FeatureExtraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_create_and_find(self, app, db_session):
        from quantimage2_backend_common.models import FeatureExtraction

        extraction = FeatureExtraction(
            user_id="user-123",
            album_id="album-456",
        )
        extraction.save_to_db()

        found = FeatureExtraction.find_by_id(extraction.id)
        assert found is not None
        assert found.user_id == "user-123"
        assert found.album_id == "album-456"

    def test_find_latest_by_user_and_album_id(self, app, db_session):
        from quantimage2_backend_common.models import FeatureExtraction

        FeatureExtraction("user-1", "album-1").save_to_db()
        second = FeatureExtraction("user-1", "album-1")
        second.save_to_db()

        latest = FeatureExtraction.find_latest_by_user_and_album_id("user-1", "album-1")
        assert latest.id == second.id

    def test_find_by_user(self, app, db_session):
        from quantimage2_backend_common.models import FeatureExtraction

        FeatureExtraction("user-search", "album-a").save_to_db()
        FeatureExtraction("user-search", "album-b").save_to_db()
        FeatureExtraction("other-user", "album-c").save_to_db()

        results = FeatureExtraction.find_by_user("user-search")
        assert len(results) == 2

    def test_to_dict(self, app, db_session):
        from quantimage2_backend_common.models import FeatureExtraction

        extraction = FeatureExtraction("user-dict", "album-dict")
        extraction.save_to_db()
        d = extraction.to_dict()

        assert d["user_id"] == "user-dict"
        assert d["album_id"] == "album-dict"
        assert "tasks" in d
        assert "modalities" in d
        assert "rois" in d


# ---------------------------------------------------------------------------
# FeatureExtractionTask
# ---------------------------------------------------------------------------


class TestFeatureExtractionTask:
    def test_create_task(self, app, db_session):
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureExtractionTask,
        )

        extraction = FeatureExtraction("user-task", "album-task")
        extraction.save_to_db()

        task = FeatureExtractionTask(
            feature_extraction_id=extraction.id,
            study_uid="1.2.3.4.5",
            task_id="celery-task-abc",
        )
        task.save_to_db()

        assert task.id is not None
        assert task.study_uid == "1.2.3.4.5"
        assert task.feature_extraction_id == extraction.id

    def test_task_to_dict(self, app, db_session):
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureExtractionTask,
        )

        extraction = FeatureExtraction("u", "a")
        extraction.save_to_db()
        task = FeatureExtractionTask(extraction.id, "1.2.3", "task-id")
        task.save_to_db()

        d = task.to_dict()
        assert d["study_uid"] == "1.2.3"
        assert d["feature_extraction_id"] == extraction.id

    def test_extraction_tasks_relationship(self, app, db_session):
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureExtractionTask,
        )

        extraction = FeatureExtraction("u-rel", "a-rel")
        extraction.save_to_db()
        FeatureExtractionTask(extraction.id, "study1", "tid1").save_to_db()
        FeatureExtractionTask(extraction.id, "study2", "tid2").save_to_db()

        # Refresh to load relationship
        found = FeatureExtraction.find_by_id(extraction.id)
        assert len(found.tasks) == 2


# ---------------------------------------------------------------------------
# FeatureDefinition, Modality, ROI
# ---------------------------------------------------------------------------


class TestMetadataModels:
    def test_modality_create(self, app, db_session):
        from quantimage2_backend_common.models import Modality

        m = Modality("MR_TEST")
        m.save_to_db()
        assert m.id is not None
        assert m.name == "MR_TEST"

    def test_roi_create(self, app, db_session):
        from quantimage2_backend_common.models import ROI

        r = ROI("GTV-T")
        r.save_to_db()
        assert r.id is not None
        assert r.name == "GTV-T"

    def test_feature_definition_create(self, app, db_session):
        from quantimage2_backend_common.models import FeatureDefinition

        fd = FeatureDefinition("original_glcm_JointAverage")
        fd.save_to_db()
        assert fd.id is not None

    def test_feature_definition_find_by_name(self, app, db_session):
        from quantimage2_backend_common.models import FeatureDefinition

        FeatureDefinition("feat_a").save_to_db()
        FeatureDefinition("feat_b").save_to_db()
        FeatureDefinition("feat_c").save_to_db()

        found = FeatureDefinition.find_by_name(["feat_a", "feat_c"])
        names = {f.name for f in found}
        assert "feat_a" in names
        assert "feat_c" in names
        assert "feat_b" not in names


# ---------------------------------------------------------------------------
# FeatureValue
# ---------------------------------------------------------------------------


class TestFeatureValue:
    def _create_prerequisites(self, db_session):
        """Create prerequisite records for a FeatureValue."""
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureExtractionTask,
            FeatureDefinition,
            Modality,
            ROI,
        )

        extraction = FeatureExtraction("u-fv", "a-fv")
        extraction.save_to_db()

        task = FeatureExtractionTask(extraction.id, "study-fv", "task-fv")
        task.save_to_db()

        modality, _ = Modality.get_or_create(
            criteria={"name": "CT_FV"}, defaults={"name": "CT_FV"}
        )
        roi, _ = ROI.get_or_create(
            criteria={"name": "GTV_FV"}, defaults={"name": "GTV_FV"}
        )
        feat_def = FeatureDefinition("original_shape_Maximum2DDiameterSlice")
        feat_def.save_to_db()

        return task, modality, roi, feat_def

    def test_create_feature_value(self, app, db_session):
        from quantimage2_backend_common.models import FeatureValue

        task, modality, roi, feat_def = self._create_prerequisites(db_session)
        fv = FeatureValue(
            value=42.5,
            feature_definition_id=feat_def.id,
            feature_extraction_task_id=task.id,
            modality_id=modality.id,
            roi_id=roi.id,
        )
        fv.save_to_db()
        assert fv.id is not None
        assert fv.value == 42.5

    def test_save_features_batch(self, app, db_session):
        from quantimage2_backend_common.models import FeatureValue

        task, modality, roi, feat_def = self._create_prerequisites(db_session)
        batch = [
            {
                "value": float(i),
                "feature_definition_id": feat_def.id,
                "feature_extraction_task_id": task.id,
                "modality_id": modality.id,
                "roi_id": roi.id,
            }
            for i in range(5)
        ]
        FeatureValue.save_features_batch(batch)

        values = FeatureValue.find_by_tasks_modality_roi_features(
            [task.id], modality.id, roi.id, [feat_def.id]
        )
        assert len(values) == 5


# ---------------------------------------------------------------------------
# LabelCategory & Label
# ---------------------------------------------------------------------------


class TestLabels:
    def test_label_category_create(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory

        lc = LabelCategory(
            album_id="album-lbl",
            label_type="Classification",
            name="PLC Status",
            user_id="user-lbl",
            pos_label="Positive",
        )
        lc.save_to_db()
        assert lc.id is not None
        assert lc.pos_label == "Positive"

    def test_label_category_find_by_album(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory

        LabelCategory("alb-1", "Classification", "Cat1", "u-1").save_to_db()
        LabelCategory("alb-1", "Survival", "Cat2", "u-1").save_to_db()
        LabelCategory("alb-2", "Classification", "Cat3", "u-1").save_to_db()

        results = LabelCategory.find_by_album("alb-1", "u-1")
        assert len(results) == 2

    def test_label_save_and_retrieve(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory, Label

        lc = LabelCategory("alb-l", "Classification", "Cat", "u-l")
        lc.save_to_db()

        label = Label(lc.id, "patient-001", {"Outcome": "Positive"})
        label.save_to_db()

        found = Label.find_by_label_category(lc.id)
        assert len(found) == 1
        assert found[0].patient_id == "patient-001"
        assert found[0].label_content["Outcome"] == "Positive"

    def test_save_labels_bulk(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory, Label

        lc = LabelCategory("alb-bulk", "Survival", "SurvCat", "u-bulk")
        lc.save_to_db()

        labels = [
            Label(lc.id, f"patient-{i}", {"Event": True, "Time": float(i * 10)})
            for i in range(5)
        ]
        saved = Label.save_labels(lc.id, labels)
        assert len(saved) == 5

    def test_save_label_upsert(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory, Label

        lc = LabelCategory("alb-upsert", "Classification", "Upsert", "u-upsert")
        lc.save_to_db()

        Label.save_label(lc.id, "p1", {"Outcome": "Old"})
        Label.save_label(lc.id, "p1", {"Outcome": "New"})

        found = Label.find_by_label_category(lc.id)
        assert len(found) == 1
        assert found[0].label_content["Outcome"] == "New"

    def test_label_category_to_dict(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory, Label

        lc = LabelCategory("alb-d", "Classification", "Dict", "u-d")
        lc.save_to_db()
        Label(lc.id, "p1", {"Outcome": "X"}).save_to_db()

        d = lc.to_dict()
        assert d["album_id"] == "alb-d"
        assert d["label_type"] == "Classification"
        assert len(d["labels"]) == 1


# ---------------------------------------------------------------------------
# FeatureCollection
# ---------------------------------------------------------------------------


class TestFeatureCollection:
    def test_create_collection(self, app, db_session):
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureCollection,
        )

        extraction = FeatureExtraction("u-coll", "a-coll")
        extraction.save_to_db()

        collection = FeatureCollection(
            name="My Collection",
            feature_extraction_id=extraction.id,
            feature_ids=["CT\u2011GTV\u2011original_shape_Elongation"],
            data_splitting_type="traintest",
            train_test_split_type="automatic",
            training_patients=["p1", "p2"],
            test_patients=["p3"],
        )
        collection.save_to_db()

        assert collection.id is not None
        assert collection.name == "My Collection"

    def test_find_by_extraction(self, app, db_session):
        from quantimage2_backend_common.models import (
            FeatureExtraction,
            FeatureCollection,
        )

        extraction = FeatureExtraction("u-fc", "a-fc")
        extraction.save_to_db()

        FeatureCollection(
            "C1", extraction.id, ["f1"], "traintest", "automatic", ["p1"], ["p2"]
        ).save_to_db()
        FeatureCollection(
            "C2", extraction.id, ["f2"], "fulldataset", "automatic", ["p1", "p2"], None
        ).save_to_db()

        found = FeatureCollection.find_by_extraction(extraction.id)
        assert len(found) == 2


# ---------------------------------------------------------------------------
# NavigationHistory
# ---------------------------------------------------------------------------


class TestNavigationHistory:
    def test_create_entry_and_find(self, app, db_session):
        from quantimage2_backend_common.models import NavigationHistory

        entry = NavigationHistory.create_entry("/albums/123", "user-nav")
        assert entry.id is not None
        assert entry.path == "/albums/123"

        found = NavigationHistory.find_by_user("user-nav")
        assert len(found) == 1


# ---------------------------------------------------------------------------
# Album
# ---------------------------------------------------------------------------


class TestAlbum:
    def test_find_by_album_id_creates_if_missing(self, app, db_session):
        from quantimage2_backend_common.models import Album

        album = Album.find_by_album_id("new-album")
        assert album is not None
        assert album.album_id == "new-album"

    def test_save_rois(self, app, db_session):
        from quantimage2_backend_common.models import Album

        Album.find_by_album_id("roi-album")
        album = Album.save_rois("roi-album", ["GTV-T", "GTV-N"])
        assert album.rois == ["GTV-T", "GTV-N"]


# ---------------------------------------------------------------------------
# ClinicalFeatureDefinition & ClinicalFeatureValue
# ---------------------------------------------------------------------------


class TestClinicalFeatures:
    def test_definition_create(self, app, db_session):
        from quantimage2_backend_common.models import ClinicalFeatureDefinition

        cfd = ClinicalFeatureDefinition(
            name="Age",
            album_id="alb-clin",
            user_id="u-clin",
            feat_type="Number",
            encoding="Normalization",
            missing_values="Mean",
        )
        cfd.save_to_db()
        assert cfd.id is not None

    def test_definition_find_by_user_and_album(self, app, db_session):
        from quantimage2_backend_common.models import ClinicalFeatureDefinition

        ClinicalFeatureDefinition(
            "Weight", "alb-f", "u-f", "Number", "None", "Drop"
        ).save_to_db()
        ClinicalFeatureDefinition(
            "Smoking", "alb-f", "u-f", "Categorical", "One-Hot Encoding", "Mode"
        ).save_to_db()

        found = ClinicalFeatureDefinition.find_by_user_id_and_album_id("u-f", "alb-f")
        assert len(found) == 2

    def test_clinical_value_create(self, app, db_session):
        from quantimage2_backend_common.models import (
            ClinicalFeatureDefinition,
            ClinicalFeatureValue,
        )

        cfd = ClinicalFeatureDefinition(
            "Height", "alb-cv", "u-cv", "Number", "None", "Drop"
        )
        cfd.save_to_db()

        cfv = ClinicalFeatureValue("175", cfd.id, "patient-cv-1")
        cfv.save_to_db()
        assert cfv.id is not None
        assert cfv.value == "175"


# ---------------------------------------------------------------------------
# AlbumOutcome
# ---------------------------------------------------------------------------


class TestAlbumOutcome:
    def test_save_current_outcome(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory, AlbumOutcome

        lc = LabelCategory("alb-ao", "Classification", "Outcome", "u-ao")
        lc.save_to_db()

        ao = AlbumOutcome.save_current_outcome("alb-ao", "u-ao", lc.id)
        assert ao.outcome_id == lc.id

    def test_find_by_album_user_id(self, app, db_session):
        from quantimage2_backend_common.models import LabelCategory, AlbumOutcome

        lc = LabelCategory("alb-find-ao", "Classification", "O", "u-find-ao")
        lc.save_to_db()
        AlbumOutcome.save_current_outcome("alb-find-ao", "u-find-ao", lc.id)

        found = AlbumOutcome.find_by_album_user_id("alb-find-ao", "u-find-ao")
        assert found is not None
        assert found.outcome_id == lc.id


# ---------------------------------------------------------------------------
# alchemyencoder
# ---------------------------------------------------------------------------


class TestAlchemyEncoder:
    def test_encodes_date(self):
        from quantimage2_backend_common.models import alchemyencoder

        d = datetime.date(2024, 1, 15)
        assert alchemyencoder(d) == "2024-01-15Z"

    def test_encodes_datetime(self):
        from quantimage2_backend_common.models import alchemyencoder

        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        result = alchemyencoder(dt)
        assert result.startswith("2024-01-15")
        assert result.endswith("Z")

    def test_encodes_decimal(self):
        import decimal
        from quantimage2_backend_common.models import alchemyencoder

        d = decimal.Decimal("3.14")
        assert alchemyencoder(d) == 3.14

    def test_returns_none_for_other(self):
        from quantimage2_backend_common.models import alchemyencoder

        assert alchemyencoder("hello") is None


# ---------------------------------------------------------------------------
# _utcnow
# ---------------------------------------------------------------------------


class TestUtcNow:
    def test_returns_naive_datetime(self):
        from quantimage2_backend_common.models import _utcnow

        now = _utcnow()
        assert isinstance(now, datetime.datetime)
        assert now.tzinfo is None

    def test_is_approximately_current_time(self):
        from quantimage2_backend_common.models import _utcnow

        now = _utcnow()
        utc_now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        delta = abs((utc_now - now).total_seconds())
        assert delta < 2  # Within 2 seconds

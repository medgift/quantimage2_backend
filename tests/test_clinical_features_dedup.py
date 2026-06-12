"""
Tests for newest-wins deduplication of clinical features that share a name
across several uploaded files (service.clinical_features_dedup), its wiring
into the training paths, and the /clinical-features/duplicates advisory route.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _patch_validate_decorate():
    from flask import g

    def fake_validate(request):
        if request.method != "OPTIONS":
            g.user = "test-user-uuid-1234"
            g.token = "fake-jwt-token"

    return patch("routes.utils.validate_decorate", side_effect=fake_validate)


def _patch_decode_token():
    return patch(
        "routes.utils.decode_token",
        return_value={
            "sub": "test-user-uuid-1234",
            "preferred_username": "testuser",
            "resource_access": {"quantimage2-frontend": {"roles": ["admin"]}},
        },
    )


USER = "test-user-uuid-1234"
ALBUM = "alb-dedup"


class TestDedupeDefinitionsByName:
    def test_newest_file_wins(self):
        from service.clinical_features_dedup import dedupe_definitions_by_name

        defs = [
            SimpleNamespace(id=10, name="Age", clinical_feature_file_id=1),
            SimpleNamespace(id=11, name="Age", clinical_feature_file_id=3),
            SimpleNamespace(id=12, name="Age", clinical_feature_file_id=2),
        ]
        result = dedupe_definitions_by_name(defs)
        assert len(result) == 1
        assert result[0].id == 11  # file_id 3 is the newest

    def test_distinct_names_untouched(self):
        from service.clinical_features_dedup import dedupe_definitions_by_name

        defs = [
            SimpleNamespace(id=10, name="Age", clinical_feature_file_id=1),
            SimpleNamespace(id=11, name="Sex", clinical_feature_file_id=2),
        ]
        assert len(dedupe_definitions_by_name(defs)) == 2

    def test_definition_with_values_beats_newer_empty_one(self):
        """An orphan (newer file, no values) must not shadow real older data."""
        from service.clinical_features_dedup import dedupe_definitions_by_name

        defs = [
            SimpleNamespace(id=10, name="Age", clinical_feature_file_id=1),  # data
            SimpleNamespace(id=11, name="Age", clinical_feature_file_id=3),  # orphan
        ]
        # Only definition 10 (older file) has values.
        result = dedupe_definitions_by_name(defs, ids_with_values={10})
        assert len(result) == 1
        assert result[0].id == 10  # older-but-populated wins over newer-but-empty

    def test_newest_with_values_wins_when_several_have_data(self):
        from service.clinical_features_dedup import dedupe_definitions_by_name

        defs = [
            SimpleNamespace(id=10, name="Age", clinical_feature_file_id=1),
            SimpleNamespace(id=11, name="Age", clinical_feature_file_id=2),
            SimpleNamespace(id=12, name="Age", clinical_feature_file_id=3),  # empty
        ]
        # 10 and 11 have data, 12 (newest) is empty -> newest-with-data (11) wins.
        result = dedupe_definitions_by_name(defs, ids_with_values={10, 11})
        assert len(result) == 1
        assert result[0].id == 11


class TestResolveCollectionDedup:
    """A collection explicitly selecting the same name from two files must
    still feed the feature to the model only once (newest file wins)."""

    def _defs(self):
        return [
            SimpleNamespace(id=10, name="Age", clinical_feature_file_id=1),
            SimpleNamespace(id=11, name="Age", clinical_feature_file_id=2),
            SimpleNamespace(id=12, name="Sex", clinical_feature_file_id=2),
        ]

    def test_both_namespaced_copies_collapse_to_newest(self):
        from service.machine_learning import resolve_collection_clinical_definitions

        result = resolve_collection_clinical_definitions(
            ["1::Age", "2::Age"], self._defs()
        )
        assert len(result) == 1
        assert result[0].id == 11

    def test_legacy_name_plus_newer_exact_collapse_to_newest(self):
        from service.machine_learning import resolve_collection_clinical_definitions

        # Legacy bare "Age" resolves to file 1; "2::Age" is explicit. The
        # feature must still only be used once, from the newest file.
        result = resolve_collection_clinical_definitions(
            ["Age", "2::Age"], self._defs()
        )
        assert len(result) == 1
        assert result[0].id == 11


def _create_file_with_values(app, file_name, values_by_patient, feature_name="Age"):
    """Create a file + one definition + its values; returns (file_id, def_id)."""
    from quantimage2_backend_common.models import (
        ClinicalFeatureDefinition,
        ClinicalFeatureFile,
        ClinicalFeatureValue,
    )

    with app.app_context():
        f = ClinicalFeatureFile(file_name, ALBUM, USER)
        f.save_to_db()
        d = ClinicalFeatureDefinition(
            feature_name, ALBUM, USER, "Number", "Normalization", "Median", f.id
        )
        d.save_to_db()
        for patient_id, value in values_by_patient.items():
            ClinicalFeatureValue(value, d.id, patient_id).save_to_db()
        return f.id, d.id


class TestDuplicateAdvisories:
    def test_no_advisory_for_unique_names(self, app, db_session):
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        _create_file_with_values(app, "F1", {"P1": "30"}, feature_name="Age")
        _create_file_with_values(app, "F2", {"P1": "1"}, feature_name="Sex")
        with app.app_context():
            assert compute_clinical_duplicate_advisories(USER, ALBUM) == []

    def test_identical_values(self, app, db_session):
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        f1, _ = _create_file_with_values(app, "Old", {"P1": "30", "P2": "40"})
        f2, _ = _create_file_with_values(app, "New", {"P1": "30", "P2": "40"})
        with app.app_context():
            advisories = compute_clinical_duplicate_advisories(USER, ALBUM)

        assert len(advisories) == 1
        advisory = advisories[0]
        assert advisory["name"] == "Age"
        assert advisory["kept_file_id"] == f2
        assert advisory["kept_file_name"] == "New"
        assert advisory["dropped_file_ids"] == [f1]
        assert advisory["statuses"] == ["identical"]
        assert advisory["affected_patient_count"] == 0

    def test_coverage_loss(self, app, db_session):
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        # P2 only has a value in the old file -> ignored after dedup.
        _create_file_with_values(app, "Old", {"P1": "30", "P2": "40"})
        _create_file_with_values(app, "New", {"P1": "30"})
        with app.app_context():
            advisories = compute_clinical_duplicate_advisories(USER, ALBUM)

        assert advisories[0]["statuses"] == ["coverage_loss"]
        assert advisories[0]["coverage_loss_patient_count"] == 1
        assert advisories[0]["affected_patient_count"] == 1

    def test_conflict(self, app, db_session):
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        _create_file_with_values(app, "Old", {"P1": "30"})
        _create_file_with_values(app, "New", {"P1": "35"})
        with app.app_context():
            advisories = compute_clinical_duplicate_advisories(USER, ALBUM)

        assert advisories[0]["statuses"] == ["conflict"]
        assert advisories[0]["conflict_patient_count"] == 1

    def test_coverage_loss_and_conflict_combined(self, app, db_session):
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        _create_file_with_values(app, "Old", {"P1": "30", "P2": "40"})
        _create_file_with_values(app, "New", {"P1": "35"})
        with app.app_context():
            advisories = compute_clinical_duplicate_advisories(USER, ALBUM)

        assert advisories[0]["statuses"] == ["coverage_loss", "conflict"]
        assert advisories[0]["affected_patient_count"] == 2

    def test_missing_value_variants_are_not_conflicts(self, app, db_session):
        """None, "", "N/A" all count as missing — never as conflicting values."""
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        # Old file has only missing-style values for P1/P2 -> no coverage loss,
        # and a missing kept value never counts as a conflict.
        _create_file_with_values(app, "Old", {"P1": "", "P2": "N/A", "P3": None})
        _create_file_with_values(app, "New", {"P1": "30"})
        with app.app_context():
            advisories = compute_clinical_duplicate_advisories(USER, ALBUM)

        assert advisories[0]["statuses"] == ["identical"]

    def test_numeric_formatting_is_not_a_conflict(self, app, db_session):
        from service.clinical_features_dedup import (
            compute_clinical_duplicate_advisories,
        )

        _create_file_with_values(app, "Old", {"P1": "30"})
        _create_file_with_values(app, "New", {"P1": "30.0"})
        with app.app_context():
            advisories = compute_clinical_duplicate_advisories(USER, ALBUM)

        assert advisories[0]["statuses"] == ["identical"]


class TestDuplicatesRoute:
    def test_get_duplicates(self, client, app, db_session):
        _create_file_with_values(app, "Old", {"P1": "30"})
        f2, _ = _create_file_with_values(app, "New", {"P1": "35"})

        with _patch_validate_decorate(), _patch_decode_token():
            r = client.get(f"/clinical-features/duplicates?album_id={ALBUM}")

        assert r.status_code == 200
        advisories = r.get_json()
        assert len(advisories) == 1
        assert advisories[0]["name"] == "Age"
        assert advisories[0]["kept_file_id"] == f2
        assert advisories[0]["statuses"] == ["conflict"]


class TestGetClinicalFeaturesNoDuplicateColumns:
    def test_all_features_training_uses_each_name_once(self, app, db_session):
        """The training matrix must contain a duplicated name only once, namespaced
        by the newest file (this is the CenterID-used-twice bug)."""
        from service.machine_learning import get_clinical_features

        _create_file_with_values(app, "Old", {"P1": "30", "P2": "40"})
        f2, _ = _create_file_with_values(app, "New", {"P1": "31", "P2": "41"})

        with app.app_context():
            features_df = get_clinical_features(
                USER, None, ["P1", "P2"], {"album_id": ALBUM}
            )

        age_columns = [c for c in features_df.columns if "Age" in c]
        assert len(age_columns) == 1
        assert age_columns[0].startswith(f"{f2}::")

    def test_value_less_orphan_does_not_shadow_real_data(self, app, db_session):
        """A newer file with a definition but no values must not hide an older
        file that has the same feature with real values."""
        from service.machine_learning import get_clinical_features

        old_file, _ = _create_file_with_values(app, "Old", {"P1": "30", "P2": "40"})
        # Orphan: newer file, same name "Age", but zero values saved.
        new_file, _ = _create_file_with_values(app, "Orphan", {})

        with app.app_context():
            features_df = get_clinical_features(
                USER, None, ["P1", "P2"], {"album_id": ALBUM}
            )

        age_columns = [c for c in features_df.columns if "Age" in c]
        assert len(age_columns) == 1
        # The older, populated file is used; the newer empty one is not.
        assert age_columns[0].startswith(f"{old_file}::")
        assert not age_columns[0].startswith(f"{new_file}::")
        assert len(features_df) == 2

    def test_value_less_orphan_with_unique_name_is_skipped(self, app, db_session):
        """The distances-file scenario: an orphan file's unique-named definition
        with no values must be skipped, not crash set_index('PatientID')."""
        from service.machine_learning import get_clinical_features

        good_file, _ = _create_file_with_values(app, "Good", {"P1": "30", "P2": "40"})
        # Orphan with a name that exists nowhere else and has no values.
        orphan_file, _ = _create_file_with_values(
            app, "Distances", {}, feature_name="distance_x"
        )

        with app.app_context():
            # Previously raised: "None of ['PatientID'] are in the columns".
            features_df = get_clinical_features(
                USER, None, ["P1", "P2"], {"album_id": ALBUM}
            )

        assert not features_df.empty
        assert len(features_df) == 2
        assert any(str(c).startswith(f"{good_file}::") for c in features_df.columns)
        assert not any(
            str(c).startswith(f"{orphan_file}::") for c in features_df.columns
        )

"""
Tests for the multi-CSV clinical-features HTTP routes.

Covers:
- /clinical-features-files: list, create, rename, delete
- /clinical-features-definitions: POST is file-aware, DELETE is disabled
- /clinical-features: save and read values are file-scoped
"""

import json
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


def _patch_album_patients(*patient_ids):
    """Stub the album's Kheops patient list used by the save endpoint."""
    return patch(
        "routes.clinical_features.get_album_patient_ids",
        return_value=set(patient_ids),
    )


ALBUM = "alb-multi"


class TestClinicalFeatureFiles:
    def test_list_empty(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            r = client.get(f"/clinical-features-files?album_id={ALBUM}")
        assert r.status_code == 200
        assert r.get_json() == []

    def test_create_and_list(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            r = client.post(
                f"/clinical-features-files?album_id={ALBUM}",
                data=json.dumps({"name": "Cohort A"}),
                content_type="application/json",
            )
            assert r.status_code == 201
            created = r.get_json()
            assert created["name"] == "Cohort A"
            file_id = created["id"]

            r = client.get(f"/clinical-features-files?album_id={ALBUM}")
            assert r.status_code == 200
            files = r.get_json()
            assert len(files) == 1 and files[0]["id"] == file_id

    def test_create_dedupes_name(self, client, app, db_session):
        """Posting the same name twice produces ' (2)' rather than failing."""
        with _patch_validate_decorate(), _patch_decode_token():
            client.post(
                f"/clinical-features-files?album_id={ALBUM}",
                data=json.dumps({"name": "Cohort"}),
                content_type="application/json",
            )
            r = client.post(
                f"/clinical-features-files?album_id={ALBUM}",
                data=json.dumps({"name": "Cohort"}),
                content_type="application/json",
            )
        assert r.status_code == 201
        assert r.get_json()["name"] == "Cohort (2)"

    def test_rename(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            created = client.post(
                f"/clinical-features-files?album_id={ALBUM}",
                data=json.dumps({"name": "Original"}),
                content_type="application/json",
            ).get_json()
            r = client.patch(
                f"/clinical-features-files/{created['id']}",
                data=json.dumps({"name": "Renamed"}),
                content_type="application/json",
            )
        assert r.status_code == 200
        assert r.get_json()["name"] == "Renamed"

    def test_delete_cascades_definitions(self, client, app, db_session):
        from quantimage2_backend_common.models import (
            ClinicalFeatureDefinition,
            ClinicalFeatureFile,
        )

        with _patch_validate_decorate(), _patch_decode_token():
            created = client.post(
                f"/clinical-features-files?album_id={ALBUM}",
                data=json.dumps({"name": "Will be deleted"}),
                content_type="application/json",
            ).get_json()
            file_id = created["id"]

            # Insert a definition under that file directly via the model so we
            # can verify it's gone after DELETE without depending on the values
            # endpoint.
            with app.app_context():
                ClinicalFeatureDefinition(
                    "Age",
                    ALBUM,
                    "test-user-uuid-1234",
                    "Number",
                    "Normalization",
                    "Mean",
                    file_id,
                ).save_to_db()

            r = client.delete(f"/clinical-features-files/{file_id}")
        assert r.status_code == 200

        # In-memory SQLite needs PRAGMA foreign_keys=ON for ondelete=CASCADE
        # to actually fire. The conftest's collation hook doesn't set that, so
        # cleanup is most useful via the model helper:
        with app.app_context():
            ClinicalFeatureDefinition.delete_by_file_id(file_id)
            assert (
                ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
                    "test-user-uuid-1234", ALBUM, file_id
                )
                == []
            )
            assert ClinicalFeatureFile.find_by_id(file_id) is None


class TestClinicalFeatureDefinitionsRoutes:
    def test_post_requires_file_id(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            r = client.post(
                f"/clinical-features-definitions?album_id={ALBUM}",
                data=json.dumps(
                    {
                        "clinical_feature_definitions": {
                            "Age": {
                                "feat_type": "Number",
                                "encoding": "Normalization",
                                "missing_values": "Mean",
                            }
                        }
                    }
                ),
                content_type="application/json",
            )
        # Missing clinical_feature_file_id is rejected with a 400 (BadRequest),
        # not silently accepted.
        assert r.status_code == 400

    def test_post_persists_definitions_under_file(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            file_id = client.post(
                f"/clinical-features-files?album_id={ALBUM}",
                data=json.dumps({"name": "Cohort defs"}),
                content_type="application/json",
            ).get_json()["id"]

            r = client.post(
                f"/clinical-features-definitions?album_id={ALBUM}",
                data=json.dumps(
                    {
                        "clinical_feature_file_id": file_id,
                        "clinical_feature_definitions": {
                            "Age": {
                                "feat_type": "Number",
                                "encoding": "Normalization",
                                "missing_values": "Mean",
                            },
                            "Smoker": {
                                "feat_type": "Categorical",
                                "encoding": "One-Hot Encoding",
                                "missing_values": "Mode",
                            },
                        },
                    }
                ),
                content_type="application/json",
            )

        assert r.status_code == 200
        defs = r.get_json()
        assert len(defs) == 2
        assert all(d["clinical_feature_file_id"] == file_id for d in defs)
        assert {d["name"] for d in defs} == {"Age", "Smoker"}

    def test_legacy_delete_is_disabled(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            r = client.delete(f"/clinical-features-definitions?album_id={ALBUM}")
        assert r.status_code == 410


class TestClinicalFeaturesValuesRoutes:
    def _setup_file_and_defs(self, client, name, columns):
        file_id = client.post(
            f"/clinical-features-files?album_id={ALBUM}",
            data=json.dumps({"name": name}),
            content_type="application/json",
        ).get_json()["id"]
        client.post(
            f"/clinical-features-definitions?album_id={ALBUM}",
            data=json.dumps(
                {
                    "clinical_feature_file_id": file_id,
                    "clinical_feature_definitions": {
                        col: {
                            "feat_type": "Number",
                            "encoding": "None",
                            "missing_values": "Drop",
                        }
                        for col in columns
                    },
                }
            ),
            content_type="application/json",
        )
        return file_id

    def test_save_and_read_namespaces_by_file(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token(), _patch_album_patients(
            "P1"
        ):
            f1 = self._setup_file_and_defs(client, "Cohort 1", ["Age"])
            f2 = self._setup_file_and_defs(client, "Cohort 2", ["Age"])

            client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps(
                    {
                        "clinical_feature_map": {"P1": {"Age": "30"}},
                        "clinical_feature_file_id": f1,
                    }
                ),
                content_type="application/json",
            )
            client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps(
                    {
                        "clinical_feature_map": {"P1": {"Age": "55"}},
                        "clinical_feature_file_id": f2,
                    }
                ),
                content_type="application/json",
            )

            r = client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps({"patient_ids": ["P1"]}),
                content_type="application/json",
            )

        assert r.status_code == 200
        out = r.get_json()
        assert out["P1"][f"{f1}::Age"] == "30"
        assert out["P1"][f"{f2}::Age"] == "55"

    def test_resave_replaces_values_no_duplicates(self, client, app, db_session):
        """Re-uploading values for the same file replaces, not appends (fix #3)."""
        from quantimage2_backend_common.models import ClinicalFeatureValue

        with _patch_validate_decorate(), _patch_decode_token(), _patch_album_patients(
            "P1"
        ):
            f1 = self._setup_file_and_defs(client, "Cohort X", ["Age"])
            for value in ("10", "20"):
                client.post(
                    f"/clinical-features?album_id={ALBUM}",
                    data=json.dumps(
                        {
                            "clinical_feature_map": {"P1": {"Age": value}},
                            "clinical_feature_file_id": f1,
                        }
                    ),
                    content_type="application/json",
                )

            r = client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps({"patient_ids": ["P1"]}),
                content_type="application/json",
            )
        assert r.status_code == 200
        # The read returns the latest value...
        assert r.get_json()["P1"][f"{f1}::Age"] == "20"
        # ...and there is exactly one value row in the DB (no duplicates).
        with app.app_context():
            from quantimage2_backend_common.models import ClinicalFeatureDefinition

            defs = ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
                "test-user-uuid-1234", ALBUM, f1
            )
            values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
                [d.id for d in defs]
            )
            assert len(values) == 1

    def test_save_rejects_file_with_no_matching_patients(self, client, app, db_session):
        """A file whose patients are none of the album's patients is rejected."""
        from quantimage2_backend_common.models import ClinicalFeatureValue

        with _patch_validate_decorate(), _patch_decode_token(), _patch_album_patients(
            "P1"
        ):
            f1 = self._setup_file_and_defs(client, "Wrong cohort", ["Age"])
            r = client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps(
                    {
                        # None of these patients belong to the album (only P1 does).
                        "clinical_feature_map": {"X9": {"Age": "30"}},
                        "clinical_feature_file_id": f1,
                    }
                ),
                content_type="application/json",
            )

        assert r.status_code == 400
        # The rejection reason is surfaced to the user (werkzeug renders the
        # BadRequest description into the response body).
        assert b"match" in r.data.lower()
        with app.app_context():
            from quantimage2_backend_common.models import ClinicalFeatureDefinition

            defs = ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
                "test-user-uuid-1234", ALBUM, f1
            )
            values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
                [d.id for d in defs]
            )
            assert len(values) == 0

    def test_save_keeps_only_album_patients(self, client, app, db_session):
        """A partial-match file saves only the rows that belong to the album."""
        from quantimage2_backend_common.models import ClinicalFeatureValue

        with _patch_validate_decorate(), _patch_decode_token(), _patch_album_patients(
            "P1"
        ):
            f1 = self._setup_file_and_defs(client, "Partial cohort", ["Age"])
            r = client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps(
                    {
                        # P1 is in the album, FOREIGN is not.
                        "clinical_feature_map": {
                            "P1": {"Age": "30"},
                            "FOREIGN": {"Age": "99"},
                        },
                        "clinical_feature_file_id": f1,
                    }
                ),
                content_type="application/json",
            )
            assert r.status_code == 200

            read = client.post(
                f"/clinical-features?album_id={ALBUM}",
                data=json.dumps({"patient_ids": ["P1", "FOREIGN"]}),
                content_type="application/json",
            )

        out = read.get_json()
        assert out["P1"][f"{f1}::Age"] == "30"
        assert "FOREIGN" not in out
        with app.app_context():
            from quantimage2_backend_common.models import ClinicalFeatureDefinition

            defs = ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
                "test-user-uuid-1234", ALBUM, f1
            )
            values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
                [d.id for d in defs]
            )
            assert len(values) == 1
            assert values[0].patient_id == "P1"


class TestClinicalFeatureFileCascade:
    def test_delete_truly_cascades_with_fk_enforced(self, app, db_session):
        """The DELETE route relies on ON DELETE CASCADE; verify it genuinely
        removes definitions and values when SQLite FK enforcement is on (fix #4).

        conftest leaves PRAGMA foreign_keys off globally (many tests create rows
        with dangling FKs), so we enable it only for this connection/test.
        """
        from sqlalchemy import text
        from quantimage2_backend_common.models import (
            db,
            ClinicalFeatureDefinition,
            ClinicalFeatureFile,
            ClinicalFeatureValue,
        )

        with app.app_context():
            # Set the pragma on the raw DBAPI connection in autocommit so it is
            # not silently ignored inside an open transaction.
            db.session.commit()
            raw = db.session.connection().connection
            raw.execute("PRAGMA foreign_keys=ON")
            try:
                f = ClinicalFeatureFile("Cascade file", ALBUM, "test-user-uuid-1234")
                f.save_to_db()
                d = ClinicalFeatureDefinition(
                    "Age",
                    ALBUM,
                    "test-user-uuid-1234",
                    "Number",
                    "Normalization",
                    "Mean",
                    f.id,
                )
                d.save_to_db()
                ClinicalFeatureValue("5", d.id, "P1").save_to_db()
                file_id, def_id = f.id, d.id

                # Same deletion the route performs.
                ClinicalFeatureFile.delete_by_id(file_id)

                assert ClinicalFeatureFile.find_by_id(file_id) is None
                assert (
                    ClinicalFeatureDefinition.find_by_user_id_album_id_and_file_id(
                        "test-user-uuid-1234", ALBUM, file_id
                    )
                    == []
                )
                assert (
                    ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
                        [def_id]
                    )
                    == []
                )
            finally:
                db.session.commit()
                db.session.connection().connection.execute("PRAGMA foreign_keys=OFF")


class TestClinicalFeatureDefinitionsPatch:
    def _create_def(self, client, name="Age"):
        file_id = client.post(
            f"/clinical-features-files?album_id={ALBUM}",
            data=json.dumps({"name": f"File for {name}"}),
            content_type="application/json",
        ).get_json()["id"]
        defs = client.post(
            f"/clinical-features-definitions?album_id={ALBUM}",
            data=json.dumps(
                {
                    "clinical_feature_file_id": file_id,
                    "clinical_feature_definitions": {
                        name: {
                            "feat_type": "Number",
                            "encoding": "Normalization",
                            "missing_values": "Mean",
                        }
                    },
                }
            ),
            content_type="application/json",
        ).get_json()
        return defs[0]

    def test_patch_missing_body_is_400(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            r = client.patch(
                f"/clinical-features-definitions?album_id={ALBUM}",
                data=json.dumps({}),
                content_type="application/json",
            )
        assert r.status_code == 400

    def test_patch_missing_id_is_400(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            r = client.patch(
                f"/clinical-features-definitions?album_id={ALBUM}",
                data=json.dumps(
                    {"clinical_feature_definitions": [{"name": "Age", "encoding": "x"}]}
                ),
                content_type="application/json",
            )
        assert r.status_code == 400

    def test_patch_owned_definition_succeeds(self, client, app, db_session):
        with _patch_validate_decorate(), _patch_decode_token():
            d = self._create_def(client)
            d["encoding"] = "None"
            r = client.patch(
                f"/clinical-features-definitions?album_id={ALBUM}",
                data=json.dumps({"clinical_feature_definitions": [d]}),
                content_type="application/json",
            )
        assert r.status_code == 200

    def test_patch_foreign_definition_is_rejected(self, client, app, db_session):
        """A user cannot update a definition owned by someone else (fix #2, IDOR)."""
        from quantimage2_backend_common.models import (
            ClinicalFeatureFile,
            ClinicalFeatureDefinition,
        )

        with app.app_context():
            other_file = ClinicalFeatureFile(
                name="Other user file", album_id=ALBUM, user_id="other-user-9999"
            )
            other_file.save_to_db()
            other_def = ClinicalFeatureDefinition(
                "Age",
                ALBUM,
                "other-user-9999",
                "Number",
                "Normalization",
                "Mean",
                other_file.id,
            )
            other_def.save_to_db()
            foreign_id = other_def.id

        with _patch_validate_decorate(), _patch_decode_token():
            r = client.patch(
                f"/clinical-features-definitions?album_id={ALBUM}",
                data=json.dumps(
                    {
                        "clinical_feature_definitions": [
                            {"id": foreign_id, "name": "HACKED", "encoding": "None"}
                        ]
                    }
                ),
                content_type="application/json",
            )
        assert r.status_code == 404
        # The foreign row must be unchanged.
        with app.app_context():
            assert ClinicalFeatureDefinition.find_by_id(foreign_id).name == "Age"


class TestResolveCollectionClinicalDefinitions:
    """Unit tests for the legacy/namespaced feature_id resolver (fix #1)."""

    def _defs(self):
        from types import SimpleNamespace

        # Same name "Age" in two files; lower file_id is the (backfilled) original.
        return [
            SimpleNamespace(id=10, name="Age", clinical_feature_file_id=1),
            SimpleNamespace(id=11, name="Age", clinical_feature_file_id=2),
            SimpleNamespace(id=12, name="Sex", clinical_feature_file_id=2),
        ]

    def test_legacy_bare_name_resolves_to_single_lowest_file(self):
        from service.machine_learning import resolve_collection_clinical_definitions

        result = resolve_collection_clinical_definitions(["Age"], self._defs())
        # Exactly one Age definition, from the lowest file_id (1), not both.
        assert len(result) == 1
        assert result[0].id == 10

    def test_namespaced_id_selects_exact_file(self):
        from service.machine_learning import resolve_collection_clinical_definitions

        result = resolve_collection_clinical_definitions(["2::Age"], self._defs())
        assert len(result) == 1
        assert result[0].id == 11

    def test_no_duplicate_when_exact_and_legacy_overlap(self):
        from service.machine_learning import resolve_collection_clinical_definitions

        result = resolve_collection_clinical_definitions(
            ["1::Age", "Age"], self._defs()
        )
        assert len(result) == 1 and result[0].id == 10

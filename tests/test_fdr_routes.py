"""
Tests for the univariate screening route (POST /fdr/simpleFDR) together with
the database behind it: ownership of the objects named in the request, clinical
features the frontend can select but the backend never builds, and clinical
values stored more than once for the same patient.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from quantimage2_backend_common.const import CLINICAL_FEATURE_ID_SEPARATOR

USER = "test-user-uuid-1234"
OTHER_USER = "other-user-uuid-5678"
ALBUM = "alb-fdr"
PATIENTS = [f"P{i}" for i in range(1, 9)]
RADIOMICS_ID = "CT‑GTV‑original_firstorder_Mean"


def _patch_decode_token():
    return patch(
        "routes.utils.decode_token",
        return_value={
            "sub": USER,
            "preferred_username": "testuser",
            "resource_access": {"quantimage2-frontend": {"roles": ["admin"]}},
        },
    )


def _patch_radiomics(classes=("0", "1")):
    """Stand in for get_features_labels, which reads extraction feature files."""
    index = pd.Index(PATIENTS, name="PatientID")
    features = pd.DataFrame(
        {"PatientID": PATIENTS, RADIOMICS_ID: [float(i) for i in range(1, 9)]},
        index=index,
    )
    labels = pd.DataFrame({"Outcome": list(classes) * 4}, index=index)
    return patch("service.FDR.get_features_labels", return_value=(features, labels))


def _post(client, body):
    with _patch_decode_token():
        return client.post(
            "/fdr/simpleFDR", data=json.dumps(body), content_type="application/json"
        )


def _body(extraction, label_category, selected_feature_ids, collection_id=None):
    return {
        "extraction_id": extraction.id,
        "collection_id": collection_id,
        "album": {"album_id": ALBUM},
        "album_studies": [],
        "label_category_id": label_category.id,
        "labels": [],
        "training_patients": PATIENTS,
        "selected_feature_ids": selected_feature_ids,
        # Adjusted p-values never exceed 1, so q=1 lists every feature that was
        # actually tested.
        "fdr_threshold_list": [1.0],
    }


def _tested_features(response):
    return {feature["feature"] for feature in response.get_json()[0]["features"]}


def _make_extraction(user_id=USER, album_id=ALBUM):
    from quantimage2_backend_common.models import FeatureExtraction

    extraction = FeatureExtraction(user_id=user_id, album_id=album_id)
    extraction.save_to_db()
    return extraction


def _make_label_category(user_id=USER, album_id=ALBUM):
    from quantimage2_backend_common.models import LabelCategory

    label_category = LabelCategory(album_id, "Classification", "Outcome", user_id)
    label_category.save_to_db()
    return label_category


def _make_collection(extraction):
    from quantimage2_backend_common.models import FeatureCollection

    collection = FeatureCollection(
        "Selection", extraction.id, [RADIOMICS_ID], None, None, PATIENTS, []
    )
    collection.save_to_db()
    return collection


def _make_clinical_file(name):
    from quantimage2_backend_common.models import ClinicalFeatureFile

    clinical_file = ClinicalFeatureFile(name, ALBUM, USER)
    clinical_file.save_to_db()
    return clinical_file


def _make_clinical_feature(clinical_file, name, rows, feat_type="Categorical"):
    """Store ``rows`` of (patient, value) under a new definition; return its ID."""
    from quantimage2_backend_common.models import (
        ClinicalFeatureDefinition,
        ClinicalFeatureValue,
    )

    definition = ClinicalFeatureDefinition(
        name, ALBUM, USER, feat_type, "None", "None", clinical_file.id
    )
    definition.save_to_db()
    ClinicalFeatureValue.insert_values(
        [
            {
                "clinical_feature_definition_id": definition.id,
                "patient_id": patient_id,
                "value": value,
            }
            for patient_id, value in rows
        ]
    )
    return f"{clinical_file.id}{CLINICAL_FEATURE_ID_SEPARATOR}{name}"


class TestOwnership:
    """Every object named in the request must belong to the caller."""

    def _assert_not_found(self, client, body):
        with patch("routes.FDR.compute_fdr") as compute_fdr:
            response = _post(client, body)
        assert response.status_code == 404
        compute_fdr.assert_not_called()

    def test_other_users_label_category(self, client, db_session):
        body = _body(
            _make_extraction(), _make_label_category(OTHER_USER), [RADIOMICS_ID]
        )
        self._assert_not_found(client, body)

    def test_other_users_extraction(self, client, db_session):
        body = _body(
            _make_extraction(OTHER_USER), _make_label_category(), [RADIOMICS_ID]
        )
        self._assert_not_found(client, body)

    def test_collection_of_another_users_extraction(self, client, db_session):
        their_collection = _make_collection(_make_extraction(OTHER_USER))
        body = _body(
            _make_extraction(),
            _make_label_category(),
            [RADIOMICS_ID],
            collection_id=their_collection.id,
        )
        self._assert_not_found(client, body)

    def test_owned_objects_are_screened(self, client, db_session):
        extraction = _make_extraction()
        collection = _make_collection(extraction)
        body = _body(
            extraction,
            _make_label_category(),
            [RADIOMICS_ID],
            collection_id=collection.id,
        )
        with patch("routes.FDR.compute_fdr", return_value=[]) as compute_fdr:
            response = _post(client, body)
        assert response.status_code == 200
        compute_fdr.assert_called_once()


class TestRequestConsistency:
    """Objects the caller owns must still describe the album being screened."""

    def _assert_rejected(self, client, body):
        with patch("routes.FDR.compute_fdr") as compute_fdr:
            response = _post(client, body)
        assert response.status_code == 400
        compute_fdr.assert_not_called()

    def test_extraction_from_another_album(self, client, db_session):
        body = _body(
            _make_extraction(album_id="alb-other"),
            _make_label_category(),
            [RADIOMICS_ID],
        )
        self._assert_rejected(client, body)

    def test_label_category_from_another_album(self, client, db_session):
        body = _body(
            _make_extraction(),
            _make_label_category(album_id="alb-other"),
            [RADIOMICS_ID],
        )
        self._assert_rejected(client, body)

    @pytest.mark.parametrize("threshold", [1.5, -0.1, "nan", None])
    def test_invalid_fdr_threshold(self, client, db_session, threshold):
        body = _body(_make_extraction(), _make_label_category(), [RADIOMICS_ID])
        # json.dumps writes float("nan") as NaN, which Flask parses back.
        body["fdr_threshold_list"] = [
            float(threshold) if threshold == "nan" else threshold
        ]

        with _patch_radiomics():
            response = _post(client, body)

        assert response.status_code == 400


class TestOutcomeLabels:
    def test_text_classes_are_screened(self, client, db_session):
        """Classes such as "no"/"yes" must not be coerced to NaN, which left
        no patients and skipped every feature."""
        body = _body(_make_extraction(), _make_label_category(), [RADIOMICS_ID])

        with _patch_radiomics(classes=("no", "yes")):
            response = _post(client, body)

        assert response.status_code == 200
        assert _tested_features(response) == {RADIOMICS_ID}

    def test_radiomics_matrix_is_not_imputed(self, client, db_session):
        """Imputing before the training filter would average in test
        patients; missing values are deleted pairwise instead."""
        body = _body(_make_extraction(), _make_label_category(), [RADIOMICS_ID])

        with _patch_radiomics() as get_features_labels:
            _post(client, body)

        assert get_features_labels.call_args.kwargs["impute"] is False


class TestClinicalFeaturesInScreening:
    def test_selected_clinical_feature_that_is_not_built_is_skipped(
        self, client, db_session
    ):
        """The frontend offers the newest file's copy of a duplicated name, but
        the backend keeps the copy that has values. Selecting the empty copy
        must not reject the screening of every other feature."""
        old_center = [(p, "A" if i < 4 else "B") for i, p in enumerate(PATIENTS)]
        _make_clinical_feature(
            _make_clinical_file("Old cohort"), "CenterID", old_center
        )
        new_center_id = _make_clinical_feature(
            _make_clinical_file("New cohort"), "CenterID", []
        )
        body = _body(
            _make_extraction(), _make_label_category(), [RADIOMICS_ID, new_center_id]
        )

        with _patch_radiomics():
            response = _post(client, body)

        assert response.status_code == 200
        assert _tested_features(response) == {RADIOMICS_ID}

    def test_unknown_radiomics_feature_is_still_rejected(self, client, db_session):
        unknown_id = "CT‑GTV‑original_firstorder_Unknown"
        body = _body(
            _make_extraction(), _make_label_category(), [RADIOMICS_ID, unknown_id]
        )

        with _patch_radiomics():
            response = _post(client, body)

        assert response.status_code == 400
        assert unknown_id in response.get_json()["error"]

    def test_duplicate_value_rows_for_a_patient_do_not_fail_screening(
        self, client, db_session
    ):
        """Older uploads could store two rows for one (patient, definition)
        pair, which reindex rejects as duplicate labels."""
        rows = [("P1", "1")] + [(p, str(i % 2)) for i, p in enumerate(PATIENTS)]
        smoker_id = _make_clinical_feature(
            _make_clinical_file("Cohort"), "Smoker", rows, feat_type="Number"
        )
        body = _body(
            _make_extraction(), _make_label_category(), [RADIOMICS_ID, smoker_id]
        )

        with _patch_radiomics():
            response = _post(client, body)

        assert response.status_code == 200
        assert _tested_features(response) == {RADIOMICS_ID, smoker_id}

    def test_latest_stored_value_wins_for_a_duplicated_patient(self, db_session):
        from service.FDR import get_clinical_features

        rows = [("P1", "40"), ("P2", "50"), ("P1", "41")]
        age_id = _make_clinical_feature(
            _make_clinical_file("Cohort"), "Age", rows, feat_type="Number"
        )

        clinical_df, _ = get_clinical_features(
            USER, [age_id], ["P1", "P2"], {"album_id": ALBUM}
        )

        assert clinical_df.loc["P1", age_id] == 41
        assert clinical_df.loc["P2", age_id] == 50

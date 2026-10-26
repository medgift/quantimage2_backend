"""
Audit tests for GET /extractions/<id>/summary (service.extraction_summary).

Adversarial / edge-case inputs not covered by the frozen
tests/test_extraction_summary.py: odd YAML that is valid for yaml.safe_load but
not JSON-serialisable, non-UTF-8 files, deeply nested YAML, numeric modality
keys, symlinked base dirs and user_id edge cases.

The contract (AC4) says the endpoint must never return a 500 and must always
produce a response the browser can parse.
"""

import importlib
import json
import os
from unittest.mock import patch

import pytest

USER = "test-user-uuid-1234"
OTHER_USER = "other-user-uuid-5678"
ALBUM = "alb-summary"
NOT_FOUND = {"error": "Feature extraction not found"}
NOT_OK_CONFIG_SHAPE = {"general": None, "modalities": []}

MINIMAL_OK_YAML = """\
general:
  padding: 5
volume_preprocessing:
  common:
    bspline_resampler:
      resampling_spacing: [1.0, 1.0, 1.0]
      order: 1
feature_extraction:
  CT:
    pyradiomics:
      original:
        imageType:
          Original: {}
        featureClass:
          firstorder:
        setting:
          binWidth: 25
"""


# ---------------------------------------------------------------------------
# Helpers (copied from the frozen test file)
# ---------------------------------------------------------------------------


def _svc():
    return importlib.import_module("service.extraction_summary")


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


def _make_extraction(user_id=USER, album_id=ALBUM, config_file=None, feature_names=()):
    from quantimage2_backend_common.models import (
        FeatureDefinition,
        FeatureExtraction,
        db,
    )

    extraction = FeatureExtraction(user_id=user_id, album_id=album_id)
    extraction.config_file = config_file
    extraction.save_to_db()
    for name in feature_names:
        extraction.feature_definitions.append(FeatureDefinition(name))
    db.session.commit()
    return extraction.id


@pytest.fixture
def configs_base(tmp_path, monkeypatch):
    import config

    base = tmp_path / "extractions"
    monkeypatch.setattr(config, "EXTRACTIONS_BASE_DIR", str(base))
    configs_root = base / config.CONFIGS_SUBDIR
    configs_root.mkdir(parents=True)
    return configs_root


def _write_config(configs_root, user_id, content, name="extraction-1-config.yaml"):
    folder = configs_root / user_id / ALBUM
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return str(path)


def _strict_json(response):
    """Parse like a browser would (JSON.parse rejects NaN / Infinity)."""

    def _reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(response.get_data(as_text=True), parse_constant=_reject)


def _summary_for_yaml(client, app, configs_base, content):
    path = _write_config(configs_base, USER, content)
    with app.app_context():
        extraction_id = _make_extraction(config_file=path)
    return _get_summary(client, extraction_id)


# ===========================================================================
# Odd YAML through the route: never a 500, always valid JSON
# ===========================================================================


class TestOddYamlNeverBreaksTheRoute:
    @pytest.mark.parametrize(
        "label, content",
        [
            (
                "set_in_general",
                MINIMAL_OK_YAML.replace("padding: 5", "padding: !!set {a, b}"),
            ),
            (
                "binary_in_setting",
                MINIMAL_OK_YAML.replace("binWidth: 25", "binWidth: !!binary aGVsbG8="),
            ),
            (
                "set_as_spacing",
                MINIMAL_OK_YAML.replace(
                    "resampling_spacing: [1.0, 1.0, 1.0]",
                    "resampling_spacing: !!set {1, 2}",
                ),
            ),
            (
                "mixed_key_types_in_normalizer_params",
                MINIMAL_OK_YAML.replace(
                    "      order: 1\n",
                    "      order: 1\n    standardizer:\n      1: a\n      b: c\n",
                ),
            ),
            (
                "recursive_alias_in_padding",
                MINIMAL_OK_YAML.replace("padding: 5", "padding: &a [*a]"),
            ),
        ],
    )
    def test_odd_but_safe_yaml_values_do_not_500(
        self, client, app, db_session, configs_base, label, content
    ):
        response = _summary_for_yaml(client, app, configs_base, content)

        assert response.status_code == 200, label
        assert response.get_json()["config"]["status"] in (
            "ok",
            "invalid",
            "unrecognized",
        )

    @pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
    def test_non_finite_numbers_still_give_browser_parsable_json(
        self, client, app, db_session, configs_base, value
    ):
        # Reachable without hand-editing: Python's json.loads (used for the
        # POSTed config) accepts NaN / Infinity, and yaml.dump writes .nan / .inf
        content = MINIMAL_OK_YAML.replace("binWidth: 25", f"binWidth: {value}")

        response = _summary_for_yaml(client, app, configs_base, content)

        assert response.status_code == 200
        _strict_json(response)

    def test_non_utf8_file_is_not_a_500(self, client, app, db_session, configs_base):
        content = b"general:\n  padding: 5\nfeature_extraction:\n  CT: \xff\xfe\x80\n"

        response = _summary_for_yaml(client, app, configs_base, content)

        assert response.status_code == 200
        assert response.get_json()["config"] == {
            "status": "invalid",
            **NOT_OK_CONFIG_SHAPE,
        }

    def test_deeply_nested_yaml_is_not_a_500(
        self, client, app, db_session, configs_base
    ):
        depth = 5000
        content = "general: " + "[" * depth + "]" * depth + "\nfeature_extraction: {}\n"

        response = _summary_for_yaml(client, app, configs_base, content)

        assert response.status_code == 200

    def test_billion_laughs_aliases_are_not_expanded_in_the_response(
        self, client, app, db_session, configs_base
    ):
        # Small version (4^6 leaves) so the test stays fast; the point is that
        # aliased values are echoed verbatim, so the response grows with the
        # alias expansion, not with the file size.
        lines = ["a0: &a0 [x, x, x, x]"]
        for i in range(1, 7):
            prev = f"*a{i - 1}"
            lines.append(f"a{i}: &a{i} [{prev}, {prev}, {prev}, {prev}]")
        content = (
            "\n".join(lines)
            + "\n"
            + MINIMAL_OK_YAML.replace("padding: 5", "padding: *a6")
        )

        response = _summary_for_yaml(client, app, configs_base, content)

        assert response.status_code == 200
        # The file is < 1 KB; a summary that is orders of magnitude larger means
        # unbounded values are echoed back.
        assert len(response.get_data()) < 50 * len(content)


# ===========================================================================
# Modality keys that are not strings
# ===========================================================================


@pytest.mark.unit
class TestNonStringModalityKeys:
    def test_numeric_modality_key_keeps_its_extractors_and_resampler(self):
        config = {
            "volume_preprocessing": {
                1: {"bspline_resampler": {"resampling_spacing": [2, 2, 2], "order": 0}}
            },
            "feature_extraction": {
                1: {
                    "pyradiomics": {
                        "original": {
                            "imageType": {"Original": {}},
                            "featureClass": {"glcm": None},
                            "setting": {"binWidth": 10},
                        }
                    }
                }
            },
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        [modality] = result["modalities"]
        assert modality["modality"] in (1, "1")
        assert modality["resampling"] == {
            "spacing": [2, 2, 2],
            "order": 0,
            "interpolation": "nearest",
        }
        assert len(modality["extractors"]) == 1
        assert modality["extractors"][0]["settings"] == {"binWidth": 10}

    def test_mixed_key_types_do_not_raise(self):
        config = {
            "feature_extraction": {1: None, "CT": None, None: None, 2.5: None},
        }

        result = _svc().parse_extraction_config(config)

        assert result["status"] == "ok"
        assert len(result["modalities"]) == 4

    def test_boolean_order_has_no_interpolation_label(self):
        config = {
            "volume_preprocessing": {
                "CT": {
                    "bspline_resampler": {
                        "resampling_spacing": [1, 1, 1],
                        "order": True,
                    }
                }
            },
        }

        result = _svc().parse_extraction_config(config)

        assert result["modalities"][0]["resampling"]["interpolation"] is None


# ===========================================================================
# Feature-name grouping edge cases
# ===========================================================================


@pytest.mark.unit
class TestFeatureNameEdgeCases:
    @pytest.mark.parametrize(
        "name, family, image_filter",
        [
            ("squareroot_firstorder_Mean", "firstorder", "squareroot"),
            ("logarithm_glrlm_RunLengthNonUniformity", "glrlm", "logarithm"),
            ("log-sigma-3-0-mm-3D_glszm_ZoneEntropy", "glszm", "log"),
            ("wavelet-HHH_firstorder_Mean", "firstorder", "wavelet"),
            ("gradient_ngtdm_Busyness", "ngtdm", "gradient"),
            ("original_shape", "other", None),  # only 2 tokens
            ("", "other", None),
            ("lbp-3D-m1_glcm_Contrast", "other", None),  # not a known prefix
            ("PatientAge", "other", None),  # clinical
        ],
    )
    def test_family_and_filter(self, name, family, image_filter):
        result = _svc().summarize_feature_names([name])

        assert result["total"] == 1
        assert result["by_family"] == {family: 1}
        assert result["filters"] == ({image_filter: 1} if image_filter else {})


# ===========================================================================
# Path checks
# ===========================================================================


class TestPathEdgeCases:
    def test_symlinked_base_dir_still_reads_owner_config(
        self, client, app, db_session, tmp_path, monkeypatch
    ):
        import config

        real_base = tmp_path / "real-extractions"
        (real_base / config.CONFIGS_SUBDIR).mkdir(parents=True)
        link_base = tmp_path / "link-extractions"
        link_base.symlink_to(real_base, target_is_directory=True)
        monkeypatch.setattr(config, "EXTRACTIONS_BASE_DIR", str(link_base))

        # Stored path written through the symlinked base, as save_config would
        path = _write_config(link_base / config.CONFIGS_SUBDIR, USER, MINIMAL_OK_YAML)
        with app.app_context():
            extraction_id = _make_extraction(config_file=path)

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        assert response.get_json()["config"]["status"] == "ok"

    def test_symlinked_base_dir_with_real_path_stored(
        self, client, app, db_session, tmp_path, monkeypatch
    ):
        import config

        real_base = tmp_path / "real-extractions"
        (real_base / config.CONFIGS_SUBDIR).mkdir(parents=True)
        link_base = tmp_path / "link-extractions"
        link_base.symlink_to(real_base, target_is_directory=True)
        monkeypatch.setattr(config, "EXTRACTIONS_BASE_DIR", str(link_base))

        path = _write_config(real_base / config.CONFIGS_SUBDIR, USER, MINIMAL_OK_YAML)
        with app.app_context():
            extraction_id = _make_extraction(config_file=path)

        response = _get_summary(client, extraction_id)

        assert response.get_json()["config"]["status"] == "ok"

    def test_relative_config_file_outside_user_dir_is_refused(
        self, client, app, db_session, configs_base
    ):
        other = _write_config(configs_base, OTHER_USER, MINIMAL_OK_YAML)
        relative = os.path.relpath(other, os.getcwd())
        with app.app_context():
            extraction_id = _make_extraction(config_file=relative)

        response = _get_summary(client, extraction_id)

        assert response.status_code == 200
        assert response.get_json()["config"]["status"] == "missing"

    @pytest.mark.parametrize("user_id", ["", ".", "..", "/"])
    def test_degenerate_user_id_cannot_read_other_users_configs(
        self, client, app, db_session, configs_base, user_id
    ):
        """An owner id that collapses the allowed dir to the configs root (or
        above) must not unlock every user's configs."""
        other = _write_config(configs_base, OTHER_USER, MINIMAL_OK_YAML)
        with app.app_context():
            extraction_id = _make_extraction(user_id=user_id, config_file=other)

        response = _get_summary(client, extraction_id, user=user_id)

        if response.status_code == 200:
            assert response.get_json()["config"]["status"] == "missing"
        else:
            assert response.status_code in (401, 404)

    def test_fifo_config_file_does_not_hang(
        self, client, app, db_session, configs_base
    ):
        folder = configs_base / USER / ALBUM
        folder.mkdir(parents=True)
        fifo = folder / "extraction-1-config.yaml"
        os.mkfifo(fifo)
        with app.app_context():
            extraction_id = _make_extraction(config_file=str(fifo))

        response = _get_summary(client, extraction_id)

        assert response.get_json()["config"]["status"] == "missing"


# ===========================================================================
# Access
# ===========================================================================


class TestAccessEdgeCases:
    @pytest.mark.parametrize("bad_id", ["-1", "0"])
    def test_out_of_range_ids_are_404(
        self, client, app, db_session, configs_base, bad_id
    ):
        response = _get_summary(client, bad_id)

        assert response.status_code == 404

    def test_other_users_extraction_404_is_identical_to_unknown(
        self, client, app, db_session, configs_base
    ):
        path = _write_config(configs_base, OTHER_USER, MINIMAL_OK_YAML)
        with app.app_context():
            extraction_id = _make_extraction(user_id=OTHER_USER, config_file=path)

        owned_by_other = _get_summary(client, extraction_id)
        unknown = _get_summary(client, extraction_id + 1000)

        assert owned_by_other.status_code == unknown.status_code == 404
        assert owned_by_other.get_data() == unknown.get_data()

    def test_post_is_not_allowed(self, client, app, db_session, configs_base):
        with app.app_context():
            extraction_id = _make_extraction()

        with _patch_decode_token():
            response = client.post(f"/extractions/{extraction_id}/summary")

        assert response.status_code == 405

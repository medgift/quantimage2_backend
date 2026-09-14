"""
Tests for univariate feature screening with FDR control (service.FDR):
test selection per feature/outcome type, the survival tests (log-rank and the
Cox Wald test), and the handling of features that cannot be tested.
"""

import numpy as np
import pandas as pd
import pytest

from quantimage2_backend_common.const import MODEL_TYPES

pytestmark = pytest.mark.unit


def _classification_targets(values):
    from service.FDR import OUTCOME_FIELD_CLASSIFICATION

    return pd.Series(values, name=OUTCOME_FIELD_CLASSIFICATION)


def _survival_targets(time, event):
    from service.FDR import (
        OUTCOME_FIELD_SURVIVAL_EVENT,
        OUTCOME_FIELD_SURVIVAL_TIME,
    )

    return pd.DataFrame(
        {OUTCOME_FIELD_SURVIVAL_TIME: time, OUTCOME_FIELD_SURVIVAL_EVENT: event}
    )


class TestClassificationTestSelection:
    def test_normal_continuous_feature_still_uses_mann_whitney(self):
        """No normality pre-test: normally distributed data must not get a t-test."""
        from service.FDR import (
            CATEGORY_CONTINUOUS,
            TEST_MANNWHITNEYU,
            _run_univariate_test,
        )

        rng = np.random.default_rng(0)
        targets = _classification_targets([0] * 40 + [1] * 40)
        series = pd.Series(
            np.r_[rng.normal(0, 1, 40), rng.normal(2, 1, 40)], name="feature"
        )

        result = _run_univariate_test(
            series, targets, CATEGORY_CONTINUOUS, MODEL_TYPES.CLASSIFICATION
        )

        assert result.test_used == TEST_MANNWHITNEYU
        assert result.pvalue < 0.001

    def test_non_normal_continuous_feature_uses_mann_whitney(self):
        from service.FDR import (
            CATEGORY_CONTINUOUS,
            TEST_MANNWHITNEYU,
            _run_univariate_test,
        )

        rng = np.random.default_rng(1)
        targets = _classification_targets([0] * 40 + [1] * 40)
        series = pd.Series(
            np.r_[rng.exponential(1, 40), rng.exponential(6, 40)], name="feature"
        )

        result = _run_univariate_test(
            series, targets, CATEGORY_CONTINUOUS, MODEL_TYPES.CLASSIFICATION
        )

        assert result.test_used == TEST_MANNWHITNEYU
        assert result.pvalue < 0.01

    def test_more_than_two_classes_uses_kruskal_wallis(self):
        """classes[0]/classes[1] used to silently drop every further class."""
        from service.FDR import CATEGORY_CONTINUOUS, TEST_KRUSKAL, _run_univariate_test

        rng = np.random.default_rng(2)
        targets = _classification_targets([0] * 30 + [1] * 30 + [2] * 30)
        series = pd.Series(
            np.r_[rng.normal(0, 1, 30), rng.normal(3, 1, 30), rng.normal(6, 1, 30)],
            name="feature",
        )

        result = _run_univariate_test(
            series, targets, CATEGORY_CONTINUOUS, MODEL_TYPES.CLASSIFICATION
        )

        assert result.test_used == TEST_KRUSKAL
        assert result.pvalue < 0.001

    def test_sparse_two_by_two_table_uses_fisher(self):
        from service.FDR import CATEGORY_BINARY, TEST_FISHER, _run_univariate_test

        # Small and unbalanced, so 3 of the 4 expected counts fall below 5.
        targets = _classification_targets([0] * 8 + [1] * 4)
        series = pd.Series(["a"] * 7 + ["b"] + ["a"] + ["b"] * 3, name="feature")

        result = _run_univariate_test(
            series, targets, CATEGORY_BINARY, MODEL_TYPES.CLASSIFICATION
        )

        assert result.test_used == TEST_FISHER
        assert not np.isnan(result.pvalue)

    def test_sparse_nominal_table_falls_back_to_chi_square(self):
        """scipy's fisher_exact only accepts 2x2; a kx2 table used to raise."""
        from service.FDR import CATEGORY_NOMINAL, TEST_CHI2, _run_univariate_test

        targets = _classification_targets([0, 1] * 9)
        series = pd.Series(["a", "b", "c"] * 6, name="feature")

        result = _run_univariate_test(
            series, targets, CATEGORY_NOMINAL, MODEL_TYPES.CLASSIFICATION
        )

        assert result.test_used == TEST_CHI2
        assert not np.isnan(result.pvalue)


class TestSurvivalTests:
    def test_categorical_feature_uses_logrank(self):
        from service.FDR import CATEGORY_BINARY, TEST_LOGRANK, _run_univariate_test

        rng = np.random.default_rng(3)
        group = np.array(["low"] * 60 + ["high"] * 60)
        time = rng.exponential(np.where(group == "high", 3.0, 15.0))
        targets = _survival_targets(time, (rng.random(120) < 0.8).astype(int))
        series = pd.Series(group, name="feature")

        result = _run_univariate_test(
            series, targets, CATEGORY_BINARY, MODEL_TYPES.SURVIVAL
        )

        assert result.test_used == TEST_LOGRANK
        assert result.pvalue < 0.01

    def test_logrank_handles_more_than_two_groups(self):
        from service.FDR import CATEGORY_NOMINAL, TEST_LOGRANK, _run_univariate_test

        rng = np.random.default_rng(4)
        group = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
        scale = np.select([group == "a", group == "b", group == "c"], [3.0, 9.0, 27.0])
        targets = _survival_targets(
            rng.exponential(scale), (rng.random(120) < 0.8).astype(int)
        )

        result = _run_univariate_test(
            pd.Series(group, name="feature"),
            targets,
            CATEGORY_NOMINAL,
            MODEL_TYPES.SURVIVAL,
        )

        assert result.test_used == TEST_LOGRANK
        assert result.pvalue < 0.01

    def test_continuous_feature_uses_cox_wald(self):
        from service.FDR import CATEGORY_CONTINUOUS, TEST_COX_WALD, _run_univariate_test

        rng = np.random.default_rng(5)
        x = rng.normal(size=150)
        targets = _survival_targets(
            rng.exponential(np.exp(-0.9 * x)), (rng.random(150) < 0.8).astype(int)
        )

        result = _run_univariate_test(
            pd.Series(x, name="feature"),
            targets,
            CATEGORY_CONTINUOUS,
            MODEL_TYPES.SURVIVAL,
        )

        assert result.test_used == TEST_COX_WALD
        assert result.pvalue < 0.001

    def test_cox_wald_pvalue_is_scale_invariant(self):
        """Raw radiomics scales vary hugely; the p-value must not care."""
        from service.FDR import CATEGORY_CONTINUOUS, _run_univariate_test

        rng = np.random.default_rng(6)
        x = rng.normal(size=150)
        targets = _survival_targets(
            rng.exponential(np.exp(-0.7 * x)), (rng.random(150) < 0.8).astype(int)
        )

        unscaled = _run_univariate_test(
            pd.Series(x, name="f"), targets, CATEGORY_CONTINUOUS, MODEL_TYPES.SURVIVAL
        )
        scaled = _run_univariate_test(
            pd.Series(x * 10_000, name="f"),
            targets,
            CATEGORY_CONTINUOUS,
            MODEL_TYPES.SURVIVAL,
        )

        assert unscaled.pvalue == pytest.approx(scaled.pvalue)

    def test_constant_feature_is_skipped_not_fatal(self):
        """A zero-variance column makes the Cox fit singular."""
        from service.FDR import CATEGORY_CONTINUOUS, _run_univariate_test

        rng = np.random.default_rng(7)
        targets = _survival_targets(
            rng.exponential(size=60), (rng.random(60) < 0.8).astype(int)
        )

        result = _run_univariate_test(
            pd.Series(np.ones(60), name="feature"),
            targets,
            CATEGORY_CONTINUOUS,
            MODEL_TYPES.SURVIVAL,
        )

        assert result.skipped_reason is not None
        assert np.isnan(result.pvalue)

    def test_too_few_events_is_skipped(self):
        from service.FDR import CATEGORY_CONTINUOUS, _run_univariate_test

        rng = np.random.default_rng(8)
        event = np.zeros(60, dtype=int)
        event[0] = 1
        targets = _survival_targets(rng.exponential(size=60), event)

        result = _run_univariate_test(
            pd.Series(rng.normal(size=60), name="feature"),
            targets,
            CATEGORY_CONTINUOUS,
            MODEL_TYPES.SURVIVAL,
        )

        assert result.skipped_reason == "too few observed events"


class TestSkippedFeatures:
    def test_unsupported_category_is_skipped(self):
        """This path used to reach float(None) and raise TypeError."""
        from service.FDR import CATEGORY_UNSUPPORTED, _run_univariate_test

        result = _run_univariate_test(
            pd.Series([f"note-{i}" for i in range(20)], name="feature"),
            _classification_targets([0, 1] * 10),
            CATEGORY_UNSUPPORTED,
            MODEL_TYPES.CLASSIFICATION,
        )

        assert result.test_used is None
        assert np.isnan(result.pvalue)

    def test_missing_values_are_deleted_pairwise(self):
        from service.FDR import _drop_incomplete_observations

        series = pd.Series([1.0, np.nan, 3.0, 4.0], name="feature")
        targets = _classification_targets([0, 1, np.nan, 1])

        clean_series, clean_targets = _drop_incomplete_observations(series, targets)

        assert list(clean_series) == [1.0, 4.0]
        assert list(clean_targets) == [0, 1]


class TestFdrCorrection:
    def test_untestable_feature_does_not_void_the_correction(self):
        """multipletests returns all-NaN if any single input p-value is NaN."""
        from service.FDR import apply_fdr_correction

        results = pd.DataFrame(
            {
                "pvalue": [0.001, 0.02, np.nan, 0.5],
                "skipped_reason": [None, None, "feature is constant", None],
            },
            index=pd.Index(["a", "b", "c", "d"], name="feature"),
        )

        corrected = apply_fdr_correction(results)

        assert corrected.loc[["a", "b", "d"], "pval_adj"].notna().all()
        assert np.isnan(corrected.loc["c", "pval_adj"])

    def test_adjusted_pvalues_match_benjamini_hochberg(self):
        from service.FDR import apply_fdr_correction

        results = pd.DataFrame(
            {"pvalue": [0.01, 0.02, 0.03, 0.04], "skipped_reason": [None] * 4},
            index=pd.Index(list("abcd"), name="feature"),
        )

        corrected = apply_fdr_correction(results)

        # BH: p * n / rank, enforced monotone non-decreasing.
        assert list(corrected["pval_adj"]) == pytest.approx([0.04, 0.04, 0.04, 0.04])

    def test_response_payload_is_json_serialisable(self):
        """Strict JSON: no numpy types, and no Infinity or NaN."""
        import json

        from service.FDR import apply_fdr_correction, build_threshold_results

        results = apply_fdr_correction(
            pd.DataFrame(
                {
                    "pvalue": [0.0001, 0.02, np.nan, 0.5],
                    "test_used": ["cox_wald", "fisher_exact", None, "chi_square"],
                    "statistic": [np.nan, np.inf, np.nan, 0.3],
                    "skipped_reason": [None, None, "feature is constant", None],
                },
                index=pd.Index(["f1", "f2", "f3", "f4"], name="feature"),
            )
        )

        payload = build_threshold_results(results, [0.01, 0.05, 0.9])
        # allow_nan=False rejects Infinity and NaN, as the browser's JSON.parse does.
        encoded = json.dumps(payload, allow_nan=False)

        assert payload[0]["featureCount"] == 1
        assert payload[0]["features"][0]["feature"] == "f1"
        assert payload[0]["features"][0]["test_used"] == "cox_wald"
        # Non-finite statistics go out as null: NaN for f1, inf for f2.
        assert payload[0]["features"][0]["statistic"] is None
        assert payload[1]["features"][1]["statistic"] is None
        # The untestable feature is absent at every threshold, including 0.9.
        assert "f3" not in encoded

    def test_infinite_fisher_odds_ratio_is_sent_as_null(self):
        """Fisher's odds ratio is infinite when an off-diagonal cell is zero.

        Flask wrote it as Infinity, the browser's JSON.parse rejected the whole
        response, and the page went blank.
        """
        import json

        from service.FDR import (
            CATEGORY_BINARY,
            OUTCOME_FIELD_CLASSIFICATION,
            TEST_FISHER,
            apply_fdr_correction,
            build_threshold_results,
            compute_univariate_tests,
        )

        # Contingency table [[5, 0], [2, 4]]: no patient with "a" has outcome 1.
        index = pd.Index([f"P{i}" for i in range(11)], name="PatientID")
        features_df = pd.DataFrame({"1::Smoker": ["a"] * 5 + ["b"] * 6}, index=index)
        labels_df = pd.DataFrame(
            {OUTCOME_FIELD_CLASSIFICATION: [0] * 5 + [0] * 2 + [1] * 4}, index=index
        )

        results = compute_univariate_tests(
            features_df,
            labels_df,
            {"1::Smoker": {"univariate_category": CATEGORY_BINARY}},
            MODEL_TYPES.CLASSIFICATION,
        )
        # Guard the fixture: it must really produce an infinite odds ratio.
        assert results.loc["1::Smoker", "test_used"] == TEST_FISHER
        assert np.isinf(results.loc["1::Smoker", "statistic"])

        payload = build_threshold_results(
            apply_fdr_correction(results), [0.05, 0.1, 0.9]
        )

        json.dumps(payload, allow_nan=False)  # raises on Infinity or NaN
        assert payload[-1]["featureCount"] == 1
        assert payload[-1]["features"][0]["statistic"] is None

    def test_threshold_is_inclusive(self):
        """BH rejects at p_adj <= q; strict < dropped features exactly on it."""
        from service.FDR import build_threshold_results

        results = pd.DataFrame(
            {
                "pval_adj": [0.05],
                "pvalue": [0.05],
                "test_used": ["logrank"],
                "statistic": [1.0],
            },
            index=pd.Index(["f1"], name="feature"),
        )

        assert build_threshold_results(results, [0.05])[0]["featureCount"] == 1

    def test_all_features_untestable_yields_no_selection(self):
        from service.FDR import apply_fdr_correction

        results = pd.DataFrame(
            {"pvalue": [np.nan, np.nan], "skipped_reason": ["constant", "constant"]},
            index=pd.Index(["a", "b"], name="feature"),
        )

        corrected = apply_fdr_correction(results)

        assert corrected["pval_adj"].isna().all()
        # NaN <= threshold is False, so nothing is ever selected.
        assert len(corrected[corrected["pval_adj"] <= 0.05].index) == 0

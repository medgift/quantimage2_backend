"""Univariate feature screening with Benjamini-Hochberg FDR control.

Each selected feature is tested on its own against the outcome, then the whole
set of p-values is corrected for multiple testing. The test is picked from the
feature's measurement scale and the outcome type:

===============  ==================  ==============================================
Outcome          Feature             Test
===============  ==================  ==============================================
Classification   continuous          Mann-Whitney U (Kruskal-Wallis for 3+ classes)
Classification   binary / nominal    chi-square, or Fisher (2x2)
Survival         continuous          Wald test on a univariate Cox
Survival         binary / nominal    k-sample log-rank
===============  ==================  ==============================================

A feature that cannot be tested (constant, too few observations, a Cox fit that
does not converge) is *skipped* rather than fatal: it gets a NaN p-value, is
held out of the correction, and never appears in the selection. This matters
because ``multipletests`` returns an all-NaN array if a single input p-value is
NaN, which would silently void the correction for every other feature.
"""

import logging
import math
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    kruskal,
    mannwhitneyu,
)
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff
from statsmodels.stats.multitest import multipletests

from quantimage2_backend_common.const import (
    CLINICAL_FEATURE_ID_SEPARATOR,
    MODEL_TYPES,
)
from quantimage2_backend_common.models import (
    ClinicalFeatureMissingValues,
    ClinicalFeatureTypes,
    ClinicalFeatureValue,
)
from service.feature_transformation import (
    OUTCOME_FIELD_CLASSIFICATION,
    OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME,
)
from service.machine_learning import get_features_labels, resolve_clinical_definitions

logger = logging.getLogger(__name__)

# Measurement scale of a feature, which decides the test used
CATEGORY_CONTINUOUS = "continuous"
CATEGORY_BINARY = "binary"
CATEGORY_NOMINAL = "nominal"
CATEGORY_UNSUPPORTED = "unsupported"

# Test identifiers reported back per feature
TEST_MANNWHITNEYU = "mannwhitneyu"
TEST_KRUSKAL = "kruskal_wallis"
TEST_CHI2 = "chi_square"
TEST_FISHER = "fisher_exact"
TEST_LOGRANK = "logrank"
TEST_COX_WALD = "cox_wald"

FDR_METHOD = "fdr_bh"

# No test says anything useful about a group smaller than this.
MIN_GROUP_SIZE = 3
# A Cox fit on one or two events is meaningless and usually singular.
MIN_EVENTS = 3
# A categorical feature with more distinct values than this is treated as
# continuous (numeric) or as unsupported free text (non-numeric).
MAX_CATEGORIES = 10
# The chi-square approximation degrades once too many expected counts are small.
LOW_EXPECTED_COUNT = 5
MAX_LOW_EXPECTED_PROPORTION = 0.2


class UnivariateResult(NamedTuple):
    """Outcome of testing one feature. ``pvalue`` is NaN when it was skipped."""

    test_used: Optional[str]
    statistic: float
    pvalue: float
    skipped_reason: Optional[str]


def _skipped(reason: str) -> UnivariateResult:
    return UnivariateResult(None, float("nan"), float("nan"), reason)


def _yield_to_event_loop():
    """Let other greenlets run during a long CPU-bound screening loop."""
    try:
        import eventlet

        eventlet.sleep(0)
    except ImportError:  # not running under eventlet (tests, CLI)
        pass


def _json_number(value):
    """``value`` as a float that is valid JSON, or None when it is not finite.

    Non-finite values do occur: Fisher's odds ratio is infinite whenever an
    off-diagonal cell of the 2x2 table is zero. Flask writes that as
    ``Infinity``, which is not JSON, so the browser's JSON.parse rejected the
    whole response. The float() call also turns numpy scalars into Python ones.
    """
    value = float(value)
    return value if math.isfinite(value) else None


def compute_fdr(
    extraction_id,
    collection_id,
    album,
    studies,
    label_category,
    gt,
    training_patients,
    user_id,
    selected_feature_ids,
    fdr_threshold_list,
):
    """Screen ``selected_feature_ids`` and report what survives each q-value."""
    model_type = MODEL_TYPES(label_category.label_type)

    features_df, labels_df, feature_metadata = assemble_features_for_collection(
        extraction_id,
        collection_id,
        album,
        studies,
        model_type,
        gt,
        training_patients,
        user_id,
    )

    missing = set(selected_feature_ids) - set(features_df.columns)
    if missing:
        raise ValueError(f"Selected features not found in dataset: {sorted(missing)}")

    # dict.fromkeys keeps the caller's order while dropping any duplicate, which
    # would otherwise produce a duplicated column and be counted twice by the
    # multiple-testing correction.
    selected_feature_ids = list(dict.fromkeys(selected_feature_ids))
    features_df = features_df[selected_feature_ids]
    feature_metadata = {
        feature: metadata
        for feature, metadata in feature_metadata.items()
        if feature in set(selected_feature_ids)
    }

    univariate_results_df = compute_univariate_tests(
        features_df, labels_df, feature_metadata, model_type
    )
    univariate_results_df = apply_fdr_correction(univariate_results_df)

    skipped = univariate_results_df["skipped_reason"].notna().sum()
    if skipped:
        logger.warning(
            "FDR: %d of %d features could not be tested and were excluded "
            "from the correction (reasons: %s)",
            skipped,
            len(univariate_results_df.index),
            sorted(set(univariate_results_df["skipped_reason"].dropna())),
        )

    return build_threshold_results(univariate_results_df, fdr_threshold_list)


def build_threshold_results(univariate_results_df, fdr_threshold_list):
    """The response body: what survives each q-value the frontend asks about.

    Every number goes through :func:`_json_number`, so the body is strict JSON.
    """
    results_by_qvalues = []

    for threshold in fdr_threshold_list:
        # BH rejects at p_adj <= q. NaN comparisons are False, so a feature that
        # could not be tested is never selected.
        selected_df = univariate_results_df[
            univariate_results_df["pval_adj"] <= threshold
        ]
        results_by_qvalues.append(
            {
                "qvalue": _json_number(threshold),
                "featureCount": len(selected_df.index),
                "features": [
                    {
                        "feature": str(feature_name),
                        "pval_adj": _json_number(row["pval_adj"]),
                        "pvalue": _json_number(row["pvalue"]),
                        "test_used": row["test_used"],
                        "statistic": _json_number(row["statistic"]),
                    }
                    for feature_name, row in selected_df.iterrows()
                ],
            }
        )

    return results_by_qvalues


def assemble_features_for_collection(
    extraction_id,
    collection_id,
    album,
    studies,
    model_type,
    gt,
    training_patients,
    user_id,
):
    """Build the (features, labels, metadata) triple the screening runs on."""
    if model_type == MODEL_TYPES.CLASSIFICATION:
        outcome_columns = [OUTCOME_FIELD_CLASSIFICATION]
    elif model_type == MODEL_TYPES.SURVIVAL:
        outcome_columns = [OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT]
    else:
        raise NotImplementedError(f"No univariate screening for {model_type}")

    features_df, labels_df_indexed = get_features_labels(
        extraction_id, collection_id, studies, gt, outcome_columns=outcome_columns
    )

    features_df = features_df.loc[features_df.index.isin(training_patients)]
    labels_df_indexed = labels_df_indexed.loc[
        labels_df_indexed.index.isin(training_patients)
    ]
    labels_df_indexed = labels_df_indexed.apply(pd.to_numeric, errors="coerce")

    # get_features_labels already mean-imputes the radiomics matrix, and every
    # radiomics feature is a continuous measurement.
    feature_metadata = {
        feature: {
            "feat_type": ClinicalFeatureTypes.NUMBER.value,
            "univariate_category": CATEGORY_CONTINUOUS,
        }
        for feature in features_df.columns
        # PatientID is carried along as a column by concatenate_modalities_rois.
        if feature != "PatientID"
    }

    # Indexed on the requested patients rather than on features_df.index, so a
    # collection holding only clinical features (where features_df carries just
    # the PatientID column) still produces rows.
    clinical_features_df, clinical_metadata = get_clinical_features(
        user_id, collection_id, training_patients, album
    )
    feature_metadata.update(clinical_metadata)

    # feature_metadata already excludes PatientID, which is always present.
    has_radiomics = len(feature_metadata) > len(clinical_metadata)
    has_clinical = len(clinical_features_df.columns) > 0

    if has_radiomics and has_clinical:
        features_df = pd.merge(
            features_df,
            clinical_features_df,
            left_index=True,
            right_index=True,
            how="left",
        )
    elif has_clinical:
        features_df = clinical_features_df
    elif not has_radiomics:
        raise ValueError("Neither clinical nor imaging features were selected")

    return features_df, labels_df_indexed, feature_metadata


def get_clinical_features(
    user_id: str, collection_id: str, radiomics_patient_ids: List[str], album
):
    """Clinical feature columns for the given patients, plus their metadata.

    Always returns a ``(DataFrame, dict)`` pair; the frame has no columns when
    the album has no usable clinical features.
    """
    radiomics_index = pd.Index(radiomics_patient_ids, name="PatientID")
    empty = (pd.DataFrame(index=radiomics_index), {})

    clin_feature_definitions = resolve_clinical_definitions(
        user_id, collection_id, album
    )
    if not clin_feature_definitions:
        return empty

    feature_frames = []
    feature_metadata = {}

    for clin_feature in clin_feature_definitions:
        col_name = (
            f"{clin_feature.clinical_feature_file_id}"
            f"{CLINICAL_FEATURE_ID_SEPARATOR}{clin_feature.name}"
        )

        values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
            [clin_feature.id]
        )
        if not values:
            # A definition can exist with no stored values (e.g. an upload whose
            # patients didn't match the album). It carries no data, so skip it.
            continue

        df = pd.DataFrame.from_dict([v.to_dict() for v in values])[
            ["patient_id", "value"]
        ]
        df = df.rename(columns={"value": col_name, "patient_id": "PatientID"})
        df = df.set_index("PatientID").reindex(radiomics_index)

        df[col_name] = correct_type_of_clinical_values(
            df[col_name], clin_feature.feat_type
        )
        df[col_name] = apply_missing_value_strategy(
            df[col_name], ClinicalFeatureMissingValues(clin_feature.missing_values)
        )

        feature_frames.append(df[[col_name]])
        feature_metadata[col_name] = {
            "feat_type": clin_feature.feat_type,
            "univariate_category": get_clinical_feature_univariate_type(
                df[col_name], clin_feature.feat_type
            ),
        }

    if not feature_frames:
        return empty

    raw_df = pd.concat(feature_frames, axis=1)
    raw_df.index.name = "PatientID"
    return raw_df, feature_metadata


def apply_missing_value_strategy(series, strategy):
    """Impute per the user's configured strategy.

    DROP and NONE deliberately leave NaNs in place: the screening deletes those
    patients pairwise for that feature only, rather than dropping them from
    every other feature's test too.
    """
    series = series.copy()

    if not series.isna().any():
        return series

    if strategy == ClinicalFeatureMissingValues.MEAN:
        return series.fillna(series.mean())
    if strategy == ClinicalFeatureMissingValues.MEDIAN:
        return series.fillna(series.median())
    if strategy == ClinicalFeatureMissingValues.MODE:
        mode = series.mode()
        return series.fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return series


def correct_type_of_clinical_values(series: pd.Series, feat_type: str) -> pd.Series:
    series = series.copy()

    if feat_type == ClinicalFeatureTypes.CATEGORICAL.value:
        return series.astype("string")
    if feat_type == ClinicalFeatureTypes.NUMBER.value:
        return pd.to_numeric(series, errors="coerce")

    return series


def get_clinical_feature_univariate_type(
    series, feat_type, max_categories: int = MAX_CATEGORIES
):
    """Measurement scale of a clinical feature, which decides the test used."""
    n_unique = series.nunique()

    if n_unique == 2:
        return CATEGORY_BINARY
    if n_unique <= max_categories:
        return CATEGORY_NOMINAL
    if feat_type == ClinicalFeatureTypes.NUMBER.value:
        return CATEGORY_CONTINUOUS

    # High-cardinality non-numeric text (free-form notes, identifiers): a
    # contingency table would be one patient per cell and tell us nothing.
    return CATEGORY_UNSUPPORTED


def compute_univariate_tests(features_df, labels_df, feature_metadata, model_type):
    """Run one test per feature. Returns a frame indexed by feature name."""
    # The clinical merge and the label filtering can leave the two frames with
    # different patients; test only those present in both.
    common_index = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_index]
    labels_df = labels_df.loc[common_index]

    # Survival needs both outcome columns; classification is a single column.
    targets = (
        labels_df
        if model_type == MODEL_TYPES.SURVIVAL
        else labels_df[OUTCOME_FIELD_CLASSIFICATION]
    )

    records = []
    for feature in features_df.columns:
        category = feature_metadata.get(feature, {}).get(
            "univariate_category", CATEGORY_UNSUPPORTED
        )
        series, feature_targets = _drop_incomplete_observations(
            features_df[feature], targets
        )
        result = _run_univariate_test(series, feature_targets, category, model_type)

        records.append(
            {
                "feature": feature,
                "test_used": result.test_used,
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
                "skipped_reason": result.skipped_reason,
            }
        )

        # This loop runs on the single eventlet thread that serves every user,
        # and a Cox fit never waits on I/O, so without a yield the backend stops
        # answering anyone until it ends. Yield after every feature: answering a
        # request takes several trips through the event loop, so yielding every
        # 25 fits still kept other users waiting ~5 s at n=300 (~0.4 s now).
        _yield_to_event_loop()

    if not records:
        raise ValueError("No features were available for univariate screening")

    return pd.DataFrame(records).set_index("feature").sort_values(by="pvalue")


def _drop_incomplete_observations(series, targets):
    """Pairwise deletion of patients missing this feature or the outcome."""
    mask = series.notna()
    if isinstance(targets, pd.DataFrame):
        mask &= targets.notna().all(axis=1)
    else:
        mask &= targets.notna()
    return series[mask], targets[mask]


def _run_univariate_test(series, targets, category, model_type):
    if category == CATEGORY_UNSUPPORTED:
        return _skipped("no univariate test for this feature type")
    if len(series) < MIN_GROUP_SIZE:
        return _skipped("too few complete observations")
    if series.nunique() < 2:
        return _skipped("feature is constant")

    try:
        if model_type == MODEL_TYPES.SURVIVAL:
            return _run_survival_test(series, targets, category)
        return _run_classification_test(series, targets, category)
    except Exception:
        # One pathological column (singular Cox fit, degenerate contingency
        # table) must not fail the whole screening: skip it and carry on.
        logger.debug("Univariate test failed for %s", series.name, exc_info=True)
        return _skipped("test failed on this feature")


def _run_classification_test(series, targets, category):
    classes = pd.unique(targets)
    if len(classes) < 2:
        return _skipped("outcome has fewer than two classes")

    if category == CATEGORY_CONTINUOUS:
        groups = [series[targets == label] for label in classes]
        if any(len(group) < MIN_GROUP_SIZE for group in groups):
            return _skipped("a class has too few observations")

        # Rank tests throughout, with no normality pre-test. Radiomics features
        # are rarely normal, and letting the data choose between a t-test and a
        # rank test makes the p-values depend on that choice and mixes two test
        # families inside one correction. Kruskal-Wallis is the extension of
        # Mann-Whitney to three or more classes.
        if len(classes) > 2:
            statistic, pvalue = kruskal(*groups)
            return UnivariateResult(TEST_KRUSKAL, statistic, pvalue, None)

        statistic, pvalue = mannwhitneyu(*groups, alternative="two-sided")
        return UnivariateResult(TEST_MANNWHITNEYU, statistic, pvalue, None)

    contingency = pd.crosstab(series, targets)
    if min(contingency.shape) < 2:
        return _skipped("contingency table has an empty row or column")

    chi2_statistic, chi2_pvalue, _, expected = chi2_contingency(contingency)
    low_expected = (expected < LOW_EXPECTED_COUNT).mean()

    # Fisher is exact where chi-square's approximation breaks down, but SciPy
    # only implements the 2x2 case; larger tables keep chi-square rather than
    # raising "The input `table` must be of shape (2, 2)".
    if low_expected > MAX_LOW_EXPECTED_PROPORTION and contingency.shape == (2, 2):
        odds_ratio, pvalue = fisher_exact(contingency)
        return UnivariateResult(TEST_FISHER, odds_ratio, pvalue, None)

    return UnivariateResult(TEST_CHI2, chi2_statistic, chi2_pvalue, None)


def _run_survival_test(series, targets, category):
    time = targets[OUTCOME_FIELD_SURVIVAL_TIME]
    event = targets[OUTCOME_FIELD_SURVIVAL_EVENT]

    if event.sum() < MIN_EVENTS:
        return _skipped("too few observed events")

    if category == CATEGORY_CONTINUOUS:
        # Wald test on the coefficient of a Cox model with this feature as the
        # only covariate. The p-value is invariant to the feature's scale, so
        # raw radiomics values need no standardisation.
        fitted = PHReg(time, series.to_numpy(dtype=float)[:, None], status=event).fit()
        return UnivariateResult(
            TEST_COX_WALD, fitted.tvalues[0], fitted.pvalues[0], None
        )

    # k-sample log-rank: the same test that compares Kaplan-Meier curves.
    chisq, pvalue = survdiff(time, event, series)
    return UnivariateResult(TEST_LOGRANK, chisq, pvalue, None)


def apply_fdr_correction(univariate_results_df, method=FDR_METHOD):
    """Benjamini-Hochberg adjust the p-values of the features that were tested.

    ``multipletests`` propagates a single NaN across its whole output, so
    skipped features are held out and given a NaN adjusted p-value instead.
    """
    adjusted = pd.Series(np.nan, index=univariate_results_df.index, dtype=float)
    testable = univariate_results_df["pvalue"].notna()

    if testable.any():
        _, pvals_adjusted, _, _ = multipletests(
            univariate_results_df.loc[testable, "pvalue"], method=method
        )
        adjusted.loc[testable] = pvals_adjusted

    univariate_results_df["pval_adj"] = adjusted
    return univariate_results_df

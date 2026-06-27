from typing import List

import pandas
import pandas as pd

from service.machine_learning import (
    resolve_collection_clinical_definitions,
    dedupe_definitions_by_name,
    definition_ids_with_values,
    get_features_labels,
)
from quantimage2_backend_common.models import (
    ClinicalFeatureDefinition,
    ClinicalFeatureValue,
    ClinicalFeatureTypes,
    ClinicalFeatureMissingValues,
    FeatureCollection,
)
from quantimage2_backend_common.const import (
    CLINICAL_FEATURE_ID_SEPARATOR,
    MODEL_TYPES,
)
from service.feature_transformation import (
    OUTCOME_FIELD_CLASSIFICATION,
    OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME,
)


def compute_fdr(
    extraction_id,
    collection_id,
    album,
    studies,
    label_category,
    gt,
    training_patients,
    test_patients,
    user_id,
    selected_feature_ids,
    fdr_threshold,
):

    features_df, labels_df = assemble_features_for_collection(
        extraction_id,
        collection_id,
        album,
        studies,
        label_category,
        gt,
        training_patients,
        test_patients,
        user_id,
    )

    missing = set(selected_feature_ids) - set(features_df.columns)

    if missing:
        raise ValueError(f"Selected features not found in dataset: {missing}")

    features_df = features_df[selected_feature_ids]

    # BEFORE IMPUTE VALUES FOR RADIOMICS FEATURES
    # compute univariate tests

    return


def assemble_features_for_collection(
    extraction_id,
    collection_id,
    album,
    studies,
    label_category,
    gt,
    training_patients,
    test_patients,
    user_id,
):

    if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION:
        outcome_columns = [OUTCOME_FIELD_CLASSIFICATION]
    elif MODEL_TYPES(label_category.label_type) == MODEL_TYPES.SURVIVAL:
        outcome_columns = [OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT]
    else:
        raise NotImplementedError()

    features_df, labels_df_indexed = get_features_labels(
        extraction_id, collection_id, studies, gt, outcome_columns=outcome_columns
    )

    labels_df_indexed = labels_df_indexed.apply(pandas.to_numeric)

    print("features_df (radiomics) before appending metadata strategy")
    print(features_df)

    features_df = features_df.drop(columns=["PatientID"])

    radiomics_feature_metadata = {}
    for feature in features_df.columns:
        radiomics_feature_metadata[feature] = {
            "type": ClinicalFeatureTypes.NUMBER.value,
        }

    # TODO - Impute NA values for radiomics features

    all_patients = (
        training_patients + test_patients if test_patients else training_patients
    )

    clinical_features, clinical_feature_metadata = get_clinical_features(
        user_id, collection_id, all_patients, album
    )

    full_feature_metadata = radiomics_feature_metadata | clinical_feature_metadata

    if len(clinical_features) > 0 and len(features_df) > 0:
        features_df = pandas.merge(
            features_df,
            clinical_features,
            left_index=True,
            right_index=True,
            how="left",
        )
    elif len(features_df) > 0:
        features_df = features_df
    elif len(clinical_features) > 0:
        features_df = clinical_features
        # features_df["PatientID"] = features_df.index
    else:
        raise ValueError("Neither clinical nore imaging features where selected")

    return features_df, labels_df_indexed


def get_clinical_features(
    user_id: str, collection_id: str, radiomics_patient_ids: List[str], album
):
    # load all definitions
    full_clin_feature_definitions = (
        ClinicalFeatureDefinition.find_by_user_id_and_album_id(
            user_id, album["album_id"]
        )
    )

    # keep only definitions that actually have values
    ids_with_values = definition_ids_with_values(
        [d.id for d in full_clin_feature_definitions]
    )

    if collection_id:
        feature_collection = FeatureCollection.find_by_id(collection_id)

        clin_feature_definitions = resolve_collection_clinical_definitions(
            feature_collection.feature_ids,
            full_clin_feature_definitions,
            ids_with_values,
        )
    else:
        clin_feature_definitions = dedupe_definitions_by_name(
            full_clin_feature_definitions, ids_with_values
        )

    if not clin_feature_definitions:
        return pd.DataFrame()

    feature_strategies = {
        f"{f.clinical_feature_file_id}{CLINICAL_FEATURE_ID_SEPARATOR}{f.name}": ClinicalFeatureMissingValues(
            f.missing_values
        )
        for f in clin_feature_definitions
    }

    # build per-feature columns
    feature_frames = []

    feature_metadata = {}

    radiomics_index = pd.Index(radiomics_patient_ids, name="PatientID")

    for clin_feature in clin_feature_definitions:
        col_name = f"{clin_feature.clinical_feature_file_id}{CLINICAL_FEATURE_ID_SEPARATOR}{clin_feature.name}"

        values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
            [clin_feature.id]
        )

        if not values:
            continue

        df = pd.DataFrame.from_dict([v.to_dict() for v in values])

        df = df[["patient_id", "value"]]

        if df.empty:
            continue

        df = df.rename(columns={"value": col_name, "patient_id": "PatientID"})
        df = df.set_index("PatientID")

        df = df.reindex(radiomics_index)

        df[col_name] = correct_type_of_clinical_values(
            df[col_name], clin_feature.feat_type
        )

        strategy = feature_strategies.get(col_name, ClinicalFeatureMissingValues.NONE)

        df[col_name] = apply_missing_value_strategy(df[col_name], strategy)

        df = df[[col_name]]

        feature_frames.append(df)

        feature_metadata[col_name] = {"feat_type": clin_feature.feat_type}

    if not feature_frames:
        return pd.DataFrame(index=radiomics_index)

    raw_df = pd.concat(feature_frames, axis=1)

    # ensure correct index name
    raw_df.index.name = "PatientID"

    return raw_df, feature_metadata


def apply_missing_value_strategy(series, strategy):

    series = series.copy()

    if strategy == ClinicalFeatureMissingValues.MEAN:
        return series.fillna(series.mean())

    elif strategy == ClinicalFeatureMissingValues.MEDIAN:
        return series.fillna(series.median())

    elif strategy == ClinicalFeatureMissingValues.MODE:
        m = series.mode()
        return series.fillna(m.iloc[0] if not m.empty else "Unknown")

    return series


def correct_type_of_clinical_values(series: pd.Series, feat_type: str) -> pd.Series:

    series = series.copy()

    if feat_type == ClinicalFeatureTypes.CATEGORICAL.value:
        series = series.astype("string")
    elif feat_type == ClinicalFeatureTypes.NUMBER.value:
        series = pd.Series(pd.to_numeric(series, errors="coerce"))

    return series

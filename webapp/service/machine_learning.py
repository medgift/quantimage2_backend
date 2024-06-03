import os
import sys
from typing import List

import pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from modeling.classification import Classification
from modeling.survival import Survival
from modeling.utils import get_random_seed
from quantimage2_backend_common.const import (
    FEATURE_ID_SEPARATOR,
    MODEL_TYPES,
    ESTIMATOR_STEP,
)
from quantimage2_backend_common.models import (
    FeatureCollection,
    FeatureExtraction,
    ClinicalFeatureDefinition,
    ClinicalFeatureValue,
    ClinicalFeatureEncodings,
    ClinicalFeatureTypes,
    ClinicalFeatureMissingValues,
)
from quantimage2_backend_common.utils import get_training_id
from service.feature_transformation import (
    OUTCOME_FIELD_CLASSIFICATION,
    OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME,
    transform_studies_collection_features_to_df,
    transform_studies_features_to_df,
)


def get_features_labels(
    extraction_id,
    collection_id,
    studies,
    gt,
    outcome_columns=[OUTCOME_FIELD_CLASSIFICATION],
):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        print("Collection", collection)
        header, features_df = transform_studies_collection_features_to_df(
            collection, studies
        )
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

    # Get Labels DataFrame
    # TODO - Allow choosing a mode (Patient only or Patient + ROI)
    print(gt)
    labels_df = pandas.DataFrame(gt, columns=["PatientID", *outcome_columns])

    labels_df_indexed = labels_df.set_index("PatientID", drop=True)

    # TODO - Check how to best deal with this, so far we ignore unlabelled patients
    label_conditions = None
    for column in outcome_columns:
        if label_conditions is None:
            label_conditions = (labels_df[column] != "") & (labels_df[column].notnull())
        else:
            label_conditions = (
                label_conditions
                & (labels_df[column] != "")
                & (labels_df[column].notnull())
            )

    labelled_patients = list(labels_df[label_conditions]["PatientID"])

    # Filter out unlabelled patients
    features_df = features_df[features_df.PatientID.isin(labelled_patients)]
    labels_df_indexed = labels_df_indexed.filter(items=labelled_patients, axis=0)

    # Concatenate features by modality & ROI
    features_df = concatenate_modalities_rois(features_df)

    # Sort both dataframe by index to ensure same order
    labels_df_indexed.sort_index(inplace=True)
    features_df.sort_index(inplace=True)

    # TODO - This will be done in Melampus also in the future
    # Impute mean for NaNs
    features_df = features_df.fillna(features_df.mean())

    return features_df, labels_df_indexed


def concatenate_modalities_rois(features_df):
    # Concatenate features from the various modalities & ROIs (if necessary)

    # Keep PatientID
    patient_id_df = features_df["PatientID"].to_frame()
    unique_pid_df = patient_id_df.drop_duplicates(subset="PatientID")
    unique_pid_df = unique_pid_df.set_index("PatientID", drop=False)

    separator = FEATURE_ID_SEPARATOR

    to_concat = [unique_pid_df]
    # Groupe dataframes by Modality & ROI
    for group, groupDf in features_df.groupby(["Modality", "ROI"]):
        # Only keep selected modalities & ROIs
        without_modality_and_roi_df = groupDf.drop(["Modality", "ROI"], axis=1)
        without_modality_and_roi_df = without_modality_and_roi_df.set_index(
            "PatientID", drop=True
        )
        prefix = separator.join(group)
        without_modality_and_roi_df = without_modality_and_roi_df.add_prefix(
            prefix + separator
        )
        # Drop columns with only NaNs
        without_modality_and_roi_df.dropna(axis=1, inplace=True, how="all")
        to_concat.append(without_modality_and_roi_df)

    # Add back the Patient ID at the end
    concatenated_df = pandas.concat(to_concat, axis=1)

    return concatenated_df


def train_model(
    extraction_id,
    collection_id,
    album,
    studies,
    feature_selection,
    label_category,
    data_splitting_type,
    train_test_splitting_type,
    training_patients,
    test_patients,
    gt,
    user_id,
):

    if MODEL_TYPES(label_category.label_type) == MODEL_TYPES.CLASSIFICATION:
        outcome_columns = [OUTCOME_FIELD_CLASSIFICATION]
        ModelClass = Classification
        estimator_step = ESTIMATOR_STEP.CLASSIFICATION.value
        refit_metric = "auc"
    elif MODEL_TYPES(label_category.label_type) == MODEL_TYPES.SURVIVAL:
        outcome_columns = [OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT]
        ModelClass = Survival
        estimator_step = ESTIMATOR_STEP.SURVIVAL.value
        refit_metric = "c-index"
    else:
        raise NotImplementedError()

    features_df, labels_df_indexed = get_features_labels(
        extraction_id,
        collection_id,
        studies,
        gt,
        outcome_columns=outcome_columns,
    )

    # Convert to numeric values
    labels_df_indexed = labels_df_indexed.apply(pandas.to_numeric)

    # clinical features - we use the training_patients + test_patients concatenation to keep track of all patients
    # in the system - this may lead to features becoming null values if the train and test patients have more patients
    # than the clinical features
    all_patients = (
        training_patients + test_patients if test_patients else training_patients
    )

    clinical_features = get_clinical_features(
        user_id, collection_id, all_patients, album
    )

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
        features_df["PatientID"] = features_df.index
    else:
        raise ValueError("Neither clinical nore imaging features where selected")

    # Convert to numeric values
    labels_df_indexed = labels_df_indexed.apply(pandas.to_numeric)

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    training_id = get_training_id(extraction_id, collection_id)

    model = ModelClass(
        feature_extraction_id=extraction_id,
        collection_id=collection_id,
        album=album,
        feature_selection=feature_selection,
        feature_names=features_df.columns,  # TODO - This might change with feature selection
        estimator_step=estimator_step,
        label_category=label_category,
        features_df=features_df,
        labels_df=labels_df_indexed,
        data_splitting_type=data_splitting_type,
        train_test_splitting_type=train_test_splitting_type,
        training_patients=training_patients,
        test_patients=test_patients,
        random_seed=random_seed,
        refit_metric=refit_metric,
        n_jobs=int(os.environ["GRID_SEARCH_CONCURRENCY"]),
        training_id=training_id,
    )

    return model.create_model()


def get_clinical_features(
    user_id: str, collection_id: str, radiomics_patient_ids: List[str], album: str
):
    full_clin_feature_definitions = ClinicalFeatureDefinition.find_by_user_id_and_album_id(
        user_id, album["album_id"]
    )

    if collection_id:
        feature_collection = FeatureCollection.find_by_id(collection_id)

        selected_clinical_features = []
        for feature_id in feature_collection.feature_ids:
            if FEATURE_ID_SEPARATOR in feature_id:
                # In the front end - clinical features are saved with no nesting - and the FEATURE_ID_SEPARATOR is used
                # to save nesting levels from the radiomics feature - https://github.com/medgift/quantimage2-frontend/blob/34e393867c2ecd364409a4aabaac5fe42dcd4172/src/Visualisation.js#L66
                # if not present it means it's a clinical feature.
                continue
            else:
                selected_clinical_features.append(feature_id)

        clin_feature_definitions = [
            i for i in full_clin_feature_definitions if i.name in selected_clinical_features
        ]

    # If the front end passes clinical feature names that are not matched by features in the db we should raise an error as the 
    # frontend may have passed in bad data
    if len(full_clin_feature_definitions) > 0 and len(clin_feature_definitions) == 0:
        raise ValueError(f"Assumed that the following clinical featurees were selected {full_clin_feature_definitions} but got {selected_clinical_features} in the available clin features.")
    
    if len(clin_feature_definitions) == 0:
        return pandas.DataFrame()

    all_features = []

    # Here we could implement the logic to transform the clinical features [one hot encoding, normalization etc..
    print("Number of clin feature definitions", len(clin_feature_definitions))
    print("names of clin feat def", [i.name for i in clin_feature_definitions])
    print(user_id)
    for clin_feature in clin_feature_definitions:
        clin_feature_values = (
            ClinicalFeatureValue.find_by_clinical_feature_definition_ids(
                [clin_feature.id]
            )
        )
        clin_feature_df = pandas.DataFrame.from_dict(
            [i.to_dict() for i in clin_feature_values]
        )
        clin_feature_df.rename(
            columns={"value": clin_feature.name, "patient_id": "PatientID"},
            inplace=True,
        )
        clin_feature_df.set_index("PatientID", inplace=True)
        radiomics_patient_ids_df = pd.DataFrame(index=radiomics_patient_ids)
        clin_feature_df = pd.merge(
            clin_feature_df,
            radiomics_patient_ids_df,
            left_index=True,
            right_index=True,
            how="outer",
        )

        index = clin_feature_df.index
        clin_feature_df.drop(columns=["clinical_feature_definition_id"], inplace=True)

        clin_feature_encoding = ClinicalFeatureEncodings(clin_feature.encoding)
        clin_feature_type = ClinicalFeatureTypes(clin_feature.feat_type)
        clin_missing_values = ClinicalFeatureMissingValues(clin_feature.missing_values)

        missing_values_idx = (
            clin_feature_df[clin_feature.name].apply(lambda x: x is None)
            | clin_feature_df[clin_feature.name].isnull()
        )

        non_missing_values = clin_feature_df.loc[~missing_values_idx][
            [clin_feature.name]
        ]

        if (
            missing_values_idx.sum() > 0
        ):  # only apply missing values logic if there are actually missing values
            print("missing values for", clin_feature.name, clin_missing_values)
            if clin_missing_values != ClinicalFeatureMissingValues.DROP:
                if clin_missing_values == ClinicalFeatureMissingValues.NONE:
                    pass
                elif clin_missing_values == ClinicalFeatureMissingValues.MEDIAN:
                    try:
                        value = non_missing_values.astype(float).median().values[0]
                    except:
                        raise ValueError(
                            f"Tried to compute the median of {clin_feature.name} but failed"
                        )

                elif clin_missing_values == ClinicalFeatureMissingValues.MEAN:
                    try:
                        value = non_missing_values.astype(float).mean().values[0]
                    except:
                        raise ValueError(
                            f"Tried to compute the mean of {clin_feature.name} but failed"
                        )
                elif clin_missing_values == ClinicalFeatureMissingValues.MODE:
                    try:
                        value = non_missing_values.mode().values[0]
                    except:
                        raise ValueError(
                            f"Tried to compute the mode of {clin_feature.name} but failed"
                        )

                clin_feature_df.loc[missing_values_idx, clin_feature.name] = value

            else:  # If we drop the missing values we need to get rid of them before the encoding.
                clin_feature_df = clin_feature_df.loc[~missing_values_idx]

        if clin_feature_type == ClinicalFeatureTypes.CATEGORICAL:
            if clin_feature_encoding == ClinicalFeatureEncodings.ONE_HOT_ENCODING:
                enc = OneHotEncoder(handle_unknown="ignore")
                enc.fit(clin_feature_df[[clin_feature.name]])
                transformed = enc.transform(
                    clin_feature_df[[clin_feature.name]]
                ).toarray()
                clin_feature_df = pandas.DataFrame(
                    data=transformed,
                    index=index,
                    columns=enc.get_feature_names_out([clin_feature.name]),
                )
            elif clin_feature_encoding == ClinicalFeatureEncodings.ORDERED_CATEGORIES:
                ordered_categories_encoder = OrdinalEncoder()
                ordered_categories_encoder.fit(clin_feature_df[[clin_feature.name]])
                ordered_categories_order = "_".join(
                    ordered_categories_encoder.categories_[0]
                )

                transformed = ordered_categories_encoder.transform(
                    clin_feature_df[[clin_feature.name]]
                )
                clin_feature_df = pandas.DataFrame(
                    data=transformed,
                    index=index,
                    columns=[f"{clin_feature.name}_{ordered_categories_order}"],
                )
            else:
                raise ValueError(
                    f"We do not support this feature type / encoding combination yet - got {clin_feature_type} and {clin_feature_encoding}"
                )
        elif clin_feature_type == ClinicalFeatureTypes.NUMBER:
            if clin_feature_encoding == ClinicalFeatureEncodings.NORMALIZATION:
                scaler = MinMaxScaler()
                transformed = scaler.fit_transform(clin_feature_df[[clin_feature.name]])
                clin_feature_df = pandas.DataFrame(
                    data=transformed, index=index, columns=[clin_feature.name]
                )

            elif clin_feature_encoding == ClinicalFeatureEncodings.NONE:
                try:
                    clin_feature_df[[clin_feature.name]] = clin_feature_df[
                        [clin_feature.name]
                    ].astype(float)
                except ValueError as e:
                    print("Error with feature", clin_feature.name)
                    raise e
            else:
                raise ValueError(
                    f"We do not support this feature type / encoding combination yet - got {clin_feature_type} and {clin_feature_encoding}"
                )
        else:
            raise ValueError("Feature type not supported yet.")

        print(clin_feature.name, clin_feature_df.columns)
        all_features.append(clin_feature_df)

    return pandas.concat(all_features, axis=1, join="outer")


def check_if_patients_in_dataframe(features_df, patient_ids):
    """Given a features_df that has patient ids as index, check if all patient ids are in the dataframe"""
    for patient_id in patient_ids:
        if patient_id not in features_df.index:
            raise ValueError(f"Patient {patient_id} not found in the dataframe")

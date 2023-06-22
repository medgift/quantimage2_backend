import traceback
import os

import pandas
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from flask import jsonify, make_response
from modeling.classification import Classification
from modeling.survival import Survival
from modeling.utils import get_random_seed
from quantimage2_backend_common.const import FEATURE_ID_SEPARATOR, MODEL_TYPES, ESTIMATOR_STEP
from quantimage2_backend_common.models import (FeatureCollection,
                                               FeatureExtraction, 
                                               ClinicalFeatureDefinition,
                                               ClinicalFeatureValue,
                                               ClinicalFeatureEncodings,
                                               ClinicalFeatureTypes)
from quantimage2_backend_common.utils import get_training_id
from service.feature_transformation import (
    OUTCOME_FIELD_CLASSIFICATION, OUTCOME_FIELD_SURVIVAL_EVENT,
    OUTCOME_FIELD_SURVIVAL_TIME, transform_studies_collection_features_to_df,
    transform_studies_features_to_df)


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
    elif MODEL_TYPES(label_category.label_type) == MODEL_TYPES.SURVIVAL:
        outcome_columns = [OUTCOME_FIELD_SURVIVAL_TIME, OUTCOME_FIELD_SURVIVAL_EVENT]
        ModelClass = Survival
        estimator_step = ESTIMATOR_STEP.SURVIVAL.value
    else:
        raise NotImplementedError()

    features_df, labels_df_indexed = get_features_labels(
        extraction_id,
        collection_id,
        studies,
        gt,
        outcome_columns=outcome_columns,
    )

    #clinical features
    clinical_features = get_clinical_features(user_id)
    print(clinical_features.head())
    print(clinical_features.columns)

    if len(clinical_features) > 0:
        features_df = pandas.merge(features_df, clinical_features, left_index=True, right_index=True, how='left')

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
        refit_metric="auc",
        n_jobs=int(os.environ["GRID_SEARCH_CONCURRENCY"]),
        training_id=training_id,
    )

    return model.classify()


def get_clinical_features(user_id: str):
    clin_feature_definitions = ClinicalFeatureDefinition.find_by_user_id(user_id)

    all_features = []

    # Here we could implement the logic to transform the clinical features [one hot encoding, normalization etc..
    print("Number of clin feature definitions", len(clin_feature_definitions))
    print("names of clin feat def", [i.name for i in clin_feature_definitions])
    print(user_id)
    for clin_feature in clin_feature_definitions:
        clin_feature_values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids([clin_feature.id])
        clin_feature_df = pandas.DataFrame.from_dict([i.to_dict() for i in clin_feature_values])
        clin_feature_df.rename(columns={'value': clin_feature.name}, inplace=True)
        clin_feature_df.set_index('patient_id', inplace=True)
        index = clin_feature_df.index
        clin_feature_df.drop(columns=['clinical_feature_definition_id'], inplace=True)

        clin_feature_encoding = ClinicalFeatureEncodings(clin_feature.encoding)
        clin_feature_type = ClinicalFeatureTypes(clin_feature.feat_type)

        if clin_feature_encoding == ClinicalFeatureEncodings.ONE_HOT_ENCODING:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(clin_feature_df[[clin_feature.name]])
            transformed = enc.transform(clin_feature_df[[clin_feature.name]]).toarray()
            clin_feature_df = pandas.DataFrame(data=transformed, index=index, columns=enc.get_feature_names_out([clin_feature.name]))
        
        if clin_feature_encoding == ClinicalFeatureEncodings.NORMALIZATION:
            scaler = MinMaxScaler()
            transformed = scaler.fit_transform(clin_feature_df[[clin_feature.name]])
            clin_feature_df = pandas.DataFrame(data=transformed, index=index, columns=[clin_feature.name])
        
        if clin_feature_type == ClinicalFeatureTypes.Integer and not clin_feature_encoding == ClinicalFeatureEncodings.ONE_HOT_ENCODING:
            clin_feature_df[[clin_feature.name]] = clin_feature_df[[clin_feature.name]].astype(int)
        print(clin_feature.name, clin_feature_df.columns)
        all_features.append(clin_feature_df)

    return pandas.concat(all_features, axis=1)


def check_if_patients_in_dataframe(features_df, patient_ids):
    """Given a features_df that has patient ids as index, check if all patient ids are in the dataframe"""
    for patient_id in patient_ids: 
        if patient_id not in features_df.index:
            raise ValueError(f"Patient {patient_id} not found in the dataframe")
import pandas

from quantimage2_backend_common.models import FeatureExtraction, FeatureCollection, ClinicalFeatureDefinition, ClinicalFeatureValue
from service.feature_transformation import (
    transform_studies_collection_features_to_df,
    transform_studies_features_to_df,
    OUTCOME_FIELD_CLASSIFICATION,
)

from quantimage2_backend_common.const import FEATURE_ID_SEPARATOR


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


def get_clinical_features(user_id: str):
    clin_feature_definitions = ClinicalFeatureDefinition.find_by_user_id(user_id)

    all_features = []

    # Here we could implement the logic to transform the clinical features [one hot encoding, normalization etc..
    for clin_feature in clin_feature_definitions:
        clin_feature_values = ClinicalFeatureValue.find_by_clinical_feature_definition_ids([clin_feature.id])
        clin_feature_df = pandas.DataFrame.from_dict([i.to_dict() for i in clin_feature_values])
        clin_feature_df.rename(columns={'value': clin_feature.name}, inplace=True)
        clin_feature_df.set_index('patient_id', inplace=True)
        clin_feature_df.drop(columns=['clinical_feature_definition_id'], inplace=True)

        if clin_feature.name == 'Gender':
            clin_feature_df[clin_feature.name] = clin_feature_df[clin_feature.name].apply(lambda x: map_gender(x))

        all_features.append(clin_feature_df)
    
    return pandas.concat(all_features, axis=1)


def map_gender(gender: str):
    map = {
        "M": 0,
        "F": 1,
    }
    return map[gender]
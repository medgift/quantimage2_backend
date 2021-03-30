import csv
import io
import collections
import pandas
import requests

from ttictoc import tic, toc

from imaginebackend_common.kheops_utils import dicomFields, endpoints, get_token_header
from imaginebackend_common.models import FeatureValue
from imaginebackend_common.utils import read_feature_file

PATIENT_ID_FIELD = "PatientID"
MODALITY_FIELD = "Modality"
ROI_FIELD = "ROI"

OUTCOME_FIELD_CLASSIFICATION = "Outcome"

STUDY_UID_COLUMN = "study_uid"
NAME_COLUMN = "name"
VALUE_COLUMN = "value"


def get_collection_features(feature_extraction, studies, collection):
    features, names = FeatureValue.get_for_collection(collection)

    tabular_features = transform_feature_values_to_tabular(features, studies)

    return tabular_features, list(tabular_features.columns)


def get_extraction_features(feature_extraction, studies):
    features, names = FeatureValue.get_for_extraction(feature_extraction)

    tabular_features = transform_feature_values_to_tabular(features, studies)

    return tabular_features, list(tabular_features.columns)


def transform_feature_values_to_tabular(values, studies):
    df = pandas.DataFrame(values)

    # Make a map of Study UID -> Patient ID to replace in the dataframe
    study_to_patient_map = {
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0]: study[
            dicomFields.PATIENT_ID
        ][dicomFields.VALUE][0]
        for study in studies
    }

    # Pivot table to make it tabular (feature names are columns, values are rows)
    # Reset Index to keep the "ID" columns (Study, Modality, ROI) in the Dataframe
    transformed_df = df.pivot_table(
        VALUE_COLUMN,
        [STUDY_UID_COLUMN, MODALITY_FIELD.lower(), ROI_FIELD.lower()],
        NAME_COLUMN,
    ).reset_index()

    # Replace Study UIDs with Patient IDs
    transformed_df[STUDY_UID_COLUMN] = transformed_df[STUDY_UID_COLUMN].replace(
        study_to_patient_map
    )

    # Rename columns to have the old structure
    renamed_df = transformed_df.rename(
        columns={
            STUDY_UID_COLUMN: PATIENT_ID_FIELD,
            MODALITY_FIELD.lower(): MODALITY_FIELD,
            ROI_FIELD.lower(): ROI_FIELD,
        }
    )

    # Sort DataFrame
    sorted_df = renamed_df.sort_values(by=[PATIENT_ID_FIELD, MODALITY_FIELD, ROI_FIELD])

    return sorted_df


def transform_studies_collection_features_to_df(
    feature_extraction, studies, collection
):
    features_df, names = get_collection_features(
        feature_extraction, studies, collection
    )

    return names, features_df


def transform_studies_features_to_df(feature_extraction, studies):
    # Get features of all studies directly in a single request, for more efficiency
    features_df, names = get_extraction_features(feature_extraction, studies)

    return names, features_df


def assemble_csv_header(features_by_modality_and_label):
    header_entries = [PATIENT_ID_FIELD, MODALITY_FIELD, ROI_FIELD]
    feature_names = set()
    for modality, features_by_label in features_by_modality_and_label.items():
        first_features = features_by_label[list(features_by_label.keys())[0]]

        for feature_name, feature_value in first_features.items():
            if not feature_name == "patientID":
                feature_names.add(feature_name)

    sorted_feature_names = sorted(feature_names)

    header_entries += sorted_feature_names

    return header_entries


def assemble_csv_data_lines(features_by_modality_and_label, csv_header):
    always_present_fields = [PATIENT_ID_FIELD, MODALITY_FIELD, ROI_FIELD]

    data_lines = []
    for modality, features_by_label in features_by_modality_and_label.items():
        for label, features in features_by_label.items():
            data_line = []

            patientID = features["patientID"]
            data_line.append(patientID)
            data_line.append(modality)
            data_line.append(label)

            for feature_name in csv_header:
                if not feature_name in always_present_fields:
                    data_line.append(features[feature_name])

            data_lines.append(data_line)

    return data_lines


def get_csv_file_content(rows):
    mem_file = io.StringIO()
    csv_writer = csv.writer(mem_file)
    csv_writer.writerows(rows)

    return mem_file.getvalue()


def make_study_file_name(patient_id, study_date):
    return f"features_{patient_id}_{study_date}.csv"


def make_album_file_name(album_name):
    return f"features_album_{album_name.replace(' ', '-')}.zip"


def make_album_collection_file_name(album_name, collection_name):
    return f"features_album_{album_name.replace(' ', '-')}_{collection_name.replace(' ', '-')}.zip"

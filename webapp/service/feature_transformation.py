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

    # for study in studies:
    #     patient_id = study[dicomFields.PATIENT_ID][dicomFields.VALUE][0]
    #     study_uid = study[dicomFields.STUDY_UID][dicomFields.VALUE][0]
    #     study_date = study[dicomFields.STUDY_DATE][dicomFields.VALUE][0]
    #
    #     # Stop this for now, it should be unique usually
    #     # dated_patient_id = f"{patient_id}_{study_date}"
    #
    #     study_tasks = list(
    #         filter(lambda task: task.study_uid == study_uid, feature_extraction.tasks)
    #     )
    #
    #     [study_header, study_data] = transform_study_features_to_tabular(
    #         study_tasks, patient_id
    #     )
    #
    #     if not header:
    #         header = study_header
    #     data += study_data
    #
    #     i += 1
    #     print(f"Transformed {i}/{len(studies)} studies")

    return names, features_df


def transform_study_features_to_tabular(tasks, patient_id):
    feature_contents = []

    # Get the feature contents
    for task in tasks:
        feature_content = read_feature_file(task.features_path)
        feature_contents.append(feature_content)

    # Group the features by modality & region
    grouped_filtered_features = {}
    leave_out_prefix = "diagnostics_"
    for feature_content in feature_contents:
        for modality, modality_features in feature_content.items():
            if modality not in grouped_filtered_features:
                grouped_filtered_features[modality] = {}

            for label, label_features in modality_features.items():
                if label not in grouped_filtered_features[modality]:
                    grouped_filtered_features[modality][
                        label
                    ] = collections.OrderedDict({"patientID": patient_id})

                filtered_label_features = {
                    feature_name: feature_value
                    for feature_name, feature_value in label_features.items()
                    if not feature_name.startswith(leave_out_prefix)
                }

                grouped_filtered_features[modality][label].update(
                    filtered_label_features
                )

    csv_header = assemble_csv_header(grouped_filtered_features)
    csv_data = assemble_csv_data_lines(grouped_filtered_features, csv_header)

    return [csv_header, csv_data]


def separate_features_by_modality_and_roi(all_features_as_csv):
    string_mem = io.StringIO(all_features_as_csv)
    df = pandas.read_csv(string_mem)

    # Group data by modality & label in order
    # to save separate CSV files in a ZIP file
    features_by_group = {}
    for idx, row in df.iterrows():
        key = f"{row.Modality}_{row.ROI}"

        if key not in features_by_group:
            features_by_group[key] = []

        features_by_group[key].append(row.values.tolist())

    return features_by_group


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


def make_study_file_name(patient_id, study_date, families):
    return f"features_{patient_id}_{study_date}_{'-'.join(families)}.csv"


def make_album_file_name(album_name, families):
    return f"features_album_{album_name.replace(' ', '-')}_{'-'.join(families)}.zip"


def make_album_collection_file_name(album_name, collection_name):
    return f"features_album_{album_name.replace(' ', '-')}_{collection_name.replace(' ', '-')}.zip"

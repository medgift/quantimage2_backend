import csv
import io
import collections
import pandas

from imaginebackend_common.kheops_utils import dicomFields
from imaginebackend_common.utils import read_feature_file

PATIENT_ID_FIELD = "PatientID"
MODALITY_FIELD = "Modality"
ROI_FIELD = "ROI"


def transform_studies_features_to_csv(feature_extraction, studies):

    header = None
    data = []

    for study in studies:
        print(study)
        patient_id = study[dicomFields.PATIENT_ID][dicomFields.VALUE][0]
        study_uid = study[dicomFields.STUDY_UID][dicomFields.VALUE][0]
        study_date = study[dicomFields.STUDY_DATE][dicomFields.VALUE][0]

        dated_patient_id = f"{patient_id}_{study_date}"

        study_tasks = list(
            filter(lambda task: task.study_uid == study_uid, feature_extraction.tasks)
        )

        [study_header, study_data] = transform_study_features_to_tabular(
            study_tasks, dated_patient_id
        )

        if not header:
            header = study_header
        data += study_data

    print(data)

    return [header, data]


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
    return f"features_album_{album_name}_{'-'.join(families)}.zip"

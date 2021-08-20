import csv
import io
import collections
import pandas
import requests

from ttictoc import tic, toc

from imaginebackend_common.kheops_utils import dicomFields, endpoints, get_token_header
from imaginebackend_common.models import (
    FeatureValue,
    db,
    FeatureExtractionTask,
    FeatureCollection,
    feature_collection_value,
)
from imaginebackend_common.utils import read_feature_file

PATIENT_ID_FIELD = "PatientID"
MODALITY_FIELD = "Modality"
ROI_FIELD = "ROI"

OUTCOME_FIELD_CLASSIFICATION = "Outcome"
OUTCOME_FIELD_SURVIVAL_EVENT = "Event"
OUTCOME_FIELD_SURVIVAL_TIME = "Time"

STUDY_UID_COLUMN = "study_uid"
NAME_COLUMN = "name"
VALUE_COLUMN = "value"


def get_collection_features(studies, collection):
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


def transform_studies_collection_features_to_df(studies, collection):
    features_df, names = get_collection_features(studies, collection)

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


def get_data_points_collection(studies, collection_id):
    study_uid_patient_id_map = {
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0]: study[
            dicomFields.PATIENT_ID
        ][dicomFields.VALUE][0]
        for study in studies
    }

    # Low-level DBAPI fetchall()
    compiled = (
        db.select([FeatureExtractionTask.__table__.c.study_uid])
        .select_from(
            FeatureExtractionTask.__table__.join(
                FeatureValue.__table__,
                FeatureValue.__table__.c.feature_extraction_task_id
                == FeatureExtractionTask.__table__.c.id,
            )
            .join(
                feature_collection_value,
                feature_collection_value.c.feature_value_id
                == FeatureValue.__table__.c.id,
            )
            .join(
                FeatureCollection.__table__,
                FeatureCollection.__table__.c.id
                == feature_collection_value.c.feature_collection_id,
            )
        )
        .distinct()
        .where(FeatureCollection.__table__.c.id == collection_id)
        .compile(dialect=db.engine.dialect, compile_kwargs={"literal_binds": True})
    )

    sql = str(compiled)

    conn = db.engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(sql)

    study_uids = []
    for row in cursor.fetchall():
        # ensure that we fully fetch!
        if row[0] in study_uid_patient_id_map:
            study_uids.append(row[0])

    patient_ids = [study_uid_patient_id_map[study_uid] for study_uid in study_uids]

    conn.close()

    return patient_ids


def get_data_points_extraction(result, studies):
    # Filter out studies that weren't processed successfully
    successful_studies = [
        study
        for study in studies
        if study[dicomFields.STUDY_UID][dicomFields.VALUE][0] not in result.errors
    ]

    # Get Patient IDs from studies
    patient_ids = []
    for study in successful_studies:
        patient_id = study[dicomFields.PATIENT_ID][dicomFields.VALUE][0]
        if not patient_id in patient_ids:
            patient_ids.append(patient_id)

    return patient_ids

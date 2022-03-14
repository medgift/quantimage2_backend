import pandas

from imaginebackend_common.models import FeatureExtraction, FeatureCollection
from modeling.classification import Classification
from service.feature_transformation import (
    transform_studies_features_to_df,
    transform_studies_collection_features_to_df,
)


def train_classification_model(
    extraction_id,
    collection_id,
    studies,
    data_splitting_type,
    training_patients,
    test_patients,
    gt,
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
    labels_df = pandas.DataFrame(gt, columns=["PatientID", "Label"])

    labels_df_indexed = labels_df.set_index("PatientID", drop=True)

    # TODO - Check how to best deal with this, so far we ignore unlabelled patients
    labelled_patients = list(labels_df[labels_df["Label"] != ""]["PatientID"])

    # Filter out unlabelled patients
    features_df = features_df[features_df.PatientID.isin(labelled_patients)]

    # Concatenate features by modality & ROI
    features_df = concatenate_modalities_rois(features_df)

    # Sort both dataframe by index to ensure same order
    labels_df_indexed.sort_index(inplace=True)
    features_df.sort_index(inplace=True)

    # TODO - This will be done in Melampus also in the future
    # Impute mean for NaNs
    features_df = features_df.fillna(features_df.mean())

    training_validation = None
    test_validation = None
    test_validation_params = None

    random_seed = get_random_seed(
        extraction_id=extraction_id, collection_id=collection_id
    )

    classifier = Classification(
        features_df,
        labels_df_indexed,
        data_splitting_type,
        training_patients,
        test_patients,
        random_seed,
    )

    # Run modeling pipeline depending on the type of validation (full CV, train/test)
    (
        model,
        training_validation,
        training_validation_params,
        test_validation,
        test_validation_params,
        metrics,
    ) = classifier.classify()

    return (
        model,
        training_validation,
        training_validation_params,
        test_validation,
        test_validation_params,
        metrics,
    )


def concatenate_modalities_rois(features_df, keep_identifiers=False):
    # Concatenate features from the various modalities & ROIs (if necessary)

    # Keep PatientID
    patient_id_df = features_df["PatientID"].to_frame()
    unique_pid_df = patient_id_df.drop_duplicates(subset="PatientID")
    unique_pid_df = unique_pid_df.set_index("PatientID", drop=False)

    to_concat = [unique_pid_df]
    # Groupe dataframes by Modality & ROI
    for group, groupDf in features_df.groupby(["Modality", "ROI"]):
        # Only keep selected modalities & ROIs
        without_modality_and_roi_df = groupDf.drop(["Modality", "ROI"], axis=1)
        without_modality_and_roi_df = without_modality_and_roi_df.set_index(
            "PatientID", drop=True
        )
        prefix = "-".join(group)
        without_modality_and_roi_df = without_modality_and_roi_df.add_prefix(
            prefix + "-"
        )
        # Drop columns with NaNs (should not exist anyway)
        without_modality_and_roi_df.dropna(axis=1, inplace=True)
        to_concat.append(without_modality_and_roi_df)

    # Add back the Patient ID at the end
    concatenated_df = pandas.concat(to_concat, axis=1)

    return concatenated_df


def get_random_seed(extraction_id=None, collection_id=None):
    return 100000 + collection_id if collection_id else extraction_id

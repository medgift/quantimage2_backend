import csv
import io
import math
import os
import random

import pandas
import tempfile
import numpy as np
from sklearn.model_selection import train_test_split

from imaginebackend_common.const import VALIDATION_TYPES
from imaginebackend_common.models import FeatureExtraction, FeatureCollection
from service.feature_transformation import (
    transform_studies_features_to_df,
    transform_studies_collection_features_to_df,
)

from melampus.classifier import MelampusClassifier


def train_classification_model(
    extraction_id,
    collection_id,
    studies,
    algorithm_type,
    data_normalization,
    validation_type,
    training_size,
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

    # TODO - Check how to best deal with this, so far we ignore unlabelled patients
    labelled_patients = list(labels_df[labels_df["Label"] != ""]["PatientID"])

    # Filter out unlabelled patients
    features_df = features_df[features_df.PatientID.isin(labelled_patients)]

    # Concatenate features by modality & ROI
    features_df = concatenate_modalities_rois(features_df)

    # TODO - This will be done in Melampus also in the future
    # Impute mean for NaNs
    features_df = features_df.fillna(features_df.mean())

    # Run classification pipeline depending on the type of validation (full CV, train/test)
    if VALIDATION_TYPES(validation_type) == VALIDATION_TYPES.CROSSVALIDATION:
        # Get labels for each patient (to make sure they are in the same order)
        labels_list = []
        for index, row in features_df.iterrows():
            patient_label = labels_df[
                labels_df["PatientID"] == row.PatientID
            ].Label.values[0]
            labels_list.append(patient_label)

        # Create temporary file for CSV content & dump it there
        with tempfile.NamedTemporaryFile(mode="w+") as temp:

            # Save filtered DataFrame to CSV file (to feed it to Melampus)
            features_df.to_csv(temp.name, index=False)

            (
                model,
                cv_strategy,
                cv_params,
                metrics,
                training_patient_ids,
                test_patient_ids,
            ) = classification_full_dataset(
                temp.name,
                labels_list,
                data_normalization,
                algorithm_type,
                extraction_id,
            )
    else:
        (
            model,
            cv_strategy,
            cv_params,
            metrics,
            training_patient_ids,
            test_patient_ids,
        ) = classification_train_test(
            features_df,
            labels_df,
            training_size / 100,  # Transform to percentage
            data_normalization,
            algorithm_type,
            extraction_id,
        )

    return (
        model,
        cv_strategy,
        cv_params,
        training_patient_ids,
        test_patient_ids,
        metrics,
    )


def classification_train_test(
    features, labels, training_size, normalization, algorithm, extraction_id
):
    features_clean = features.drop("PatientID", axis=1)
    labels_clean = labels.set_index("PatientID", drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        features_clean,
        labels_clean,
        train_size=training_size,
        random_state=extraction_id,
    )

    return "TODO"  # TODO - Implement train/test cross-evaluation & metrics


def classification_full_dataset(
    filename, labels_list, data_normalization, algorithm_type, extraction_id
):
    # Call Melampus classifier with the right parameters
    standardization = True if data_normalization == "standardization" else False
    l2norm = True if data_normalization == "l2norm" else False
    my_classifier = MelampusClassifier(
        filename,
        None,
        None,
        labels_list,
        algorithm_name=algorithm_type,
        scaling=standardization,
        normalize=l2norm,
    )

    model, cv_strategy, cv_params = my_classifier.train_and_evaluate(
        random_state=extraction_id
    )

    # TODO - Removed the confusion matrix for now
    # myClassifier.metrics["specificity"] = myClassifier.metrics["true_neg"] / (
    #     myClassifier.metrics["true_neg"] + myClassifier.metrics["false_pos"]
    # )

    # Metrics of the model
    metrics = my_classifier.metrics

    return (
        model,
        cv_strategy,
        cv_params,
        metrics,
        list(
            my_classifier.ids.PatientID
        ),  # TODO - Support other types of IDs, not only Patient ID
        None,
    )


def test_classification_model(model, test_features_df):
    print("Aha, let's look at this!")


def concatenate_modalities_rois(featuresDf, keep_identifiers=False):
    # Concatenate features from the various modalities & ROIs (if necessary)

    # Keep PatientID
    pidDf = featuresDf["PatientID"].to_frame()
    uniquePidDf = pidDf.drop_duplicates(subset="PatientID")
    uniquePidDf = uniquePidDf.set_index("PatientID", drop=False)

    to_concat = [uniquePidDf]
    # Groupe dataframes by Modality & ROI
    for group, groupDf in featuresDf.groupby(["Modality", "ROI"]):
        # Only keep selected modalities & ROIs
        withoutModalityAndROIDf = groupDf.drop(["Modality", "ROI"], axis=1)
        withoutModalityAndROIDf = withoutModalityAndROIDf.set_index(
            "PatientID", drop=True
        )
        prefix = "-".join(group)
        withoutModalityAndROIDf = withoutModalityAndROIDf.add_prefix(prefix + "-")
        # Drop columns with NaNs (should not exist anyway)
        withoutModalityAndROIDf.dropna(axis=1, inplace=True)
        to_concat.append(withoutModalityAndROIDf)

    # Add back the Patient ID at the end
    concatenatedDf = pandas.concat(to_concat, axis=1)

    return concatenatedDf

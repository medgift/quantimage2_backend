import csv
import io
import os
import pandas
import tempfile
import numpy as np

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
    gt,
):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        header, features_df = transform_studies_collection_features_to_df(
            studies, collection
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

    # TODO - Analyze what is the best thing to do, try concatenation so far
    features_df = concatenate_modalities_rois(features_df)

    # TODO - This will be done in Melampus also in the future
    # Impute mean for NaNs
    features_df = features_df.fillna(features_df.mean())

    # Get labels for each patient (to make sure they are in the same order)
    labelsList = []
    for index, row in features_df.iterrows():
        patient_label = labels_df[labels_df["PatientID"] == row.PatientID].Label.values[
            0
        ]
        labelsList.append(int(patient_label))

    # Create temporary file for CSV content & dump it there
    with tempfile.NamedTemporaryFile(mode="w+") as temp:

        # Save filtered DataFrame to CSV file (to feed it to Melampus)
        features_df.to_csv(temp.name, index=False)

        # Call Melampus classifier with the right parameters
        standardization = True if data_normalization == "standardization" else False
        l2norm = True if data_normalization == "l2norm" else False
        myClassifier = MelampusClassifier(
            temp.name,
            None,
            None,
            labelsList,
            algorithm_name=algorithm_type,
            scaling=standardization,
            normalize=l2norm,
        )

        model, cv_strategy, cv_params = myClassifier.train_and_evaluate(
            random_state=extraction_id
        )

        # TODO - Removed the confusion matrix for now
        # myClassifier.metrics["specificity"] = myClassifier.metrics["true_neg"] / (
        #     myClassifier.metrics["true_neg"] + myClassifier.metrics["false_pos"]
        # )

        # Save metrics in the model for now
        model.metrics = myClassifier.metrics

        # train with metrics (doesn't work withe the small dataset)
        # myClassifier.train_and_evaluate()

        # Convert metrics to native Python types
        # metrics = {}
        # for metric_name, metric_value in myClassifier.metrics.items():
        #     metrics[metric_name] = metric_value.item()

        # os.unlink(temp.name)

    return model, cv_strategy, cv_params, labelled_patients


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

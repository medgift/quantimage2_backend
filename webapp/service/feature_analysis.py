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


def train_model_with_metric(
    extraction_id, collection_id, studies, algorithm_type, modalities, rois, gt
):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        header, features_df = transform_studies_collection_features_to_df(
            extraction, studies, collection
        )
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

    # # Transform array to CSV in order to create a DataFrame
    # mem_file = io.StringIO()
    # csv_writer = csv.writer(mem_file)
    # csv_writer.writerow(header)
    # csv_writer.writerows(features)
    # mem_file.seek(0)
    #
    # # Create DataFrame from CSV data
    # featuresDf = pandas.read_csv(mem_file, dtype={"PatientID": np.str})

    # TODO - How to deal with multiple modalities & ROIs? - Concatenate for now...
    # Grab just CT & GTV_L as a test
    # featuresDf = featuresDf[
    #     (featuresDf["Modality"] == "CT") & (featuresDf["ROI"] == "GTV_L")
    # ]

    # Keep just GTV_L for now
    # featuresDf = featuresDf[(featuresDf["ROI"] == "GTV_L")]

    # Drop the ROI column
    # featuresDf = featuresDf.drop(["ROI"], axis=1)

    # TODO - Analyze what is the best thing to do, try concatenation so far
    features_df = concatenate_modalities_rois(features_df, modalities, rois)

    # TODO - Should Melampus be able to deal with string columns?
    # Drop modality & ROI from the dataframe, as Melampus doesn't support string values
    # featuresDf = featuresDf.drop(["Modality", "ROI"], axis=1)

    # Get Labels DataFrame
    # TODO - Allow choosing a mode (Patient only or Patient + ROI)
    labels_df = pandas.DataFrame(gt, columns=["PatientID", "Label"])

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

        # Call Melampus classifier
        myClassifier = MelampusClassifier(temp.name, algorithm_type, labelsList)

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

    return model, cv_strategy, cv_params


def concatenate_modalities_rois(featuresDf, modalities, rois):
    # Concatenate features from the various modalities & ROIs (if necessary)

    # Keep PatientID
    pidDf = featuresDf["PatientID"].to_frame()
    uniquePidDf = pidDf.drop_duplicates(subset="PatientID")
    uniquePidDf = uniquePidDf.set_index("PatientID", drop=False)

    to_concat = [uniquePidDf]
    # Groupe dataframes by Modality & ROI
    for group, groupDf in featuresDf.groupby(["Modality", "ROI"]):
        # Only keep selected modalities & ROIs
        if (len(modalities) == 0 and len(rois) == 0) or (
            group[0] in modalities and group[1] in rois
        ):
            withoutModalityAndROIDf = groupDf.drop(["Modality", "ROI"], axis=1)
            withoutModalityAndROIDf = withoutModalityAndROIDf.set_index(
                "PatientID", drop=True
            )
            prefix = "-".join(group)
            withoutModalityAndROIDf = withoutModalityAndROIDf.add_prefix(prefix + "_")
            to_concat.append(withoutModalityAndROIDf)

    # Add back the Patient ID at the end
    concatenatedDf = pandas.concat(to_concat, axis=1)

    return concatenatedDf

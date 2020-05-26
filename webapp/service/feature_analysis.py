import csv
import io
import pandas
import tempfile

from imaginebackend_common.models import FeatureExtraction
from service.feature_transformation import transform_studies_features_to_csv

from melampus import classifier


def train_model_with_metric(extraction_id, studies, album, gt):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    [header, features] = transform_studies_features_to_csv(extraction, studies)

    # Transform array to CSV in order to create a DataFrame
    mem_file = io.StringIO()
    csv_writer = csv.writer(mem_file)
    csv_writer.writerow(header)
    csv_writer.writerows(features)
    mem_file.seek(0)

    # Create DataFrame from CSV data
    featuresDf = pandas.read_csv(mem_file)

    # TODO - How to deal with multiple modalities & ROIs?
    # Grab just CT & GTV_T as a test
    featuresDf = featuresDf[
        (featuresDf["Modality"] == "CT") & (featuresDf["ROI"] == "GTV_L")
    ]

    # TODO - Should Melampus be able to deal with string columns?
    # Drop modality & ROI from the dataframe, as Melampus doesn't support string values
    featuresDf = featuresDf.drop(["Modality", "ROI"], axis=1)

    # TODO - Get actual labels dynamically (CSV file or something)
    labelsDf = gt

    labelsList = []
    for index, row in featuresDf.iterrows():
        patient_label = labelsDf[
            labelsDf["PatientID"] == row.PatientID[0 : row.PatientID.rindex("_")]
        ].Label.values[0]
        labelsList.append(patient_label)

    # trainLabels = []
    # for index, row in trainDf.iterrows():
    #     patient_label = labelsDf[
    #         labelsDf["PatientID"] == row.PatientID[0 : row.PatientID.rindex("_")]
    #     ].Label.values[0]
    #     trainLabels.append(patient_label)
    #
    # testLabels = []
    # for index, row in testDf.iterrows():
    #     patient_label = labelsDf[
    #         labelsDf["PatientID"] == row.PatientID[0 : row.PatientID.rindex("_")]
    #     ].Label.values[0]
    #     testLabels.append(patient_label)

    # Create temporary file for CSV content & dump it there
    with tempfile.NamedTemporaryFile(mode="w+") as temp:

        # Save filtered DataFrame to CSV file (to feed it to Melampus)
        featuresDf.to_csv(temp.name)

        # Call Melampus classifier
        myClassifier = classifier.MelampusClassifier(
            temp.name, "logistic_regression", labelsList
        )

        myClassifier.train()

        # Convert metrics to native Python types
        metrics = {}
        for metric_name, metric_value in myClassifier.metrics.items():
            metrics[metric_name] = metric_value.item()

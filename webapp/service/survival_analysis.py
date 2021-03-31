import csv
import io
import tempfile

import pandas
import numpy as np
from melampus.classifier import MelampusClassifier
from melampus.survival_analysis import MelampusSurvivalAnalyzer
from melampus.feature_selector import MelampusFeatureSelector

from imaginebackend_common.models import FeatureExtraction
from service.feature_analysis import concatenate_modalities_rois
from service.feature_transformation import transform_studies_features_to_df


def train_survival_model(extraction_id, collection_id, studies, gt):
    extraction = FeatureExtraction.find_by_id(extraction_id)

    if collection_id:
        collection = FeatureCollection.find_by_id(collection_id)
        header, features_df = transform_studies_collection_features_to_df(
            extraction, studies, collection
        )
    else:
        header, features_df = transform_studies_features_to_df(extraction, studies)

    # Transform array to CSV in order to create a DataFrame
    mem_file = io.StringIO()
    csv_writer = csv.writer(mem_file)
    csv_writer.writerow(header)
    csv_writer.writerows(features)
    mem_file.seek(0)

    # Create DataFrame from CSV data
    featuresDf = pandas.read_csv(mem_file, dtype={"PatientID": np.str})

    # Index the dataframe by patient ID
    # Keep only numeric columns in one dataframe & categorical in another
    featuresDfIndexed = featuresDf.set_index("PatientID", drop=False)
    featuresDfNumericOnly = featuresDfIndexed.drop(
        ["PatientID", "Modality", "ROI"], axis=1
    )

    featuresDfCategoricalOnly = featuresDfIndexed[["PatientID", "Modality", "ROI"]]

    # Remove highly correlated features as they hurt the convergence of Cox models

    featureSelector = MelampusFeatureSelector(dataframe=featuresDfNumericOnly)

    filteredFeatures, p_value = featureSelector.variance_threshold()

    filteredFeaturesWithCategorical = pandas.concat(
        [featuresDfCategoricalOnly, filteredFeatures], axis=1
    )

    # TODO - Analyze what is the best thing to do, try concatenation so far
    featuresDf = concatenate_modalities_rois(filteredFeaturesWithCategorical)

    # Index the features Dataframe (for combining with the labels)
    featuresDf.set_index("PatientID", drop=False, inplace=True)

    # Get Labels DataFrame
    # TODO - Allow choosing a mode (Patient only or Patient + ROI)
    labelsDf = pandas.DataFrame(gt, columns=["PatientID", "Time", "Event"])

    # Index the labels Dataframe (for combining with the features)
    labelsDf.set_index("PatientID", drop=True, inplace=True)

    # Combine labels with features
    combinedDf = pandas.concat(
        [featuresDf, labelsDf.set_index(featuresDf.index)], axis=1
    )

    # Create temporary file for CSV content & dump it there
    with tempfile.NamedTemporaryFile(mode="w+") as temp:
        # Index the DataFrame by PatientID (it should not be a column when training the survival model)
        combinedDf.set_index("PatientID", drop=True, inplace=True)

        # Save combined DataFrame to CSV file (to feed it to Melampus)
        combinedDf.to_csv(temp.name, index=False)

        # Call Melampus survival analysis
        mySurvivalAnalyzer = MelampusSurvivalAnalyzer(temp.name, "Time", "Event")

        analyzer = mySurvivalAnalyzer.analyzer

        fitted_analyzer, concordance_index = mySurvivalAnalyzer.train()

        # train with metrics (doesn't work withe the small dataset)
        # myClassifier.train_and_evaluate()

        # Convert metrics to native Python types
        # metrics = {}
        # for metric_name, metric_value in myClassifier.metrics.items():
        #     metrics[metric_name] = metric_value.item()

        # os.unlink(temp.name)

    return (
        fitted_analyzer,
        f"variance_threshold (p={p_value})",
        list(filteredFeatures.columns),
    )

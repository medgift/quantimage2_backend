import os
import traceback
import csv
import json
import io
import base64

import pandas as pd

from pathvalidate import sanitize_filename
from sqlalchemy.orm import joinedload
from flask import Blueprint, jsonify, request, g, make_response, Response
from quantimage2_backend_common.models import Model, LabelCategory, Album, Label
from quantimage2_backend_common.utils import get_training_id, format_model
from routes.utils import validate_decorate
from service.feature_extraction import get_album_details
from service.machine_learning import train_model, model_compare_permuation_test

# Note: Plotly imports kept for potential future use
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define blueprint
bp = Blueprint(__name__, "models")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/models/<album_id>", methods=("GET", "POST"))
def models_by_album(album_id):

    if request.method == "GET":
        models = Model.find_by_album(album_id, g.user)
        formatted_models = list(map(lambda model: format_model(model), models))
        return jsonify(formatted_models)

    if request.method == "POST":
        # Dictionary with the key corresponding
        # to a MODALITY + ROI combination and
        # the value being a JSON representation
        # of the features (including patient ID)
        body = request.json

        label_category = LabelCategory.find_by_id(body["label-category-id"])
        user_id = g.user

        feature_extraction_id = body["extraction-id"]
        collection_id = body["collection-id"]
        studies = body["studies"]
        album = body["album"]
        gt = body["labels"]
        data_splitting_type = body["data-splitting-type"]
        train_test_split_type = body["train-test-split-type"]
        training_patients = body["training-patients"]
        test_patients = body["test-patients"]
        feature_selection = None

        try:
            n_steps = train_model(
                feature_extraction_id,
                collection_id,
                album,
                studies,
                feature_selection,
                label_category,
                data_splitting_type,
                train_test_split_type,
                training_patients,
                test_patients,
                gt,
                g.user,
            )

            training_id = get_training_id(feature_extraction_id, collection_id)

            return jsonify({"training-id": training_id, "n-steps": n_steps})

        except Exception as e:
            traceback.print_exc()
            error_message = {"error": str(e)}
            return make_response(jsonify(error_message), 500)


@bp.route("/models/<id>", methods=["DELETE"])
def model(id):
    the_model = Model.delete_by_id(
        id,
        options=(
            joinedload(Model.feature_extraction),
            joinedload(Model.label_category),
        ),
    )
    formatted_model = format_model(the_model)
    model_path = the_model.model_path
    try:
        os.unlink(model_path)
    except FileNotFoundError as e:
        pass
    return jsonify(formatted_model)


@bp.route("/models")
def models_by_user():
    albums = Model.find_by_user(g.user)
    return jsonify(albums)


@bp.route("/models/compare", methods=["POST"])
def compare_models():
    
    print("file in the models api")
    model_ids = json.loads(request.data)["model_ids"]
    models = [Model.find_by_id(i) for i in model_ids]
    model_comparison_df = model_compare_permuation_test(models)
    model_ids_string = [str(i) for i in model_ids]
    csv_filename = f"model_comparisons_{'_'.join(model_ids_string)}.csv"
    # Create a string buffer
    output = io.StringIO()
    
    model_comparison_df.insert(0, "model/model", model_comparison_df.index)
    # Write the DataFrame to the string buffer as CSV
    model_comparison_df.to_csv(output, index=False)

    # Get the CSV content as a string
    csv_content_string = output.getvalue()

    # Close StringIO to free up resources
    output.close()

    print("file content")
    print(csv_content_string)

    return Response(
        csv_content_string,
        mimetype="text/csv",
        headers={
            "Content-disposition": f"attachment; filename={csv_filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

@bp.route("/models/<id>/plot-test-predictions", methods=["GET", "POST"])
def plot_test_predictions(id):
    if request.method == "POST":
        # Get additional model ID from request body
        print("Request JSON:", request.json)  # Debug print
        model_ids_data = request.json.get('model_ids', [])
        print("Model IDs data:", model_ids_data)  # Debug print

        # Handle both string and array formats
        if isinstance(model_ids_data, str):
            model_ids = [x.strip() for x in model_ids_data.split(',') if x.strip()]
        elif isinstance(model_ids_data, list):
            model_ids = [str(x).strip() for x in model_ids_data if str(x).strip()]
        else:
            model_ids = []
    else:
        # Handle GET request - split the URL parameter if it contains commas
        model_ids = [x.strip() for x in id.split(',') if x.strip()]

    # Convert model IDs to integers
    model_ids = [int(mid) for mid in model_ids]
    print("Final model_ids:", model_ids)  # Debug print

    # Prepare data structure for frontend Plotly components
    models_data = []
    
    for model_id in model_ids:
        model = Model.find_by_id(model_id)
        if not model:
            return jsonify({"error": f"Model {model_id} not found"}), 404
            
        # Get the test predictions values
        test_predictions = model.test_predictions
        test_probabilities = model.test_predictions_probabilities

        # Create dictionary to store ground truth labels
        ground_truth = {}
        labels = Label.find_by_label_category(model.label_category_id)
        for label in labels:
            label_value = next(iter(label.label_content.values()))
            ground_truth[label.patient_id] = int(label_value)

        # Prepare patient data for this model
        patients_data = []
        for patient_id in test_predictions.keys():
            pred = test_predictions[patient_id]["prediction"]
            prob = test_probabilities[patient_id]["probabilities"][1]
            gt = ground_truth.get(patient_id, None)

            if gt is not None:
                patients_data.append({
                    "patient_id": patient_id,
                    "probability": prob,
                    "prediction": pred,
                    "ground_truth": gt
                })

        # Get metrics from the model
        test_metrics = model.test_metrics
        auc_value = test_metrics.get('auc', {}).get('mean', 0) if test_metrics else 0

        # Add model data to response
        models_data.append({
            "model_id": model_id,
            "model_name": f"Model {model_id}",
            "patients": patients_data,
            "auc": auc_value,
            "metrics": test_metrics
        })

    return jsonify(models_data)
    
@bp.route("/models/<id>/plot-train-predictions", methods=["GET", "POST"])
def plot_train_predictions(id):
    if request.method == "POST":
        # Get additional model ID from request body
        print("Request JSON:", request.json)  # Debug print
        model_ids_data = request.json.get('model_ids', [])
        print("Model IDs data:", model_ids_data)  # Debug print

        # Handle both string and array formats
        if isinstance(model_ids_data, str):
            model_ids = [x.strip() for x in model_ids_data.split(',') if x.strip()]
        elif isinstance(model_ids_data, list):
            model_ids = [str(x).strip() for x in model_ids_data if str(x).strip()]
        else:
            model_ids = []
    else:
        # Handle GET request - split the URL parameter if it contains commas
        model_ids = [x.strip() for x in id.split(',') if x.strip()]

    # Convert model IDs to integers
    model_ids = [int(mid) for mid in model_ids]
    print("Final model_ids:", model_ids)  # Debug print

    # Prepare data structure for frontend Plotly components
    models_data = []
    
    for model_id in model_ids:
        model = Model.find_by_id(model_id)
        if not model:
            return jsonify({"error": f"Model {model_id} not found"}), 404
            
        # Get the train predictions values
        train_predictions = model.train_predictions
        train_probabilities = model.train_predictions_probabilities

        # Create dictionary to store ground truth labels
        ground_truth = {}
        labels = Label.find_by_label_category(model.label_category_id)
        for label in labels:
            label_value = next(iter(label.label_content.values()))
            ground_truth[label.patient_id] = int(label_value)

        # Prepare patient data for this model
        patients_data = []
        for patient_id in train_predictions.keys():
            pred = train_predictions[patient_id]["prediction"]
            prob = train_probabilities[patient_id]["probabilities"][1]
            gt = ground_truth.get(patient_id, None)

            if gt is not None:
                patients_data.append({
                    "patient_id": patient_id,
                    "probability": prob,
                    "prediction": pred,
                    "ground_truth": gt
                })

        # Get metrics from the model
        training_metrics = model.training_metrics
        auc_value = training_metrics.get('auc', {}).get('mean', 0) if training_metrics else 0

        # Add model data to response
        models_data.append({
            "model_id": model_id,
            "model_name": f"Model {model_id}",
            "patients": patients_data,
            "auc": auc_value,
            "metrics": training_metrics
        })

    return jsonify(models_data)

@bp.route("/models/<id>/download-test-bootstrap-values")
def download_test_bootstrap_values(id):
    # Get the model
    model = Model.find_by_id(id)

    if not model:
        return jsonify({"error": "Model not found"}), 404

    # Get the test bootstrap values
    test_bootstrap_values = model.test_bootstrap_values

    # Convert JSON content to a list suitable for CSV file
    csv_content = []

    # Add header row (based on the number of bootstrap repetitions)
    n_bootstrap = len(list(test_bootstrap_values.values())[0])
    header_row = ["Metric", *[f"Repetition {n + 1}" for n in range(n_bootstrap)]]
    csv_content.append(header_row)

    # Iterate over the metrics and their bootstrapped means
    for metric_index, (metric, values) in enumerate(
        zip(test_bootstrap_values.keys(), test_bootstrap_values.values())
    ):
        row = [metric, *values]
        csv_content.append(row)

    # CSV file name
    album_id = model.feature_extraction.album_id
    album_details = get_album_details(album_id, g.token)
    model_suffix = album_details["name"]
    if model.feature_collection is not None:
        model_suffix = f"{model_suffix}_{model.feature_collection.name}"

    csv_filename = f"test_bootstrap_{model_suffix}_{model.best_algorithm}_{model.best_data_normalization}_{id}.csv"
    csv_filename = sanitize_filename(csv_filename, "_").replace(" ", "_")

    # Writing data into CSV format as String
    output = io.StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerows(csv_content)
    csv_content_string = output.getvalue()

    # Close StringIO to free up resources
    output.close()

    # Return the CSV content as a response
    return Response(
        csv_content_string,
        mimetype="text/csv",
        headers={
            "Content-disposition": f"attachment; filename={csv_filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

@bp.route("/models/<id>/download-test-scores-values")
def download_test_scores_values(id):
    # Get the model
    model = Model.find_by_id(id)

    if not model:
        return jsonify({"error": "Model not found"}), 404

    # Get the test scores values
    test_scores_values = model.test_scores_values

    # Add header row (based on the number of bootstrap repetitions)
    header_row = ["Metric", *[f"Repetition {n + 1}" for n in range(len(test_scores_values))]]
    
    
    df = pd.DataFrame.from_dict(test_scores_values).T.reset_index()
    df.columns = header_row

    # CSV file name
    album_id = model.feature_extraction.album_id
    album_details = get_album_details(album_id, g.token)
    model_suffix = album_details["name"]
    if model.feature_collection is not None:
        model_suffix = f"{model_suffix}_{model.feature_collection.name}"

    csv_filename = f"test_scores_{model_suffix}_{model.best_algorithm}_{model.best_data_normalization}_{id}.csv"
    csv_filename = sanitize_filename(csv_filename, "_").replace(" ", "_")

    # Create a string buffer
    output = io.StringIO()

    # Write the DataFrame to the string buffer as CSV
    df.to_csv(output, index=False)

    # Get the CSV content as a string
    csv_content_string = output.getvalue()

    # Close StringIO to free up resources
    output.close()

    # Return the CSV content as a response
    return Response(
        csv_content_string,
        mimetype="text/csv",
        headers={
            "Content-disposition": f"attachment; filename={csv_filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@bp.route("/models/<id>/download-feature-importances")
def download_feature_importances(id):
    # Get the model
    model = Model.find_by_id(id)

    if not model:
        return jsonify({"error": "Model not found"}), 404

    # Get the test scores values
    test_feature_importance = model.test_feature_importance

    if not test_feature_importance:
        df = pd.DataFrame.from_dict([{"feature_importance_results": "Feature importance not saved for this model - please retrain."}]).T.reset_index()
    else:
        df = pd.DataFrame.from_dict([model.test_feature_importance]).T.reset_index()
        df.columns = ["feature_name", "feature_importance_value"]

    # CSV file name
    album_id = model.feature_extraction.album_id
    album_details = get_album_details(album_id, g.token)
    model_suffix = album_details["name"]
    if model.feature_collection is not None:
        model_suffix = f"{model_suffix}_{model.feature_collection.name}"

    csv_filename = f"test_feature_importances_{model_suffix}_{model.best_algorithm}_{model.best_data_normalization}_{id}.csv"
    csv_filename = sanitize_filename(csv_filename, "_").replace(" ", "_")

    # Create a string buffer
    output = io.StringIO()

    # Write the DataFrame to the string buffer as CSV
    df.to_csv(output, index=False)

    # Get the CSV content as a string
    csv_content_string = output.getvalue()

    # Close StringIO to free up resources
    output.close()

    # Return the CSV content as a response
    return Response(
        csv_content_string,
        mimetype="text/csv",
        headers={
            "Content-disposition": f"attachment; filename={csv_filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

@bp.route("/models/<id>/feature-importances")
def get_feature_importances(id):
    # Get the model
    model = Model.find_by_id(id)

    if not model:
        return jsonify({"error": "Model not found"}), 404

    # Get the test feature importance
    test_feature_importance = model.test_feature_importance

    if not test_feature_importance:
        return jsonify({"error": "Feature importance not saved for this model - please retrain."}), 404
    
    # Convert to list of dictionaries for easier frontend consumption
    feature_importance_data = [
        {"feature_name": feature_name, "importance_value": importance_value}
        for feature_name, importance_value in test_feature_importance.items()
    ]
    
    # Sort by importance value in descending order
    feature_importance_data.sort(key=lambda x: x["importance_value"], reverse=True)
    
    return jsonify({
        "feature_importances": feature_importance_data,
        "model_id": id,
        "algorithm": model.best_algorithm,
        "normalization": model.best_data_normalization
    })

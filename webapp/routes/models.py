import os
import traceback
import csv
import json
import io
import base64

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathvalidate import sanitize_filename
from sqlalchemy.orm import joinedload
from flask import Blueprint, jsonify, request, g, make_response, Response
from quantimage2_backend_common.models import Model, LabelCategory, Album, Label
from quantimage2_backend_common.utils import get_training_id, format_model
from routes.utils import validate_decorate
from service.feature_extraction import get_album_details
from service.machine_learning import train_model, model_compare_permuation_test

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

@bp.route("/models/<id>/plot-predictions", methods=["GET", "POST"])
def plot_predictions(id):
    if request.method == "POST":
        # Get additional model ID from request body
        print("Request JSON:", request.json)  # Debug print
        model_ids_str = request.json.get('model_ids', '')
        print("Model IDs string:", model_ids_str)  # Debug print

        # Split and clean the model IDs string
        if isinstance(model_ids_str, str):
            model_ids = [x.strip() for x in model_ids_str.split(',') if x.strip()]
        else:
            model_ids = []
    else:
        # Handle GET request - split the URL parameter if it contains commas
        model_ids = [x.strip() for x in id.split(',') if x.strip()]

    # Convert model IDs to integers
    model_ids = [int(mid) for mid in model_ids]

    # Create the plot with increased height and width
    plt.figure(figsize=(12, 6))  # Increased size for better spacing

    # Add background colors for prediction regions
    plt.axvspan(0, 0.5, color='lightblue', alpha=0.3, label='Prediction Region: Class 0')
    plt.axvspan(0.5, 1, color='mistyrose', alpha=0.3, label='Prediction Region: Class 1')

    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')

    # Different y-positions for different models with more separation
    y_positions = [-0.3, 0.1] if len(model_ids) > 1 else [-0.2]

    def adjust_label_positions(x_positions, base_offset, min_distance=0.05):
        """Adjust vertical offsets for overlapping labels by alternating above/below"""
        n = len(x_positions)
        y_offsets = [base_offset] * n

        # Sort points by x position and get indices
        idx_sorted = np.argsort(x_positions)
        x_sorted = x_positions[idx_sorted]

        # Find groups of overlapping points
        current_group = []
        groups = []

        for i in range(n):
            if not current_group or abs(x_sorted[i] - x_sorted[current_group[-1]]) < min_distance:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [i]

        if len(current_group) > 1:
            groups.append(current_group)

        # Adjust offsets for each group
        for group in groups:
            for i, idx in enumerate(group):
                real_idx = idx_sorted[idx]
                if i % 2 == 0:
                    y_offsets[real_idx] = base_offset  # Keep original offset
                else:
                    y_offsets[real_idx] = -base_offset  # Flip offset

        return y_offsets

    # Process each model
    for model_id, y_position in zip(model_ids, y_positions):
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

        # Prepare data for plotting
        probabilities = []
        predictions = []
        patient_ids = []
        ground_truths = []

        for patient_id in test_predictions.keys():
            pred = test_predictions[patient_id]["prediction"]
            prob = test_probabilities[patient_id]["probabilities"][1]
            gt = ground_truth.get(patient_id, None)

            if gt is not None:
                probabilities.append(prob)
                predictions.append(pred)
                patient_ids.append(patient_id)
                ground_truths.append(gt)

        # Create scatter plot for each class based on ground truth - smaller points
        zeros = np.array(ground_truths) == 0
        ones = np.array(ground_truths) == 1

        # Plot points with different colors based on ground truth - smaller points
        plt.scatter(np.array(probabilities)[zeros], np.zeros(sum(zeros)) + y_position, 
                   c='blue', label='Ground Truth: Class 0' if model_id == model_ids[0] else "", 
                   alpha=0.6, s=50)  # Reduced point size
        plt.scatter(np.array(probabilities)[ones], np.zeros(sum(ones)) + y_position,
                   c='red', label='Ground Truth: Class 1' if model_id == model_ids[0] else "", 
                   alpha=0.6, s=50)  # Reduced point size

        # Add patient IDs as annotations with overlap prevention
        base_offset = 20  # Increased base offset for better separation
        y_offsets = adjust_label_positions(np.array(probabilities), base_offset)

        for i, (txt, x_pos, y_offset) in enumerate(zip(patient_ids, probabilities, y_offsets)):
            plt.annotate(txt, 
                        (x_pos, y_position), 
                        xytext=(0, y_offset),
                        textcoords='offset points',
                        ha='center',
                        va='bottom' if y_offset > 0 else 'top',
                        fontsize=6,
                        rotation=90)

        # Add model ID labels on y-axis
        plt.text(-0.1, y_position, f'Model {model_id}', 
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=8)

        # Get metrics from the model
        test_metrics = model.test_metrics
        metrics_text = [
            f"Model {model_id}:" if len(model_ids) > 1 else "Metrics:",
            f"AUC: {test_metrics['auc']['mean']:.3f}",
            f"Prec: {test_metrics['precision']['mean']:.3f}",
            f"Sens: {test_metrics['sensitivity']['mean']:.3f}",
            f"Spec: {test_metrics['specificity']['mean']:.3f}"
        ]

        # Add metrics text as a separate legend for each model - smaller font
        x_pos = 0.02 if len(model_ids) == 1 or model_id == model_ids[0] else 0.25
        metrics_legend = plt.legend([plt.Rectangle((0, 0), 1, 1, fc='none', fill=False, 
                                                 edgecolor='none', linewidth=0)]*5,
                                  metrics_text,
                                  loc='upper left',
                                  bbox_to_anchor=(x_pos, 1),
                                  title=None,
                                  framealpha=0.9,
                                  fontsize=7)
        plt.gca().add_artist(metrics_legend)

    # Main legend for plot elements - smaller font and more compact
    plt.legend(loc='upper right', 
              framealpha=0.9, 
              ncol=2,  # Two columns for more compact legend
              fontsize=7,
              bbox_to_anchor=(0.99, 0.99))

    # Customize the plot
    plt.xlabel('Probability of Class 1', fontsize=9)
    plt.yticks([])  # Hide numerical y-ticks since we have model labels
    plt.grid(True, alpha=0.2)  # Reduced grid opacity
    plt.ylim(-0.5, 0.5)  # Increased y-range for better spacing
    plt.xlim(-0.05, 1.05)
    plt.title('Prediction Probabilities Distribution Test Set', 
             fontsize=10, pad=10)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()

    # Return the image as a file download
    return Response(
        buf.getvalue(),
        mimetype="image/png",
        headers={
            "Content-disposition": f"attachment; filename=predictions_plot_{'_'.join(map(str, model_ids))}.png",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

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

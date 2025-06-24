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
from routes.utils import validate_decorate, adjust_label_positions
from service.feature_extraction import get_album_details
from service.machine_learning import train_model, model_compare_permuation_test


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
                user_id,
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
        model_ids_str = request.json.get('model_ids', '')
        if isinstance(model_ids_str, str):
            model_ids = [x.strip() for x in model_ids_str.split(',') if x.strip()]
        else:
            model_ids = []
    else:
        model_ids = [x.strip() for x in id.split(',') if x.strip()]

    model_ids = [int(mid) for mid in model_ids]
    
    # Check model type from first model to determine plot type
    first_model = Model.find_by_id(model_ids[0])
    if not first_model:
        return jsonify({"error": f"Model {model_ids[0]} not found"}), 404
    
    label_category = LabelCategory.find_by_id(first_model.label_category_id)
    is_survival = label_category.label_type == "Survival"
    
    if is_survival:
        return _plot_survival_predictions(model_ids, "test")
    else:
        return _plot_classification_predictions(model_ids, "test")

@bp.route("/models/<id>/plot-train-predictions", methods=["GET", "POST"])
def plot_train_predictions(id):
    if request.method == "POST":
        model_ids_str = request.json.get('model_ids', '')
        if isinstance(model_ids_str, str):
            model_ids = [x.strip() for x in model_ids_str.split(',') if x.strip()]
        else:
            model_ids = []
    else:
        model_ids = [x.strip() for x in id.split(',') if x.strip()]

    model_ids = [int(mid) for mid in model_ids]
    
    # Check model type from first model to determine plot type
    first_model = Model.find_by_id(model_ids[0])
    if not first_model:
        return jsonify({"error": f"Model {model_ids[0]} not found"}), 404
    
    label_category = LabelCategory.find_by_id(first_model.label_category_id)
    is_survival = label_category.label_type == "Survival"
    
    if is_survival:
        return _plot_survival_predictions(model_ids, "train")
    else:
        return _plot_classification_predictions(model_ids, "train")

def _plot_classification_predictions(model_ids, prediction_type):
    """Helper function for classification model plots - handles both test and train"""
    # Create the plot with increased height and width
    plt.figure(figsize=(12, 6))  # Increased size for better spacing

    # Add background colors for prediction regions
    plt.axvspan(0, 0.5, color='lightblue', alpha=0.3, label='Prediction Region: Class 0')
    plt.axvspan(0.5, 1, color='mistyrose', alpha=0.3, label='Prediction Region: Class 1')

    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')

    # Different y-positions for different models with more separation
    y_positions = [-0.3, 0.1] if len(model_ids) > 1 else [-0.2]

    # Process each model
    for model_id, y_position in zip(model_ids, y_positions):
        model = Model.find_by_id(model_id)
        if not model:
            return jsonify({"error": f"Model {model_id} not found"}), 404

        # Get predictions and probabilities based on prediction_type
        if prediction_type == "test":
            predictions = model.test_predictions
            probabilities_data = model.test_predictions_probabilities
            metrics = model.test_metrics
            title = 'Prediction Probabilities Distribution Test Set'
        else:  # train
            predictions = model.train_predictions
            probabilities_data = model.train_predictions_probabilities
            metrics = model.training_metrics
            title = 'Prediction Probabilities Distribution Complete Train Set'

        # Create dictionary to store ground truth labels
        ground_truth = {}
        labels = Label.find_by_label_category(model.label_category_id)
        for label in labels:
            label_value = next(iter(label.label_content.values()))
            ground_truth[label.patient_id] = int(label_value)

        # Prepare data for plotting
        probabilities = []
        patient_ids = []
        ground_truths = []

        for patient_id in predictions.keys():
            prob = probabilities_data[patient_id]["probabilities"][1]
            gt = ground_truth.get(patient_id, None)

            if gt is not None:
                probabilities.append(prob)
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
        metrics_text = [
            f"Model {model_id}:" if len(model_ids) > 1 else "Metrics:",
            f"AUC: {metrics['auc']['mean']:.3f} ({metrics['auc']['inf_value']:.3f} - {metrics['auc']['sup_value']:.3f})",
            f"Prec: {metrics['precision']['mean']:.3f} ({metrics['precision']['inf_value']:.3f} - {metrics['precision']['sup_value']:.3f})",
            f"Sens: {metrics['sensitivity']['mean']:.3f} ({metrics['sensitivity']['inf_value']:.3f} - {metrics['sensitivity']['sup_value']:.3f})",
            f"Spec: {metrics['specificity']['mean']:.3f} ({metrics['specificity']['inf_value']:.3f} - {metrics['specificity']['sup_value']:.3f})"
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
    plt.title(title, fontsize=10, pad=10)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()

    # Create interactive plotly figure instead of matplotlib
    fig = go.Figure()
    
    # Add background regions
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=0.5, y1=0.5,
                  fillcolor="lightblue", opacity=0.3, layer="below")
    fig.add_shape(type="rect", x0=0.5, y0=-0.5, x1=1, y1=0.5,
                  fillcolor="mistyrose", opacity=0.3, layer="below")
    
    # Add threshold line
    fig.add_vline(x=0.5, line_dash="dash", line_color="black")
    
    # Add scatter points for each model
    for model_id, y_position in zip(model_ids, y_positions):
        # ... process model data ...
        
        # Add scatter points with hover info
        fig.add_trace(go.Scatter(
            x=np.array(probabilities)[zeros],
            y=np.zeros(sum(zeros)) + y_position,
            mode='markers',
            name=f'Model {model_id} - Class 0',
            marker=dict(color='blue', size=8),
            text=[f"Patient: {pid}" for pid in np.array(patient_ids)[zeros]],
            hovertemplate="<b>%{text}</b><br>Probability: %{x:.3f}<br>Ground Truth: Class 0"
        ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Prediction Probabilities",
        xaxis_title="Probability of Class 1",
        showlegend=True,
        height=600
    )
    
    # Return as interactive HTML or JSON
    return jsonify({
        "plot_html": fig.to_html(include_plotlyjs="cdn"),
        # or
        "plot_json": fig.to_json()
    })
    
def _plot_survival_predictions(model_ids, prediction_type):
    """Helper function for survival model plots - handles both test and train"""
    # Create the plot with increased height and width
    plt.figure(figsize=(12, 8))

    # Different colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Process each model
    for idx, model_id in enumerate(model_ids):
        model = Model.find_by_id(model_id)
        if not model:
            return jsonify({"error": f"Model {model_id} not found"}), 404

        # Get predictions based on prediction_type
        if prediction_type == "test":
            predictions = model.test_predictions
            metrics = model.test_metrics
            title = 'Survival Analysis: Risk Score vs Survival Time (Test Set)'
        else:  # train
            predictions = model.train_predictions
            metrics = model.training_metrics
            title = 'Survival Analysis: Risk Score vs Survival Time (Train Set)'
        
        print(predictions)
        # Get ground truth labels (Time and Event)
        ground_truth = {}
        labels = Label.find_by_label_category(model.label_category_id)
        for label in labels:
            ground_truth[label.patient_id] = {
                'time': float(label.label_content.get('Time', 0)),
                'event': int(label.label_content.get('Event', 0))
            }
        print(ground_truth)
        # Prepare data for plotting
        risk_scores = []
        times = []
        events = []
        patient_ids = []

        for patient_id in predictions.keys():
            risk_score = predictions[patient_id].get("risk_score", 0)
            gt = ground_truth.get(patient_id, None)
            
            if gt is not None:
                risk_scores.append(risk_score)
                times.append(gt['time'])
                events.append(gt['event'])
                patient_ids.append(patient_id)

        # Convert to numpy arrays for easier manipulation
        risk_scores = np.array(risk_scores)
        times = np.array(times)
        events = np.array(events)

        # Use different color for each model
        base_color = colors[idx % len(colors)]
        
        # Plot events (deaths) as filled circles
        event_mask = events == 1
        if np.any(event_mask):
            plt.scatter(times[event_mask], risk_scores[event_mask], 
                       c=base_color, marker='o', s=60, alpha=0.8,
                       label=f'Model {model_id} - Events' if idx == 0 else f'M{model_id} - Events',
                       edgecolors='black', linewidth=0.5)
        
        # Plot censored observations as empty circles with thicker edge
        censored_mask = events == 0
        if np.any(censored_mask):
            plt.scatter(times[censored_mask], risk_scores[censored_mask], 
                       c='white', marker='o', s=60, alpha=0.8, 
                       edgecolors=base_color, linewidth=2,
                       label=f'Model {model_id} - Censored' if idx == 0 else f'M{model_id} - Censored')

        # Add patient IDs as annotations with overlap prevention
        for i, (txt, x_pos, y_pos) in enumerate(zip(patient_ids, times, risk_scores)):
            plt.annotate(txt, 
                        (x_pos, y_pos), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        ha='left',
                        va='bottom',
                        fontsize=6,
                        alpha=0.7)

        # Add trend line to show risk-time relationship
        if len(risk_scores) > 1:
            # Fit polynomial trend line
            z = np.polyfit(times, risk_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(times.min(), times.max(), 100)
            plt.plot(x_trend, p(x_trend), '--', color=base_color, alpha=0.6, 
                    linewidth=2, label=f'Model {model_id} - Trend')

        # Add metrics text for each model - positioned below the legend to avoid overlap
        if metrics:
            c_index = metrics.get('c-index', {}).get('mean', 'N/A')
            
            # Create metrics text
            if isinstance(c_index, float):
                c_index_inf = metrics.get('c-index', {}).get('inf_value', c_index)
                c_index_sup = metrics.get('c-index', {}).get('sup_value', c_index)
                metrics_text = f"Model {model_id} - C-index: {c_index:.3f} ({c_index_inf:.3f} - {c_index_sup:.3f})"
            else:
                metrics_text = f"Model {model_id} - C-index: {c_index}"
            
            # Position metrics text below the legend on the right side
            y_pos = 0.85 - idx * 0.06  # Start lower and space them out
            plt.text(0.98, y_pos, metrics_text,
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Customize the plot
    plt.xlabel('Survival Time', fontsize=12)
    plt.ylabel('Risk Score', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Legend positioning - keep at upper right
    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Add explanatory text at bottom
    plt.figtext(0.5, 0.02, 
                'Filled circles = Events (deaths), Empty circles = Censored observations, Dashed lines = Risk trends',
                ha='center', fontsize=10, style='italic')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()

    # Return the image as a file download
    filename = f"survival_{'test' if prediction_type == 'test' else 'train'}_predictions_plot_{'_'.join(map(str, model_ids))}.png"
    return Response(
        buf.getvalue(),
        mimetype="image/png",
        headers={
            "Content-disposition": f"attachment; filename={filename}",
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

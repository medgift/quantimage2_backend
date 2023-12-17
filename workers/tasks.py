"""
Celery tasks for running and finalizing feature extractions
"""
import os
import logging
import re
import shutil
import socket
import tempfile
import traceback
from multiprocessing import current_process

import joblib
import pydevd_pycharm
import requests
import warnings

from typing import Dict, Any

from flask import jsonify
from flask_socketio import SocketIO
from celery import Celery
from celery import states as celerystates
from celery.signals import celeryd_after_setup
from zipfile import ZipFile

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from ttictoc import tic, toc

from quantimage2_backend_common.const import FAKE_SCORER_KEY, TRAINING_PHASES
from quantimage2_backend_common.feature_storage import store_features
from quantimage2_backend_common.flask_init import create_app
from quantimage2_backend_common.models import (
    FeatureExtractionTask,
    db,
    FeatureExtraction,
    Model,
)
from quantimage2_backend_common.kheops_utils import get_token_header
from quantimage2_backend_common.utils import (
    get_socketio_body_feature_task,
    MessageType,
    task_status_message,
    fetch_extraction_result,
    send_extraction_status_message,
    format_model,
)

from okapy.dicomconverter.converter import ExtractorConverter

from utils import (
    calculate_training_metrics,
    run_bootstrap,
    calculate_test_metrics,
    get_model_path,
)

warnings.filterwarnings("ignore", message="Failed to parse headers")

from quantimage2_backend_common.kheops_utils import endpoints

# Setup Debugger
if "DEBUGGER_IP" in os.environ and os.environ["DEBUGGER_IP"] != "":
    try:
        pydevd_pycharm.settrace(
            os.environ["DEBUGGER_IP"],
            port=int(os.environ["DEBUGGER_PORT"]),
            suspend=False,
            stderrToServer=True,
            stdoutToServer=True,
        )
    except ConnectionRefusedError:
        logging.warning("No debug server running")
    except socket.timeout:
        logging.warning("Could not connect to the debugger")


celery = Celery(
    "tasks",
    backend=os.environ["CELERY_RESULT_BACKEND"],
    broker=os.environ["CELERY_BROKER_URL"],
)
celery.conf.accept_content = ["pickle", "json"]


@celeryd_after_setup.connect
def setup(sender, instance, **kwargs):
    """
    Run once the Celery worker has started.

    Initialize the Flask-SocketIO instance.
    """

    # Backend client
    global socketio

    socketio = SocketIO(message_queue=os.environ["SOCKET_MESSAGE_QUEUE"])

    # Create basic Flask app and push an app context to allow DB operations
    flask_app = create_app()
    flask_app.app_context().push()


@celery.task(name="quantimage2tasks.train", bind=True)
def train_model(
    self,
    *,
    feature_extraction_id,
    collection_id,
    album,
    feature_selection,
    feature_names,
    pipeline,
    parameter_grid,
    estimator_step,
    scoring,
    refit_metric,
    cv,
    n_jobs,
    X_train,
    X_test,
    label_category,
    data_splitting_type,
    train_test_splitting_type,
    training_patients,
    test_patients,
    y_train_encoded,
    y_test_encoded,
    is_train_test,
    random_seed,
    training_id,
    user_id,
):
    db.session.commit()

    # TODO - This is a hack, would be best to find a better solution such as Dask
    current_process()._config["daemon"] = False

    # Fake scorer allowing to send progress reports
    def fake_score(y_true, y_pred, training_id=None):
        # Create basic Flask app and push an app context to allow DB operations
        training_flask_app = create_app()
        training_flask_app.app_context().push()

        training_socketio = SocketIO(message_queue=os.environ["SOCKET_MESSAGE_QUEUE"])
        socketio_body = {
            "training-id": training_id,
            "phase": TRAINING_PHASES.TRAINING.value,
        }
        training_socketio.emit(
            MessageType.TRAINING_STATUS.value, jsonify(socketio_body).get_json()
        )
        return 0

    tic()

    # Add fake scorer for progress monitoring
    scoring = {
        **scoring,
        FAKE_SCORER_KEY: make_scorer(fake_score, training_id=training_id),
    }

    try:
        # Run grid search on the defined pipeline & search space
        grid = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=False,
            verbose=100,
        )

        fitted_model = grid.fit(X_train, y_train_encoded)

        elapsed = toc()

        print(f"Fitting the model took {elapsed}")

        # Remove references to "fake" scorer in the fitted model for serialization & metrics calculation
        fitted_model.cv_results_ = {
            k: fitted_model.cv_results_[k]
            for k in fitted_model.cv_results_
            if not re.match(rf".*{FAKE_SCORER_KEY}$", k)
        }
        del fitted_model.scorer_[FAKE_SCORER_KEY]
        del fitted_model.scoring[FAKE_SCORER_KEY]

        # Calculate Training Metrics
        training_metrics = calculate_training_metrics(
            fitted_model.best_index_,
            fitted_model.cv_results_,
            scoring,
            random_seed,
        )
        test_metrics = None
        test_metrics_values = None

        # Train/test only - Perform Bootstrap on the Test set
        if is_train_test:
            tic()
            socket_io = SocketIO(message_queue=os.environ["SOCKET_MESSAGE_QUEUE"])
            scores, n_bootstrap = run_bootstrap(
                X_test,
                y_test_encoded,
                fitted_model,
                random_seed,
                scoring,
                training_id=training_id,
                socket_io=socket_io,
                n_bootstrap=100,
            )
            elapsed = toc()
            print(f"Running bootstrap took {elapsed}")

            test_metrics, test_metrics_values = calculate_test_metrics(scores, scoring, random_seed)

        # Save model in the DB
        classifier_class = fitted_model.best_params_[estimator_step]

        # Best algorithm - Check underlying estimator class name if a CalibratedClassifierCV was used
        best_algorithm = (
            classifier_class.__class__.__name__
            if not hasattr(classifier_class, "estimator")
            else classifier_class.estimator.__class__.__name__
        )
        best_normalization = fitted_model.best_params_[
            "preprocessor"
        ].__class__.__name__

        model_path = get_model_path(
            user_id,
            album["album_id"],
            label_category.label_type,
        )

        # Persist model in DB and on disk (pickle it)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(fitted_model, model_path)

        # Generate model name (for now)
        (file, ext) = os.path.splitext(os.path.basename(model_path))
        model_name = f"{album['name']}_{file}"

        # Get all other metadata for the model to be saved in the DB
        training_validation = "Repeated Stratified K-Fold Cross-Validation"
        training_validation_params = {"k": cv.cvargs["n_splits"], "n": cv.n_repeats}
        test_validation = "Bootstrap" if is_train_test else None
        test_validation_params = {"n": n_bootstrap} if is_train_test else None

        db_model = Model(
            model_name,
            best_algorithm,
            data_splitting_type,
            train_test_splitting_type,
            f"{training_validation} ({training_validation_params['k']} folds, {training_validation_params['n']} repetitions)",
            f"{test_validation} ({test_validation_params['n']} repetitions)"
            if test_validation
            else None,
            best_normalization,
            feature_selection,
            feature_names,
            training_patients,
            test_patients,
            model_path,
            training_metrics,
            test_metrics,
            test_metrics_values,
            user_id,
            album["album_id"],
            label_category.id,
            feature_extraction_id,
            collection_id,
        )
        db_model.save_to_db()

        socketio_body = {
            "training-id": training_id,
            "complete": True,
            "model": format_model(db_model),
        }
    except Exception as e:
        socketio_body = {"training-id": training_id, "failed": True, "error": str(e)}
        logging.error(e)
    finally:
        socketio.emit(
            MessageType.TRAINING_STATUS.value, jsonify(socketio_body).get_json()
        )
        db.session.remove()


@celery.task(name="quantimage2tasks.extract", bind=True)
def run_extraction(
    self,
    feature_extraction_id,
    user_id,
    feature_extraction_task_id,
    study_uid,
    album_id,
    album_name,
    album_token,
    config_path,
    rois,
):
    try:

        current_step = 1
        steps = 2

        db.session.commit()

        # Affect celery task ID to task (if needed)
        print(f"Getting Feature Extraction Task with id {feature_extraction_task_id}")
        feature_extraction_task = FeatureExtractionTask.find_by_id(
            feature_extraction_task_id
        )

        feature_extraction = FeatureExtraction.find_by_id(feature_extraction_id)

        if not feature_extraction_task:
            raise Exception("Didn't find the task in the DB!!!")

        if not feature_extraction_task.task_id:
            feature_extraction_task.task_id = self.request.id
            feature_extraction_task.save_to_db()

        # Status update - DOWNLOAD
        status_message = "Fetching DICOM files"
        update_progress(
            self,
            feature_extraction_id,
            feature_extraction_task_id,
            current_step,
            steps,
            status_message,
        )

        # Download study and write files to directory
        dicom_dir = download_study(album_token, study_uid, album_id)

        # Extract all the features
        features = extract_all_features(
            self,
            dicom_dir,
            config_path,
            rois,
            feature_extraction_id,
            feature_extraction_task_id=feature_extraction_task_id,
            current_step=current_step,
            steps=steps,
            album_name=album_name,
        )

        # Delete download DIR
        shutil.rmtree(dicom_dir, True)

        # Save the features
        store_features(
            feature_extraction_task_id,
            feature_extraction_id,
            features,
        )

        # Extraction is complete
        status_message = "Extraction Complete"

        # TODO Find a better way to do this, not from the task level hopefully
        # Check if parent is done
        extraction_status = fetch_extraction_result(
            celery, feature_extraction.result_id
        )

        return {
            "feature_extraction_task_id": feature_extraction_task_id,
            "current": steps,
            "total": steps,
            "completed": steps,
            "status_message": status_message,
        }
    except Exception as e:

        logging.error(e)

        current_step = 0
        status_message = "Failure!"
        print(
            f"Feature extraction task {feature_extraction_task_id} - {current_step}/{steps} - {status_message}"
        )

        meta = {
            "exc_type": type(e).__name__,
            "exc_message": traceback.format_exc().split("\n"),
            "feature_extraction_task_id": feature_extraction_task_id,
            "current": current_step,
            "total": steps,
            "status_message": status_message,
        }

        update_task_state(self, celerystates.FAILURE, meta)

        print("PROBLEM WHEN EXTRACTING STUDY WITH UID  " + study_uid)

        raise e

    finally:
        db.session.remove()


@celery.task(name="quantimage2tasks.finalize_extraction", bind=True)
def finalize_extraction(task, results, feature_extraction_id):
    send_extraction_status_message(
        feature_extraction_id, celery, socketio, send_extraction=True
    )
    db.session.remove()


@celery.task(name="quantimage2tasks.finalize_extraction_task", bind=True)
def finalize_extraction_task(
    task, result, feature_extraction_id, feature_extraction_task_id
):
    # Send Socket.IO message
    socketio_body = get_socketio_body_feature_task(
        task.request.id,
        feature_extraction_task_id,
        celerystates.SUCCESS,
        "Extraction Complete",
    )

    # Send Socket.IO message to clients about task
    socketio.emit(MessageType.FEATURE_TASK_STATUS.value, socketio_body)

    # Send Socket.IO message to clients about extraction
    send_extraction_status_message(feature_extraction_id, celery, socketio)

    db.session.remove()


def update_progress(
    task: celery.Task,
    feature_extraction_id: int,
    feature_extraction_task_id: int,
    current_step: int,
    steps: int,
    status_message: str,
) -> None:
    """
    Update the progress of a feature extraction Task

    :param task: The Celery Task associated with the Feature Extraction Task
    :param feature_extraction_id: The ID of the Feature Extraction to update (can include several tasks)
    :param feature_extraction_task_id: The ID of the specific Feature Task to update
    :param current_step: The current step (out of N steps) in the extraction process
    :param steps: The total number of steps in the extraction process
    :param status_message: The status message to show for the task (Downloading, Converting, etc.)
    :returns: None
    """
    print(
        f"Feature extraction task {feature_extraction_task_id} - {current_step}/{steps} - {status_message}"
    )

    # TODO : Make this an enum
    status = "PROGRESS"

    meta = {
        "task_id": task.request.id,
        "feature_extraction_task_id": feature_extraction_task_id,
        "current": current_step,
        "total": steps,
        "completed": current_step - 1,
        "status_message": status_message,
    }

    # Update task state in Celery
    update_task_state(task, status, meta)

    # Send Socket.IO message
    socketio_body = get_socketio_body_feature_task(
        task.request.id,
        feature_extraction_task_id,
        status,
        task_status_message(current_step, steps, status_message),
    )

    # Send Socket.IO message to clients about task
    socketio.emit(MessageType.FEATURE_TASK_STATUS.value, socketio_body)

    # Send Socket.IO message to clients about extraction
    send_extraction_status_message(feature_extraction_id, celery, socketio)


def update_task_state(task: celery.Task, state: str, meta: Dict[str, Any]) -> None:
    """
    Update the State of a Celery Task

    :param task: The Celery Task to update
    :param state: DICOM Study UID
    :param meta: A Dictionary with state values
    :returns: None
    """
    task.update_state(state=state, meta=meta)


def download_study(token: str, study_uid: str, album_id: str) -> str:
    """ "
    Download a study and write all files to a directory

    :param token: Valid access token for the backend
    :param study_uid: The UID of the study to download
    :param album_id: The ID of the Kheops album to filter the output by
    :returns: Path to the directory of downloaded files
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_file = tempfile.mktemp(".zip")

    study_download_url = (
        f"{endpoints.studies}/{study_uid}?accept=application/zip&album={album_id}"
    )

    access_token = get_token_header(token)

    response = requests.get(
        study_download_url,
        headers=access_token,
    )

    # Save to ZIP file
    with open(tmp_file, "wb") as f:
        f.write(response.content)

    # Unzip ZIP file
    with ZipFile(tmp_file, "r") as zipObj:
        for file in zipObj.namelist():
            if file.startswith("DICOM/"):
                zipObj.extract(file, tmp_dir)

    # Remove the ZIP file
    os.unlink(tmp_file)

    return tmp_dir


def get_bytes_from_file(filename):
    return open(filename, "rb").read()


def extract_all_features(
    task: celery.Task,
    dicom_dir: str,
    config_path: str,
    rois: list,
    feature_extraction_id: int,
    feature_extraction_task_id: int = None,
    current_step: int = None,
    steps: int = None,
    album_name: str = None,
) -> Dict[str, Any]:
    """
    Update the progress of a feature extraction Task

    :param task: Celery Task associated with the Feature Extraction Task
    :param dicom_dir: Path to the folder containing all DICOM images
    :param config_path: Path to the YAML config file with the extraction parameters (TODO - Detail more)
    :param feature_extraction_id: The ID of the Feature Extraction (global, can include multiple patients)
    :param feature_extraction_task_id: The ID of the specific Feature Task (for a given study)
    :param current_step: The current step (out of N steps) in the extraction process (should be 1 at this point)
    :param steps: The total number of steps in the extraction process (currently 3 - Download, Conversion, Extraction)
    :param album_name: The name of the Kheops album to which the study belongs (for customizing label extraction)
    :returns: A dictionary with the extracted features
    """
    try:
        # Status update - PROCESS
        current_step += 1
        status_message = "Processing data"
        update_progress(
            task,
            feature_extraction_id,
            feature_extraction_task_id,
            current_step,
            steps,
            status_message,
        )

        # Get results directly from Okapy
        converter = ExtractorConverter.from_params(config_path)

        conversion_result = converter(dicom_dir, labels=rois)

        print(f"!!!!!!!!!!!!Final Features!!!!!!!!!")
        print(conversion_result)

        result = conversion_result

        return result

    except Exception as e:

        # logging.error(e)

        print("!!!!!Caught an error while pre-processing or something!!!!!")

        current_step = 0
        status_message = "Failure!"
        print(
            f"Feature extraction task {feature_extraction_task_id} - {current_step}/{steps} - {status_message}"
        )

        meta = {
            "exc_type": type(e).__name__,
            "exc_message": traceback.format_exc().split("\n"),
            "feature_extraction_task_id": feature_extraction_task_id,
            "current": current_step,
            "total": steps,
            "status_message": status_message,
        }

        update_task_state(task, celerystates.FAILURE, meta)

        # Send Socket.IO message
        socketio_body = get_socketio_body_feature_task(
            task.request.id,
            feature_extraction_task_id,
            celerystates.FAILURE,
            task_status_message(current_step, steps, status_message),
        )

        # Send Socket.IO message to clients about task
        socketio.emit(MessageType.FEATURE_TASK_STATUS.value, socketio_body)

        # Send Socket.IO message to clients about extraction
        send_extraction_status_message(feature_extraction_id, celery, socketio)

        raise e

    finally:
        db.session.remove()

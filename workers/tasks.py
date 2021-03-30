"""
Celery tasks for running and finalizing feature extractions
"""
import os
import logging
import shutil
import tempfile
import json
import traceback
import jsonpickle

import pydevd_pycharm
import requests
import warnings

import yaml

from typing import List, Dict, Any
from flask_socketio import SocketIO
from keycloak.realm import KeycloakRealm
from requests_toolbelt import NonMultipartContentTypeException
from celery import Celery
from celery import states as celerystates
from celery.signals import celeryd_after_setup
from requests_toolbelt.multipart import decoder
from pathlib import Path
from zipfile import ZipFile

from imaginebackend_common.feature_storage import store_features
from imaginebackend_common.flask_init import create_app
from imaginebackend_common.models import FeatureExtractionTask, db, FeatureExtraction
from imaginebackend_common.kheops_utils import get_token_header, dicomFields
from imaginebackend_common.utils import (
    get_socketio_body_feature_task,
    MessageType,
    task_status_message,
    fetch_extraction_result,
    send_extraction_status_message,
)

from okapy.dicomconverter.converter import ExtractorConverter

warnings.filterwarnings("ignore", message="Failed to parse headers")

from imaginebackend_common.kheops_utils import endpoints

# Setup Debugger
if "DEBUGGER_IP" in os.environ and os.environ["DEBUGGER_IP"] != "":
    try:
        pydevd_pycharm.settrace(
            os.environ["DEBUGGER_IP"],
            port=int(os.environ["DEBUGGER_PORT_CELERY"]),
            suspend=False,
            stderrToServer=True,
            stdoutToServer=True,
        )
    except ConnectionRefusedError:
        logging.warning("No debug server running")

celery = Celery(
    "tasks",
    backend=os.environ["CELERY_RESULT_BACKEND"],
    broker=os.environ["CELERY_BROKER_URL"],
)


@celeryd_after_setup.connect
def setup(sender, instance, **kwargs):
    """
    Run once the Celery worker has started.

    Initialize the Keycloak OpenID client & the Flask-SocketIO instance.
    """

    # Backend client
    realm = KeycloakRealm(
        server_url=os.environ["KEYCLOAK_BASE_URL"],
        realm_name=os.environ["KEYCLOAK_REALM_NAME"],
    )
    global oidc_client, socketio
    oidc_client = realm.open_id_connect(
        client_id=os.environ["KEYCLOAK_IMAGINE_CLIENT_ID"],
        client_secret=os.environ["KEYCLOAK_IMAGINE_CLIENT_SECRET"],
    )

    socketio = SocketIO(message_queue=os.environ["SOCKET_MESSAGE_QUEUE"])

    # Create basic Flask app and push an app context to allow DB operations
    flask_app = create_app()
    flask_app.app_context().push()


@celery.task(name="imaginetasks.extract", bind=True)
def run_extraction(
    self,
    feature_extraction_id,
    user_id,
    feature_extraction_task_id,
    study_uid,
    # features_path,
    album_name,
    config_path,
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

        # Get a token for the given user (possible thanks to token exchange in Keycloak)
        token = oidc_client.token_exchange(
            requested_token_type="urn:ietf:params:oauth:token-type:access_token",
            audience=os.environ["KEYCLOAK_IMAGINE_CLIENT_ID"],
            requested_subject=user_id,
        )["access_token"]

        # Download study and write files to directory
        dicom_dir = download_study(token, study_uid)

        # Extract all the features
        features = extract_all_features(
            self,
            dicom_dir,
            config_path,
            feature_extraction_id,
            feature_extraction_task_id=feature_extraction_task_id,
            current_step=current_step,
            steps=steps,
            album_name=album_name,
        )

        # Delete download DIR
        shutil.rmtree(dicom_dir, True)

        # Save the features
        # json_features = jsonpickle.encode(features)
        # os.makedirs(os.path.dirname(features_path), exist_ok=True)
        # Path(features_path).write_text(json_features)
        store_features(
            feature_extraction_task_id, feature_extraction_id, features,
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


@celery.task(name="imaginetasks.finalize_extraction", bind=True)
def finalize_extraction(task, results, feature_extraction_id):
    send_extraction_status_message(
        feature_extraction_id, celery, socketio, send_extraction=True
    )
    db.session.remove()


@celery.task(name="imaginetasks.finalize_extraction_task", bind=True)
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


def download_study(token: str, study_uid: str) -> str:
    """"
    Download a study and write all files to a directory

    :param token: Valid access token for the backend
    :param study_uid: The UID of the study to download
    :returns: Path to the directory of downloaded files
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_file = tempfile.mktemp(".zip")

    study_metadata_url = endpoints.studies + "/" + study_uid + "?accept=application/zip"

    access_token = get_token_header(token)

    response = requests.get(study_metadata_url, headers=access_token,)

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

    # Get content-type header (with boundary)
    # content_type = response.headers["content-type"]
    #
    # file_index = 1
    # for part in decoder.MultipartDecoder(
    #     get_bytes_from_file(tmp_file), content_type
    # ).parts:
    #     part_content = part.content
    #     part_path = f"{tmp_dir}/{file_index}.dcm"
    #     with open(part_path, "wb") as f:
    #         f.write(part_content)
    #     file_index += 1

    return tmp_dir


def get_bytes_from_file(filename):
    return open(filename, "rb").read()


def extract_all_features(
    task: celery.Task,
    dicom_dir: str,
    config_path: str,
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

        # TODO - Remove these hard-coded cases
        if "HECKTOR" in album_name:
            labels = ["GTVt"]
        elif "Lymphangitis" in album_name:
            labels = ["GTV T", "GTV L"]
        else:
            labels = None

        # Get results directly from Okapy
        converter = ExtractorConverter.from_params(config_path)

        conversion_result = converter(dicom_dir, labels=labels)

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

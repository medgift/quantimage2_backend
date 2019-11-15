import os, logging
import shutil
import tempfile
import json

import jsonpickle
import requests

import warnings

warnings.filterwarnings("ignore", message="Failed to parse headers")

from celery import Celery
from celery.contrib import rdb
from celery.exceptions import Ignore
from requests_toolbelt.multipart import decoder
from pathlib import Path

from config_worker import dicomFields, endpoints

from radiomics import featureextractor

from okapy.dicomconverter.dicom_walker import DicomWalker

from imaginebackend_common.feature_backends import feature_backends_map

celery = Celery(
    "tasks",
    backend=os.environ["CELERY_RESULT_BACKEND"],
    broker=os.environ["CELERY_BROKER_URL"],
)


@celery.task(name="imaginetasks.extract", bind=True)
def run_extraction(self, token, feature_id, study_uid, features_path, config_path):

    try:
        current_step = 1
        steps = 3

        # Status update - DOWNLOAD
        status_message = "Fetching DICOM files"
        update_progress(self, feature_id, current_step, steps, status_message)

        # Get the list of instance URLs
        url_list = get_dicom_url_list(token, study_uid)

        # Download all the DICOM files
        dicom_dir = download_dicom_urls(token, url_list)

        features = extract_all_features(
            self,
            dicom_dir,
            config_path,
            feature_id=feature_id,
            current_step=current_step,
            steps=steps,
        )

        # Delete download DIR
        shutil.rmtree(dicom_dir, True)

        # Save the features
        json_features = jsonpickle.encode(features)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        Path(features_path).write_text(json_features)

        # Extraction is complete
        status_message = "Extraction COMPLETE"

        return {
            "feature_id": feature_id,
            "current": steps,
            "total": steps,
            "status_message": status_message,
        }
    except Exception as e:

        logging.error("!!!!!AYAYAYAYAYYYYYYYYY")
        logging.error(e)
        logging.error("AYAYAYAYAYYYYYYYYY!!!!!")

        # current_step = 0
        # status_message = "Failure!"
        # print(f"Feature {feature_id} - {current_step}/{steps} - {status_message}")
        #
        # meta = {
        #     "exc_type": type(e).__name__,
        #     "exc_message": traceback.format_exc().split("\n"),
        #     "feature_id": feature_id,
        #     "current": current_step,
        #     "total": steps,
        #     "status_message": status_message,
        # }
        #
        # update_task_state(self, "FAILURE", meta)

        raise e


def update_progress(task, feature_id, current_step, steps, status_message):
    print(f"Feature {feature_id} - {current_step}/{steps} - {status_message}")

    meta = {
        "feature_id": feature_id,
        "current": current_step,
        "total": steps,
        "status_message": status_message,
    }

    update_task_state(task, "PROGRESS", meta)


def update_task_state(task, state, meta):
    task.update_state(state=state, meta=meta)


def get_token_header(token):
    return {"Authorization": "Bearer " + token}


def get_dicom_url_list(token, study_uid):
    # Get the metadata items of the study
    study_metadata_url = (
        endpoints.studies + "/" + study_uid + "/" + endpoints.studyMetadataSuffix
    )

    access_token = get_token_header(token)

    study_metadata = requests.get(study_metadata_url, headers=access_token).json()

    # instance_urls = {"CT": [], "PET": []}
    instance_urls = []

    # Go through each study metadata item
    for entry in study_metadata:

        # Determine the modality
        modality = None
        if entry[dicomFields.MODALITY][dicomFields.VALUE][0] == "PT":
            modality = "PET"
        elif entry[dicomFields.MODALITY][dicomFields.VALUE][0] == "CT":
            modality = "CT"
        elif entry[dicomFields.MODALITY][dicomFields.VALUE][0] == "RTSTRUCT":
            modality = "RTSTRUCT"

        # If PET/CT, add URL to corresponding list
        series_uid = entry[dicomFields.SERIES_UID][dicomFields.VALUE][0]
        instance_uid = entry[dicomFields.INSTANCE_UID][dicomFields.VALUE][0]
        final_url = (
            endpoints.studies
            + "/"
            + study_uid
            + endpoints.seriesSuffix
            + "/"
            + series_uid
            + endpoints.instancesSuffix
            + "/"
            + instance_uid
        )
        if modality:
            instance_urls.append(final_url)

    return instance_urls


def download_dicom_urls(token, urls):
    tmp_dir = tempfile.mkdtemp()

    access_token = get_token_header(token)
    for url in urls:
        filename = os.path.basename(url)

        import logging

        logging.getLogger("urllib3").setLevel(logging.ERROR)
        response = requests.get(url, headers=access_token)
        # open(os.path.join(tmp_dir, filename), "wb").write(response.content)

        multipart_data = decoder.MultipartDecoder.from_response(response)

        # Get first part, with the content
        part = multipart_data.parts[0]

        fh = open(os.path.join(tmp_dir, filename), "wb")
        fh.write(part.content)
        fh.close()

    return tmp_dir


def extract_all_features(
    task, dicom_dir, config_path, feature_id=None, current_step=None, steps=None
):
    config = json.loads(Path(config_path).read_text()) if config_path else None

    features_dict = {}

    # Status update - CONVERT
    current_step += 1
    status_message = "Pre-processing data"
    update_progress(task, feature_id, current_step, steps, status_message)

    backend_inputs = {}
    for backend in config["backends"]:
        # if some features were selected for this backend
        if len(config["backends"][backend]["features"]) > 0:
            # get extractor from feature backends
            extractor = feature_backends_map[backend](config["backends"][backend])
            backend_inputs[backend] = extractor.pre_process_data(dicom_dir)

    # Status update - EXTRACT
    current_step += 1
    status_message = "Extracting features"
    update_progress(task, feature_id, current_step, steps, status_message)

    for backend in config["backends"]:
        # if there is pre-processed data for this backend
        if backend in backend_inputs.keys():
            # get extractor from feature backends
            extractor = feature_backends_map[backend](config["backends"][backend])
            print("CONFIG!!!!")
            print(config["backends"][backend])
            print("FEATURES!!!")
            features = extractor.extract_features(backend_inputs[backend])
            print(features)
            features_dict.update(features)

    result = features_dict

    return result

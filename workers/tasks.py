import os
import shutil
import tempfile

import jsonpickle
import requests
import pathlib

from celery import Celery
from requests_toolbelt.multipart import decoder

from config_worker import dicomFields, endpoints, token

from radiomics import featureextractor

from imaginebackend_common.misc_enums import FeatureStatus
from imaginebackend_common.utils import task_status_message

from okapy.dicomconverter.dicom_walker import DicomWalker

celery = Celery(
    "tasks",
    backend=os.environ["CELERY_RESULT_BACKEND"],
    broker=os.environ["CELERY_BROKER_URL"],
)


@celery.task(name="imaginetasks.extract", bind=True)
def run_extraction(self, feature_id, study_uid, features_dir, features_path):

    current_step = 1
    steps = 3

    # Status update - DOWNLOAD
    status_message = "Downloading DICOM files"
    update_progress(self, feature_id, current_step, steps, status_message)

    # Get the list of instance URLs
    url_list = get_dicom_url_list(study_uid)

    # Download all the DICOM files
    dicom_dir = download_dicom_urls(url_list)

    # Status update - CONVERT
    current_step += 1
    status_message = "Converting DICOM files to NRRD volume"
    update_progress(self, feature_id, current_step, steps, status_message)

    # Use Valentin's DicomWalker to convert DICOM to NRRD
    results_dir = convert_dicom_to_nrrd(dicom_dir)

    # Go through results files and extract features
    file_paths = [
        str(filepath.absolute()) for filepath in pathlib.Path(results_dir).glob("**/*")
    ]

    # Status update - EXTRACT
    current_step += 1
    status_message = "Extracting features"
    update_progress(self, feature_id, current_step, steps, status_message)

    features = extract_features(file_paths)

    # Delete download DIR
    shutil.rmtree(dicom_dir, True)

    # Delete results DIR
    shutil.rmtree(results_dir, True)

    # Save the features
    json_features = jsonpickle.encode(features)
    os.makedirs(features_dir, exist_ok=True)
    fh = open(features_path, "w")
    fh.write(json_features)
    fh.close()

    status_message = "Extraction COMPLETE"
    return {
        "feature_id": feature_id,
        "current": 3,
        "total": 3,
        "status": FeatureStatus.COMPLETE,
        "status_message": status_message,
    }


def update_progress(task, feature_id, current_step, steps, status_message):
    meta = {
        "feature_id": feature_id,
        "current": current_step,
        "total": steps,
        "status": FeatureStatus.IN_PROGRESS,
        "status_message": status_message,
    }

    task.update_state(state="PROGRESS", meta=meta)


def get_token_header():
    return {"Authorization": "Bearer " + token}


def convert_dicom_to_nrrd(input_dir, labels=["GTV T"]):
    output_dir = tempfile.mkdtemp()

    walker = DicomWalker(input_dir, output_dir, list_labels=labels)
    walker.walk()
    walker.fill_images()
    walker.convert()

    return output_dir


def get_dicom_url_list(study_uid):
    # Get the metadata items of the study
    study_metadata_url = (
        endpoints.studies + "/" + study_uid + "/" + endpoints.studyMetadataSuffix
    )

    study_metadata = requests.get(study_metadata_url, headers=get_token_header()).json()

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


def download_dicom_urls(urls):
    tmp_dir = tempfile.mkdtemp()

    for url in urls:
        filename = os.path.basename(url)
        response = requests.get(url, headers=get_token_header())
        # open(os.path.join(tmp_dir, filename), "wb").write(response.content)
        multipart_data = decoder.MultipartDecoder.from_response(response)

        # Get first part, with the content
        part = multipart_data.parts[0]

        fh = open(os.path.join(tmp_dir, filename), "wb")
        fh.write(part.content)
        fh.close()

    return tmp_dir


def extract_features(file_paths):
    ct_path = None
    labels_path = None
    for file_path in file_paths:
        if not ct_path and "ct" in file_path:
            ct_path = file_path

        if not labels_path and "rtstruct" in file_path:
            labels_path = file_path

    extractor = featureextractor.RadiomicsFeatureExtractor()

    result = extractor.execute(ct_path, labels_path)

    return result

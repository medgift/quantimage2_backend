import io
import os.path
from subprocess import CalledProcessError
from zipfile import ZipFile

import requests
import pydicom

import glob

from Naked.toolshed.shell import muterun_js
from flask import Blueprint, jsonify, request, g, current_app, Response
from ttictoc import tic, toc

from imaginebackend_common.kheops_utils import dicomFields, get_token_header
from imaginebackend_common.models import Album

from multiprocessing.pool import ThreadPool

from tempfile import TemporaryDirectory

# Define blueprint
from routes.utils import validate_decorate
from service.feature_extraction import get_studies_from_album, get_series_from_study

bp = Blueprint(__name__, "albums")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/albums/<album_id>", methods=("GET", "PATCH"))
def album_rois(album_id, rois=None):
    if request.method == "PATCH":
        return save_rois(album_id, rois)

    if request.method == "GET":
        return get_rois(album_id)


def save_rois(album_id, rois):
    album = Album.save_rois(album_id, rois)
    return jsonify(album.to_dict())


def get_rois(album_id):
    album = Album.find_by_album_id(album_id)

    if album.rois is None:
        rois_map = get_rois_from_kheops(album_id)
        Album.save_rois(album_id, rois_map)

    return album.rois


def get_rois_from_kheops(album_id):
    token = g.token

    studies = get_studies_from_album(album_id, token)

    roi_series = []

    # Get list of ROI series to download
    for study in studies:
        # TODO - This might be more dynamic, not hard-coded to RTSTRUCT & SEG
        series = get_series_from_study(
            study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
            ["RTSTRUCT", "SEG"],
            token,
        )
        roi_series += series

    # Download all ROI series instances
    (roi_file_paths, temp_dir) = download_roi_files(roi_series, token)

    # read_dicom_files_sync(roi_file_paths)
    # read_dicom_files_sync(roi_file_paths)

    # Create map of ROI -> Number of studies from ROI files
    rois_map = parse_roi_files(roi_file_paths)

    # Clean up the temporary directory
    temp_dir.cleanup()

    return rois_map


def download_roi_files(series, token):
    temp_dir = TemporaryDirectory()

    urls = [
        {
            "url": s[dicomFields.RETRIEVE_URL][dicomFields.VALUE][0],
            "token": token,
            "dir": temp_dir.name,
        }
        for s in series
    ]

    tic()

    # Download & extract ZIP files to temp directory
    ThreadPool(8).map(download_roi_file, urls)

    file_paths = glob.glob(f"{temp_dir.name}/**/*", recursive=True)

    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]

    elapsed = toc()
    print("Downloading ROI files took", elapsed)

    return (file_paths, temp_dir)


def download_roi_file(url_map):
    access_token = get_token_header(url_map["token"])

    response = requests.get(
        f'{url_map["url"]}?accept=application/zip', headers=access_token
    )

    z = ZipFile(io.BytesIO(response.content))
    z.extractall(url_map["dir"])

    return


NODEJS_SCRIPT_PATH = "/usr/src/app/bin/parse-dicom.js"


def parse_roi_files(files):

    try:
        tic()
        # Parse DICOM files using node.js script (faster than pydicom)
        response = muterun_js(NODEJS_SCRIPT_PATH, ",".join(files))
        all_rois = response.stdout.decode().strip().split("\t")
        elapsed = toc()
        print("Parsing dicom files took", elapsed)

        rois_map = {r: all_rois.count(r) for r in all_rois}

        return rois_map
    except Exception as err:
        raise err

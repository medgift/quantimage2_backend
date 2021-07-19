import io
import os.path
from zipfile import ZipFile

import requests

import glob

from flask import Blueprint, jsonify, request, g, current_app, Response
from ttictoc import tic, toc

from imaginebackend_common.kheops_utils import dicomFields, get_token_header
from imaginebackend_common.models import Album

from multiprocessing.pool import ThreadPool

from tempfile import TemporaryDirectory

# Define blueprint
from routes.utils import validate_decorate
from service.feature_extraction import (
    get_studies_from_album,
    get_series_from_study,
    get_instances_from_series,
    get_instance_metadata_from_instance,
)

bp = Blueprint(__name__, "albums")


@bp.before_request
def before_request():
    validate_decorate(request)


@bp.route("/albums/<album_id>", methods=("GET", "PATCH"))
def album_rois(album_id):
    if request.method == "PATCH":
        return save_rois(album_id, None)

    if request.method == "GET":
        return get_rois(album_id)


@bp.route("/albums/<album_id>/current-outcome", defaults={"labelcategory_id": None})
@bp.route(
    "/albums/<album_id>/current-outcome/<labelcategory_id>", methods=("GET", "PATCH")
)
def album_labelcategory(album_id, labelcategory_id=None):

    if request.method == "PATCH":
        return save_current_outcome(album_id, labelcategory_id)

    if request.method == "GET":
        return get_current_outcome(album_id)


@bp.route("/albums/<album_id>/force")
def album_rois_force(album_id):
    return get_rois(album_id, force=True)


def get_current_outcome(album_id):
    album = Album.find_by_album_id(album_id)

    return f"{album.current_outcome_id}"


def save_current_outcome(album_id, labelcategory_id):
    album = Album.save_current_outcome(album_id, labelcategory_id)
    return jsonify(album.to_dict())


def save_rois(album_id, rois):
    album = Album.save_rois(album_id, rois)
    return jsonify(album.to_dict())


def get_rois(album_id, force=False):
    album = Album.find_by_album_id(album_id)

    if album.rois is None or force == True:
        rois_map = get_rois_from_kheops(album_id)
        Album.save_rois(album_id, rois_map)

    return album.rois


def get_rois_from_kheops(album_id):
    token = g.token

    studies = get_studies_from_album(album_id, token)

    studies_dicts = [
        {
            "album_id": album_id,
            "token": token,
            "study": study,
        }
        for study in studies
    ]

    tic()
    # Get list of ROI series to examine
    roi_instances = ThreadPool(8).map(get_roi_metadata, studies_dicts)

    # Filter out None values from studies without ROIs
    roi_instances = [r for r in roi_instances if r]

    elapsed = toc()
    print("Getting all the ROIs metadata took", elapsed)

    # Create map of ROI -> Number of studies from ROI files
    rois_map = parse_roi_instances(roi_instances)

    return rois_map


def get_roi_metadata(study_dict):
    study = study_dict["study"]
    album_id = study_dict["album_id"]
    token = study_dict["token"]

    # TODO - This might be more dynamic, not hard-coded to RTSTRUCT & SEG
    series = get_series_from_study(
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
        ["RTSTRUCT", "SEG"],
        album_id,
        token,
    )

    if len(series) == 0:
        return None

    # Get list of instances
    instances = get_instances_from_series(
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
        series[0][dicomFields.SERIES_UID][dicomFields.VALUE][0],
        token,
    )

    # Get instance metadata
    instance_metadata = get_instance_metadata_from_instance(
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
        series[0][dicomFields.SERIES_UID][dicomFields.VALUE][0],
        instances[0][dicomFields.INSTANCE_UID][dicomFields.VALUE][0],
        token,
    )

    return instance_metadata[0]


def parse_roi_instances(instances):
    all_rois = []

    for instance in instances:
        modality = instance[dicomFields.MODALITY][dicomFields.VALUE][0]
        if modality == "RTSTRUCT":
            for roi in instance[dicomFields.STRUCTURE_SET_ROI_SEQUENCE][
                dicomFields.VALUE
            ]:
                all_rois.append(roi[dicomFields.ROI_NAME][dicomFields.VALUE][0])
        elif modality == "SEG":
            for roi in instance[dicomFields.SEGMENT_SEQUENCE][dicomFields.VALUE]:
                all_rois.append(
                    roi[dicomFields.SEGMENT_DESCRIPTION][dicomFields.VALUE][0]
                )
        else:
            raise ValueError("Unsupported ROI modality")

    rois_map = {r: all_rois.count(r) for r in all_rois}

    return rois_map

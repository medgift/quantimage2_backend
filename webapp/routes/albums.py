from itertools import chain
from collections import Counter

from flask import Blueprint, jsonify, request, g, Response
from ttictoc import tic, toc

from quantimage2_backend_common.kheops_utils import dicomFields
from quantimage2_backend_common.models import Album, LabelCategory, AlbumOutcome

from multiprocessing.pool import ThreadPool

# Define blueprint
from routes.utils import validate_decorate
from service.feature_extraction import (
    get_studies_from_album,
    get_series_from_study,
    get_series_metadata,
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


@bp.route("/albums/<album_id>/force")
def album_rois_force(album_id):
    if request.method == "GET":
        return get_rois(album_id, True)


@bp.route(
    "/albums/<album_id>/current-outcome",
    defaults={"labelcategory_id": None},
    methods=("GET", "PATCH"),
)
@bp.route(
    "/albums/<album_id>/current-outcome/<labelcategory_id>", methods=("GET", "PATCH")
)
def album_labelcategory(album_id, labelcategory_id=None):
    if request.method == "PATCH":
        return save_current_outcome(album_id, g.user, labelcategory_id)

    if request.method == "GET":
        return get_current_outcome(album_id, g.user)


def get_current_outcome(album_id, user_id):
    album_outcome = AlbumOutcome.find_by_album_user_id(album_id, user_id)

    if album_outcome:
        current_outcome = LabelCategory.find_by_id(album_outcome.outcome_id)
        return jsonify(current_outcome.to_dict())
    else:
        return Response(status=200, mimetype="application/json", response="null")


def save_current_outcome(album_id, user_id, labelcategory_id):
    album = AlbumOutcome.save_current_outcome(album_id, user_id, labelcategory_id)
    return jsonify(album.to_dict())


def save_rois(album_id, rois):
    album = Album.save_rois(album_id, rois)
    return jsonify(album.to_dict())


def get_rois(album_id, forced=False):
    token = g.token

    album = Album.find_by_album_id(album_id)

    studies = get_studies_from_album(album_id, token)

    study_uids = [
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0] for study in studies
    ]

    # Check if studies have changed, if yes then force fetching the ROIs again
    if album.studies is None or sorted(album.studies) != sorted(study_uids):
        forced = True
        album.studies = study_uids
        album.save_to_db()

    if album.rois is None or forced is True:
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
    roi_sets = ThreadPool(8).map(get_study_rois, studies_dicts)

    # Filter out None values from studies without ROIs
    elapsed = toc()
    print("Getting all the ROIs metadata took", elapsed)

    # Create map of ROI -> Number of studies from study ROI sets
    rois_map = dict(
        Counter(chain(*[roi_set for roi_set in roi_sets if roi_set is not None]))
    )

    return rois_map


def get_study_rois(study_dict):
    study = study_dict["study"]
    album_id = study_dict["album_id"]
    token = study_dict["token"]

    # TODO - This might be more dynamic, not hard-coded to RTSTRUCT & SEG
    roi_series = get_series_from_study(
        study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
        ["RTSTRUCT", "SEG"],
        album_id,
        token,
    )

    if len(roi_series) == 0:
        return None

    study_rois = set()

    for roi_serie in roi_series:

        try:
            # Get Series metadata
            series_metadata = get_series_metadata(
                study[dicomFields.STUDY_UID][dicomFields.VALUE][0],
                roi_serie[dicomFields.SERIES_UID][dicomFields.VALUE][0],
                album_id,
                token,
            )

            # Add the ROI name to the study ROIs set
            study_rois = study_rois.union(get_roi_names(series_metadata[0]))
        except Exception as e:
            print(
                f"Error obtaining ROI names for patient ID {study[dicomFields.PATIENT_ID][dicomFields.VALUE][0]}"
            )
            print(e)

    return study_rois


def get_roi_names(instance):
    roi_names = set()

    try:
        modality = instance[dicomFields.MODALITY][dicomFields.VALUE][0]
        if modality == "RTSTRUCT":
            for roi in instance[dicomFields.STRUCTURE_SET_ROI_SEQUENCE][
                dicomFields.VALUE
            ]:
                roi_names.add(roi[dicomFields.ROI_NAME][dicomFields.VALUE][0])
        elif modality == "SEG":
            for roi in instance[dicomFields.SEGMENT_SEQUENCE][dicomFields.VALUE]:
                roi_names.add(
                    roi[dicomFields.SEGMENT_DESCRIPTION][dicomFields.VALUE][0]
                )
        else:
            raise ValueError("Unsupported ROI modality")
    except Exception as e:
        print(
            f"Error obtaining ROI names for patient ID {instance[dicomFields.PATIENT_ID][dicomFields.VALUE][0]}"
        )
        print(e)

    return roi_names

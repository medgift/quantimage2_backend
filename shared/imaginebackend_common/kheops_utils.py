import os


def get_token_header(token):
    return {"Authorization": "Bearer " + token}


# Backend client
kheopsBaseURL = os.environ["KHEOPS_BASE_URL"]

kheopsBaseEndpoint = kheopsBaseURL + "/api"


class KheopsEndpoints(object):
    pass


endpoints = KheopsEndpoints()
endpoints.studies = kheopsBaseEndpoint + "/studies"
endpoints.albums = kheopsBaseEndpoint + "/albums"
endpoints.album_parameter = "album"
endpoints.seriesSuffix = "/series"
endpoints.instancesSuffix = "/instances"
endpoints.studyMetadataSuffix = "metadata"


class DicomFields(object):
    pass


dicomFields = DicomFields()
dicomFields.STUDY_UID = "0020000D"
dicomFields.STUDY_DATE = "00080020"
dicomFields.SERIES_UID = "0020000E"
dicomFields.INSTANCE_UID = "00080018"
dicomFields.DATE = "00080020"
dicomFields.PATIENT_ID = "00100020"
dicomFields.PATIENT_NAME = "00100010"
dicomFields.MODALITY = "00080060"
dicomFields.MODALITIES = "00080061"
dicomFields.DICOM_FILE_URL = "7FE00010"
dicomFields.RETRIEVE_URL = "00081190"
dicomFields.VALUE = "Value"
dicomFields.BULK_DATA_URI = "BulkDataURI"
dicomFields.ALPHABETIC = "Alphabetic"
dicomFields.DATE_FORMAT = "YYYYMMDD"

# RTSTRUCT
dicomFields.STRUCTURE_SET_ROI_SEQUENCE = "30060020"
dicomFields.ROI_NAME = "30060026"

# SEG
dicomFields.SEGMENT_SEQUENCE = "00620002"
dicomFields.SEGMENT_DESCRIPTION = "00620006"

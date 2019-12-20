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
endpoints.album_parameter = "album"
endpoints.seriesSuffix = "/series"
endpoints.instancesSuffix = "/instances"
endpoints.studyMetadataSuffix = "metadata"


class DicomFields(object):
    pass


dicomFields = DicomFields()
dicomFields.STUDY_UID = "0020000D"
dicomFields.SERIES_UID = "0020000E"
dicomFields.INSTANCE_UID = "00080018"
dicomFields.DATE = "00080020"
dicomFields.PATIENT_NAME = "00100010"
dicomFields.MODALITY = "00080060"
dicomFields.MODALITIES = "00080061"
dicomFields.DICOM_FILE_URL = "7FE00010"
dicomFields.VALUE = "Value"
dicomFields.BULK_DATA_URI = "BulkDataURI"
dicomFields.ALPHABETIC = "Alphabetic"
dicomFields.DATE_FORMAT = "YYYYMMDD"

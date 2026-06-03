import re
from enum import Enum


class MODEL_TYPES(Enum):
    CLASSIFICATION = "Classification"
    SURVIVAL = "Survival"


class DATA_SPLITTING_TYPES(Enum):
    FULLDATASET = "fulldataset"
    TRAINTESTSPLIT = "traintest"


class TRAIN_TEST_SPLIT_TYPES(Enum):
    AUTO = "automatic"
    MANUAL = "manual"


class ESTIMATOR_STEP(Enum):
    CLASSIFICATION = "classifier"
    SURVIVAL = "analyzer"


class TRAINING_PHASES(Enum):
    TRAINING = "training"
    TESTING = "testing"
    SAVING = "saving"


RIESZ_FEATURE_PREFIXES = ["tex"]
# We no longer compute ZRAD features, but the prefix is kept so that
# featureIDMatcher can still parse ZRAD feature IDs that already exist in the DB.
ZRAD_FEATURE_PREFIXES = ["zrad"]
PYRADIOMICS_FEATURE_PREFIXES = [
    "original",
    "log",
    "wavelet",
    "gradient",
    "square",
    "squareroot",
    "exponential",
    "logarithm",
]

FEATURE_ID_SEPARATOR = "‑"  # This is a non-breaking hyphen, to differentiate with regular hyphens that can occur in ROI names

# Separator for namespacing a clinical feature by its source file:
# "<clinical_feature_file_id>::<name>". Lets the same column name uploaded in
# two different CSVs map to two distinct columns. Distinct from the radiomics
# FEATURE_ID_SEPARATOR above (a non-breaking hyphen).
CLINICAL_FEATURE_ID_SEPARATOR = "::"

prefixes = RIESZ_FEATURE_PREFIXES + ZRAD_FEATURE_PREFIXES + PYRADIOMICS_FEATURE_PREFIXES
featureIDMatcher = re.compile(
    rf"(?P<modality>.*?){FEATURE_ID_SEPARATOR}(?P<roi>.*?){FEATURE_ID_SEPARATOR}(?P<feature>(?:{'|'.join(prefixes)}).*)"
)

FAKE_SCORER_KEY = "fake"

QUEUE_EXTRACTION = "extraction"
QUEUE_TRAINING = "training"

PET_MODALITY = "PT"

FIRSTORDER_PYRADIOMICS_PREFIX = "firstorder"
FIRSTORDER_REPLACEMENT_SUV = "SUV"
FIRSTORDER_REPLACEMENT_INTENSITY = "intensity"

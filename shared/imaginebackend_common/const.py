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


RIESZ_FEATURE_PREFIXES = ["tex"]
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
PET_SPECIFIC_PREFIXES = ["PET"]

prefixes = (
    RIESZ_FEATURE_PREFIXES
    + ZRAD_FEATURE_PREFIXES
    + PYRADIOMICS_FEATURE_PREFIXES
    + PET_SPECIFIC_PREFIXES
)
featureIDMatcher = re.compile(
    rf"(?P<modality>.*?)-(?P<roi>.*?)-(?P<feature>(?:{'|'.join(prefixes)}).*)"
)

FAKE_SCORER_KEY = "fake"

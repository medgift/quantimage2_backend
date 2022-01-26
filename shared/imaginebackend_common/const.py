import re
from enum import Enum


class MODEL_TYPES(Enum):
    CLASSIFICATION = "Classification"
    SURVIVAL = "Survival"


class VALIDATION_TYPES(Enum):
    CROSSVALIDATION = "crossvalidation"
    TRAINTESTSPLIT = "traintest"


RIESZ_FEATURE_PREFIXES = ["tex"]
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

prefixes = RIESZ_FEATURE_PREFIXES + PYRADIOMICS_FEATURE_PREFIXES + PET_SPECIFIC_PREFIXES
featureIDMatcher = re.compile(
    rf"(?P<modality>.*?)-(?P<roi>.*?)-(?P<feature>(?:{'|'.join(prefixes)}).*)"
)

import json

import os
import subprocess
from okapy.featureextractor import featureextractor as okapy_featureextractor
from radiomics import featureextractor
from subprocess import PIPE
from celery.contrib import rdb


class OkapyFeatureExtractor:
    def __init__(self, config):
        self.config = config

    def extract(self, images_path, labels_path, modality):
        extractor = okapy_featureextractor.create_extractor(modality, self.config)
        result = extractor.execute(images_path, labels_path)
        return result


class PyRadiomicsFeatureExtractor:
    def __init__(self, config):
        self.config = config

    def extract(self, images_path, labels_path):
        extractor = featureextractor.RadiomicsFeatureExtractor(self.config)
        result = extractor.execute(images_path, labels_path)
        return result


class RieszFeatureExtractor:
    def __init__(self, config):
        self.config = config

    def extract(self, images_path, labels_path):

        if os.environ["PYTHON_ENV"] == "development":
            rdb.set_trace()

        completed_matlab_process = subprocess.run(
            ["bin/RieszExtractor", images_path, labels_path, json.dumps(self.config)],
            stdout=PIPE,
            encoding="utf-8",
        )
        print("MATLAB STDOUT!!!!!!!!!!!!!!")
        output = completed_matlab_process.stdout
        output_lines = output.splitlines()
        result = json.loads(output_lines[-1])

        return result

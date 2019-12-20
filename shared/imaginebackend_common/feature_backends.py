import os
import pathlib
import shutil
import tempfile

import yaml
from abc import abstractmethod


class FeatureBackend:
    def __init__(self, config):
        self.config = config

    features_key = "features"
    parameters_key = "parameters"

    @abstractmethod
    def pre_process_data(self, dicom_dir):
        pass

    @abstractmethod
    def extract_features(self, processed_data):
        pass

    @abstractmethod
    def parse_config(self):
        pass

    @abstractmethod
    def format_config(self):
        pass


class PyRadiomicsFeatureBackend(FeatureBackend):
    def pre_process_data(self, dicom_dir):
        # Use Valentin's DicomWalker to convert DICOM to NRRD
        results_dir = self.convert_dicom_to_nrrd(dicom_dir)

        # Go through results files and extract features
        file_paths = [
            str(filepath.absolute())
            for filepath in pathlib.Path(results_dir).glob("**/*")
        ]

        return file_paths

    def extract_features(self, processed_data):
        from .feature_extractors import PyRadiomicsFeatureExtractor

        ct_path = None
        labels_path = None
        for file_path in processed_data:
            if not ct_path and "ct" in file_path:
                ct_path = file_path

            if not labels_path and "rtstruct" in file_path:
                labels_path = file_path

        parsed_config = self.parse_config()
        extractor = PyRadiomicsFeatureExtractor(parsed_config)
        result = extractor.extract(ct_path, labels_path)
        return result

    def format_config(self):
        normalized_config = {
            self.features_key: list(self.config["featureClass"].keys()),
            self.parameters_key: self.config["setting"]
            if "setting" in self.config.keys()
            else None,
        }

        return normalized_config

    def parse_config(self):

        parsed_features = {}
        for featureName in self.config[self.features_key]:
            parsed_features[featureName] = None

        parsed_config = {
            "featureClass": parsed_features,
            "setting": self.config[self.parameters_key]
            if self.config[self.parameters_key] is not None
            else {},
        }

        return parsed_config

    def convert_dicom_to_nrrd(self, input_dir, labels=["GTV T"]):
        from okapy.dicomconverter.dicom_walker import DicomWalker

        output_dir = tempfile.mkdtemp()

        walker = DicomWalker(input_dir, output_dir, list_labels=labels)
        walker.walk()
        walker.fill_images()  # walker.fill_dicom_files()
        walker.convert()

        return output_dir


class QuantImageFeatureBackend(FeatureBackend):
    def pre_process_data(self, dicom_dir):
        # ZIP the folder and return the path to the ZIP file
        zip_path = os.path.join(tempfile.mkdtemp(), "input")
        zip_file_path = shutil.make_archive(zip_path, "zip", dicom_dir)

        return zip_file_path

    def extract_features(self, processed_data):
        from .feature_extractors import QuantImageFeatureExtractor

        zip_path = processed_data

        parsed_config = self.parse_config(self.config)
        extractor = QuantImageFeatureExtractor(parsed_config)
        result = extractor.extract(zip_path)
        return result

    def format_config(self):
        normalized_config = {
            self.features_key: list(self.config["visibleFields"].keys()),
            self.parameters_key: self.config["params"]
            if "params" in self.config.keys()
            else None,
        }

        return normalized_config

    def parse_config(self, config):

        parsed_features = {}
        for featureName in self.config[self.features_key]:
            parsed_features[featureName] = True

        parsed_config = {
            "visibleFields": parsed_features,
            "params": config[self.parameters_key],
        }

        return parsed_config


class FeatureConfig:
    def __init__(self, features, parameters):
        self.features = features
        self.parameters = parameters


class Feature:
    def __init__(self, name):
        self.name = name


class FeatureParameter:
    def __init__(self, param_name, param_value):
        self.param_name = param_name
        self.param_value = param_value


feature_backends_map = {
    "pyradiomics": PyRadiomicsFeatureBackend,
    "quantimage": QuantImageFeatureBackend,
}

import json
import pathlib
import tempfile
import re

from abc import abstractmethod

features_key = "features"
parameters_key = "parameters"


class FeatureBackend:
    def __init__(self, config):
        self.config = config

    @classmethod
    def pre_process_data(cls, dicom_dir):
        # Use Valentin's DicomWalker to convert DICOM to NII
        # as well as resample the images (based on modality)
        results_dir = cls.convert_dicom_to_nii(dicom_dir)

        # Go through results files and extract features
        file_paths = [
            str(filepath.absolute())
            for filepath in pathlib.Path(results_dir).glob("**/*")
        ]

        return [file_paths, results_dir]

    @classmethod
    def convert_dicom_to_nii(cls, input_dir, labels=None):
        from okapy.dicomconverter.dicom_walker import DicomWalker

        output_dir = tempfile.mkdtemp()

        walker = DicomWalker(
            input_dir, output_dir, list_labels=labels, extension_output="nii"
        )
        walker.walk()
        walker.fill_dicom_files()
        walker.convert()

        return output_dir

    @classmethod
    def get_input_files(cls, files):
        image_matcher = re.compile(
            r"(?P<patient_id>.*)__(?P<modality>.*)__resampled\.(?P<extension>.*)"
        )
        roi_matcher = re.compile(
            r"(?P<patient_id>.*)__from_(?P<source_modality>.*)_mask__(?P<label>.*)__resampled_for__(?P<modality>.*)\.(?P<extension>.*)"
        )
        roi_matcher_simple = re.compile(
            r"(?P<label>.*)__resampled_for__(?P<modality>.*)\.(?P<extension>.*)"
        )

        input_files = {}

        for file_path in files:
            file_name = pathlib.PurePath(file_path).name

            # Check if it's an image or a mask
            image_matches = re.match(image_matcher, file_name)

            if image_matches:
                print("image detected : " + file_path)

                image_match_group = image_matches.groupdict()
                modality = image_match_group["modality"]

                # Create entry for the modality in the output map if necessary
                if modality not in input_files:
                    input_files[modality] = {}

                # Define image for this modality
                input_files[modality]["image"] = file_path
            else:
                roi_matches = re.match(roi_matcher, file_name)

                # If not matches are found, try with the simpler matcher alternative
                if not roi_matches:
                    roi_matches = re.match(roi_matcher_simple, file_name)

                if roi_matches:
                    print("roi detected : " + file_path)

                    roi_match_group = roi_matches.groupdict()
                    modality = roi_match_group["modality"]

                    # Create entry for the modality in the output map if necessary
                    if modality not in input_files:
                        input_files[modality] = {}

                    # Create labels map in the output map if necessary
                    if "labels" not in input_files[modality]:
                        input_files[modality]["labels"] = {}

                    # Add label to the map
                    input_files[modality]["labels"][
                        roi_match_group["label"]
                    ] = file_path
                else:
                    print(f"Unrelated file detected: {file_path}")

        print("FILES TO PROCESS : " + json.dumps(input_files))

        return input_files

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
    def extract_features(self, image_path, mask_path):
        from .feature_extractors import PyRadiomicsFeatureExtractor

        parsed_config = self.parse_config()
        extractor = PyRadiomicsFeatureExtractor(parsed_config)

        print(
            f"Extracting PyRadiomics Features with image {image_path} and labels {mask_path}"
        )

        result = extractor.extract(image_path, mask_path)
        return result

    def format_config(self):
        normalized_config = {
            features_key: list(self.config["featureClass"].keys()),
            parameters_key: self.config["setting"]
            if "setting" in self.config.keys()
            else None,
        }

        return normalized_config

    def parse_config(self):

        parsed_features = {}
        for featureName in self.config[features_key]:
            parsed_features[featureName] = None

        parsed_config = {
            "featureClass": parsed_features,
            "setting": self.config[parameters_key]
            if self.config[parameters_key] is not None
            else {},
        }

        return parsed_config


class RieszFeatureBackend(FeatureBackend):
    def extract_features(self, image_path, mask_path):
        from .feature_extractors import RieszFeatureExtractor

        parsed_config = self.parse_config()
        extractor = RieszFeatureExtractor(parsed_config)

        print(
            f"Extracting Riesz Features with image {image_path} and labels {mask_path}"
        )

        result = extractor.extract(image_path, mask_path)
        return result

    def format_config(self):
        normalized_config = {
            features_key: ["riesz3d"],
            parameters_key: self.config,
        }

        return normalized_config

    def parse_config(self):
        parsed_config = self.config[parameters_key]

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
    "riesz": RieszFeatureBackend,
}

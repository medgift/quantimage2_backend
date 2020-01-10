import pathlib
import tempfile

from abc import abstractmethod

features_key = "features"
parameters_key = "parameters"


class FeatureBackend:
    def __init__(self, config):
        self.config = config

    @classmethod
    def pre_process_data(self, dicom_dir):
        # Use Valentin's DicomWalker to convert DICOM to NII
        # as well as resample the images (based on modality)
        results_dir = self.convert_dicom_to_nii(self, dicom_dir)

        # Go through results files and extract features
        file_paths = [
            str(filepath.absolute())
            for filepath in pathlib.Path(results_dir).glob("**/*")
        ]

        return file_paths

    def get_input_files(cls, files):
        ct_path = None
        labels_path = None

        # TODO - Stop hard-coding to CT+RTSTRUCT, this should be dynamic!
        for file_path in files:
            if not ct_path and file_path.endswith("CT__resampled.nii"):
                ct_path = file_path

            if not labels_path and "mask" in file_path and file_path.endswith("CT.nii"):
                labels_path = file_path

        return {"image_path": ct_path, "labels_path": labels_path}

    @abstractmethod
    def extract_features(self, processed_data):
        pass

    @abstractmethod
    def parse_config(self):
        pass

    @abstractmethod
    def format_config(self):
        pass

    def convert_dicom_to_nii(self, input_dir, labels=["GTV T"]):
        from okapy.dicomconverter.dicom_walker import DicomWalker

        output_dir = tempfile.mkdtemp()

        walker = DicomWalker(
            input_dir, output_dir, list_labels=labels, extension_output="nii"
        )
        walker.walk()
        walker.fill_dicom_files()
        walker.convert()

        return output_dir


class PyRadiomicsFeatureBackend(FeatureBackend):
    def extract_features(self, processed_data):
        from .feature_extractors import PyRadiomicsFeatureExtractor

        input_files = self.get_input_files(processed_data)

        parsed_config = self.parse_config()
        extractor = PyRadiomicsFeatureExtractor(parsed_config)

        print(
            f"Extracting PyRadiomics Features with image {input_files['image_path']} and labels {input_files['labels_path']}"
        )

        result = extractor.extract(
            input_files["image_path"], input_files["labels_path"]
        )
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
    def extract_features(self, processed_data):
        from .feature_extractors import RieszFeatureExtractor

        input_files = self.get_input_files(processed_data)

        parsed_config = self.parse_config()
        extractor = RieszFeatureExtractor(parsed_config)

        print(
            f"Extracting Riesz Features with image {input_files['image_path']} and labels {input_files['labels_path']}"
        )

        result = extractor.extract(
            input_files["image_path"], input_files["labels_path"]
        )
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

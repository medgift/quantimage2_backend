import abc

from flask import current_app, g
from sklearn.model_selection import GridSearchCV, ParameterGrid
from ttictoc import tic, toc

from imaginebackend_common.const import DATA_SPLITTING_TYPES
from modeling.utils import (
    split_dataset,
    preprocess_features,
    preprocess_labels,
    generate_normalization_methods,
)


class Modeling:
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        *,
        feature_extraction_id,
        collection_id,
        album,
        feature_selection,
        feature_names,
        estimator_step,
        label_category,
        features_df,
        labels_df,
        data_splitting_type,
        train_test_splitting_type,
        training_patients,
        test_patients,
        random_seed,
        refit_metric,
        n_jobs=1,
        training_id,
    ):

        # Album ID & Other Metadata
        self.album = album
        self.feature_extraction_id = feature_extraction_id
        self.collection_id = collection_id
        self.feature_selection = feature_selection
        self.feature_names = feature_names
        self.label_category = label_category

        # Type of data splitting (train/test or full dataset)
        self.data_splitting_type = data_splitting_type
        self.train_test_splitting_type = train_test_splitting_type

        # Random seed (for reproducing results)
        self.random_seed = random_seed

        # Refit metric (for the grid search)
        self.refit_metric = refit_metric

        # Generate normalization options (for the grid search)
        self.preprocessor = {"preprocessor": generate_normalization_methods()}

        # Filter out unlabelled patients
        labelled_patients = list(labels_df.index)
        training_patients_filtered = [
            p for p in training_patients if p in labelled_patients
        ]
        test_patients_filtered = (
            [p for p in test_patients if p in labelled_patients]
            if test_patients
            else None
        )

        # Save filtered patients
        self.training_patients = training_patients_filtered
        self.test_patients = test_patients_filtered

        # Preprocess features & labels
        features_df = preprocess_features(features_df)
        labels_df = preprocess_labels(
            labels_df, training_patients_filtered, test_patients_filtered
        )

        self.feature_names = list(features_df.columns)

        if self.is_train_test():
            # Split training & test set based on provided Patient IDs
            self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(
                features_df,
                labels_df,
                training_patients_filtered,
                test_patients_filtered,
            )
        else:
            self.X_train = features_df
            self.y_train = labels_df

        # Keyword arguments

        # Number of parallel jobs
        self.n_jobs = n_jobs

        # Training ID (for progress reporting)
        self.training_id = training_id

        # Estimator Step (for scoring calculation)
        self.estimator_step = estimator_step

    def is_train_test(self):
        return (
            DATA_SPLITTING_TYPES(self.data_splitting_type)
            == DATA_SPLITTING_TYPES.TRAINTESTSPLIT
        )

    def create_model(self):

        # Encode Labels (if applicable)
        # TODO - Check if there is a better method to see if this was implemented by the subclass
        if hasattr(self, "encode_labels"):
            y_train_encoded = self.encode_labels(self.y_train)
            y_test_encoded = []

            if self.is_train_test():
                y_test_encoded = self.encode_labels(self.y_test)
        else:
            y_train_encoded, y_test_encoded = self.y_train, self.y_test

        pipeline = self.get_pipeline()
        parameter_grid = self.get_parameter_grid()
        cv = self.get_cv()
        scoring = self.get_scoring()

        current_app.my_celery.send_task(
            "imaginetasks.train",
            kwargs={
                "feature_extraction_id": self.feature_extraction_id,
                "collection_id": self.collection_id,
                "album": self.album,
                "feature_selection": self.feature_selection,
                "feature_names": self.feature_names,
                "pipeline": pipeline,
                "parameter_grid": parameter_grid,
                "estimator_step": self.estimator_step,
                "scoring": scoring,
                "refit_metric": self.refit_metric,
                "cv": cv,
                "n_jobs": self.n_jobs,
                "X_train": self.X_train,
                "X_test": self.X_test if self.is_train_test() else None,
                "label_category": self.label_category,
                "data_splitting_type": self.data_splitting_type,
                "train_test_splitting_type": self.train_test_splitting_type,
                "training_patients": self.training_patients,
                "test_patients": self.test_patients,
                "y_train_encoded": y_train_encoded,
                "y_test_encoded": y_test_encoded if self.is_train_test() else None,
                "is_train_test": self.is_train_test(),
                "random_seed": self.random_seed,
                "training_id": self.training_id,
                "user_id": g.user,
            },
            serializer="pickle",
        )

        pg = ParameterGrid(parameter_grid)
        n_steps = len(pg) * cv.get_n_splits()

        return n_steps

    @abc.abstractmethod
    def encode_labels(self, labels, classes=None):
        """Function to encode labels"""

    @abc.abstractmethod
    def get_cv(self, n_splits=None, n_repeats=None):
        """Create Cross-Validation object"""
        return

    @abc.abstractmethod
    def get_scoring(self):
        """Create scoring parameters"""
        return

    @abc.abstractmethod
    def get_pipeline(self):
        """Generate pipeline of steps to follow"""
        return

    @abc.abstractmethod
    def get_parameter_grid(self):
        """Generate grid of parameters to explore"""
        return

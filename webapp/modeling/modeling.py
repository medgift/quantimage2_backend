import abc

from sklearn.model_selection import GridSearchCV
from ttictoc import tic, toc

from imaginebackend_common.const import DATA_SPLITTING_TYPES
from modeling.utils import (
    split_dataset,
    run_bootstrap,
    calculate_test_metrics,
    calculate_training_metrics,
    preprocess_features,
    preprocess_labels,
    generate_normalization_methods,
)


class Modeling:
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        features_df,
        labels_df,
        data_splitting_type,
        training_patients,
        test_patients,
        random_seed,
        refit_metric,
        n_jobs=1,
    ):
        # Type of data splitting (train/test or full dataset)
        self.data_splitting_type = data_splitting_type

        # Random seed (for reproducing results)
        self.random_seed = random_seed

        # Refit metric (for the grid search)
        self.refit_metric = refit_metric

        # Generate normalization options (for the grid search)
        self.preprocessor = {"preprocessor": generate_normalization_methods()}

        # Preprocess features & labels
        features_df = preprocess_features(features_df)
        labels_df = preprocess_labels(labels_df, training_patients, test_patients)

        self.feature_names = list(features_df.columns)

        if self.is_train_test():
            # Split training & test set based on provided Patient IDs
            self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(
                features_df, labels_df, training_patients, test_patients
            )
        else:
            self.X_train = features_df
            self.y_train = labels_df

        # Keyword arguments

        # Number of parallel jobs
        self.n_jobs = n_jobs

    def is_train_test(self):
        return (
            DATA_SPLITTING_TYPES(self.data_splitting_type)
            == DATA_SPLITTING_TYPES.TRAINTESTSPLIT
        )

    def create_model(self):

        # Encode Labels (if applicable)
        # TODO - Check if there is a better method to see if this was implemented by the subclass
        if hasattr(self, "encode_labels"):
            y_train_encoded, fitted_encoder = self.encode_labels(self.y_train)
            y_test_encoded = []

            if self.is_train_test():
                # TODO - Find a better way to handle both ScikitLearn encoders & other encoding functions
                if fitted_encoder is not None:
                    y_test_encoded = fitted_encoder.transform(self.y_test)
                else:
                    y_test_encoded, _ = self.encode_labels(self.y_test)
        else:
            y_train_encoded, y_test_encoded = self.y_train, self.y_test

        pipeline = self.get_pipeline()
        grid = self.get_grid()
        cv = self.get_cv()
        scoring = self.get_scoring()

        # Run grid search on the defined pipeline & search space
        grid = GridSearchCV(
            pipeline,
            grid,
            scoring=scoring,
            refit=self.refit_metric,
            cv=cv,
            n_jobs=self.n_jobs,
            return_train_score=False,
            verbose=2,
        )

        # Fit the model on the training set
        tic()
        fitted_model = grid.fit(self.X_train, y_train_encoded)
        elapsed = toc()

        print(f"Fitting the model took {elapsed}")

        training_metrics = calculate_training_metrics(fitted_model.cv_results_, scoring)
        test_metrics = None

        # Train/test only - Perform Bootstrap on the Test set
        if self.is_train_test():
            tic()
            scores, n_bootstrap = run_bootstrap(
                self.X_test,
                y_test_encoded,
                fitted_model,
                self.random_seed,
                scoring,
            )
            elapsed = toc()
            print(f"Running bootstrap took {elapsed}")

            test_metrics = calculate_test_metrics(scores, scoring)

        return (
            fitted_model,
            self.feature_names,
            "Repeated Stratified K-Fold Cross-Validation",
            {"k": cv.cvargs["n_splits"], "n": cv.n_repeats},
            "Bootstrap" if self.is_train_test() else None,
            {"n": n_bootstrap} if self.is_train_test() else None,
            training_metrics,
            test_metrics,
        )

    @abc.abstractmethod
    def encode_labels(self, labels):
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
    def get_grid(self):
        """Generate grid of parameters to explore"""
        return

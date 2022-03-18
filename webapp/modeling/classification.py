from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ttictoc import tic, toc

from imaginebackend_common.const import DATA_SPLITTING_TYPES
from modeling.utils import (
    split_dataset,
    get_labelencoder,
    NORMALIZATION_METHODS,
    CLASSIFICATION_METHODS,
    select_normalizer,
    select_classifier,
    get_cv,
    get_scoring,
    run_bootstrap,
    calculate_test_metrics,
    calculate_training_metrics,
    preprocess_features,
    preprocess_labels,
)


class Classification:
    def __init__(
        self,
        features_df,
        labels_df,
        data_splitting_type,
        training_patients,
        test_patients,
        random_seed,
        refit_metric="auc",
        n_jobs=-1,
    ):
        self.data_splitting_type = data_splitting_type
        self.random_seed = random_seed

        features_df = preprocess_features(features_df)
        labels_df = preprocess_labels(labels_df, training_patients, test_patients)

        if self.is_train_test():
            # Split training & test set based on provided Patient IDs
            self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(
                features_df, labels_df, training_patients, test_patients
            )
        else:
            self.X_train = features_df
            self.y_train = labels_df

        # Label Encoder
        self.encoder = get_labelencoder()
        self.encoder.fit(self.y_train)

        # Pipeline elements
        self.pipeline = Pipeline(
            [("preprocessor", "passthrough"), ("classifier", "passthrough")]
        )

        # Cross-validator
        self.cv = get_cv(self.random_seed)

        # Refit metric (for the grid search)
        self.refit_metric = refit_metric

        # Number of parallel jobs
        self.n_jobs = n_jobs

    def is_train_test(self):
        return (
            DATA_SPLITTING_TYPES(self.data_splitting_type)
            == DATA_SPLITTING_TYPES.TRAINTESTSPLIT
        )

    def classify(self):
        # Encode Labels
        y_train_encoded = self.encoder.transform(self.y_train)
        y_test_encoded = []

        if self.is_train_test():
            y_test_encoded = self.encoder.transform(self.y_test)

        # Populate search space with normalization & classification options
        preprocessor = {"preprocessor": generate_normalization_methods()}
        param_grid = generate_classification_methods(preprocessor, self.random_seed)

        # Run grid search on the defined pipeline & search space
        grid = GridSearchCV(
            self.pipeline,
            param_grid,
            scoring=get_scoring(),
            refit=self.refit_metric,
            cv=self.cv,
            n_jobs=self.n_jobs,
            return_train_score=False,
            verbose=2,
        )

        # Fit the model on the training set
        tic()
        fitted_model = grid.fit(self.X_train, y_train_encoded)
        elapsed = toc()

        print(f"Fitting the model took {elapsed}")

        # Train/test only - Perform Bootstrap on the Test set
        if self.is_train_test():
            tic()
            scores, n_bootstrap = run_bootstrap(
                self.X_test, y_test_encoded, fitted_model, self.random_seed
            )
            elapsed = toc()
            print(f"Running bootstrap took {elapsed}")
            metrics = calculate_test_metrics(scores)
        else:
            print(fitted_model)
            metrics = calculate_training_metrics(fitted_model.cv_results_)

        return (
            fitted_model,
            "Repeated Stratified K-Fold Cross-Validation",
            {"k": self.cv.get_n_splits(), "n": self.cv.n_repeats},
            "Bootstrap" if self.is_train_test() else None,
            {"n": n_bootstrap} if self.is_train_test() else None,
            metrics,
        )


def generate_normalization_methods():
    methods = []
    for normalization_method in NORMALIZATION_METHODS:
        methods.append(select_normalizer(normalization_method))

    methods.append("passthrough")

    return methods


def generate_classification_methods(preprocessor, random_seed):
    methods = []
    for classification_method in CLASSIFICATION_METHODS:
        methods.append(
            {**preprocessor, **select_classifier(classification_method, random_seed)}
        )

    return methods

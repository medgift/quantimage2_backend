import numpy as np
from scipy.stats import t
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Lists of supported normalization & classification methods
NORMALIZATION_METHODS = ["standardization", "minmax"]


def preprocess_features(features, training_patients, test_patients):
    # if training_patients and test_patients:
    #     features = features.loc[training_patients + test_patients]
    # else:
    #     features = features.loc[training_patients]
    features = features.drop("PatientID", axis=1)
    features.sort_index(inplace=True)
    return features


def preprocess_labels(labels, training_patients, test_patients):

    if training_patients and test_patients:
        labels = labels.loc[training_patients + test_patients]

    labels.sort_index(inplace=True)

    return labels


def split_dataset(X, y, training_patients, test_patients):
    X_train = X.loc[training_patients]
    X_test = X.loc[test_patients]

    y_train = y.loc[training_patients]
    y_test = y.loc[test_patients]

    return X_train, X_test, y_train, y_test


def select_normalizer(normalization_name):
    scaler = None

    if normalization_name == "standardization":
        scaler = StandardScaler()
    elif normalization_name == "minmax":
        scaler = MinMaxScaler()

    return scaler


def get_random_seed():
    return 42


def generate_normalization_methods():
    methods = []
    for normalization_method in NORMALIZATION_METHODS:
        methods.append(select_normalizer(normalization_method))

    return methods


def corrected_std(differences, n_train, n_test):
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    # p_val = t.sf(t_stat, df)
    return t_stat, p_val


def compare_score(s1, s2, n_train, n_test, rope_interval=[-0.01, 0.01]):
    n = s1.shape[0]
    t_stat, p_val = compute_corrected_ttest(s1 - s2, n - 1, n_train, n_test)
    t_post = t(
        n - 1, loc=np.mean(s1 - s2), scale=corrected_std(s1 - s2, n_train, n_test)
    )
    ri = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
    return {
        "t_stat": t_stat,
        "p_value": p_val,
        "proba M1 > M2": 1 - t_post.cdf(rope_interval[1]),
        "proba M1 == M2": ri,
        "proba M1 < M2": t_post.cdf(rope_interval[0]),
    }


def corrected_ci(s, n_train, n_test, alpha=0.95):
    n = s.shape[0]
    mean = np.mean(s)
    std = corrected_std(s, n_train, n_test)
    return t.interval(alpha, n - 1, loc=mean, scale=std)

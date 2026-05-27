import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_prepared_data():
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=42,
                               n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def generate_custom_dataset(dataset_type="linear", n_samples=500, noise=0.0):
    if dataset_type == "linear":
        n_per_class = n_samples // 2
        X0 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], n_per_class)
        y0 = np.zeros(n_per_class)
        X1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_per_class)
        y1 = np.ones(n_per_class)
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])

    elif dataset_type == "xor":
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    elif dataset_type == "circle":
        X = np.random.uniform(-2, 2, (n_samples, 2))
        radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        y = (radius > 1).astype(int)

    # Добавление шума
    n_noisy = int(noise * n_samples)
    noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
    y[noisy_idx] = 1 - y[noisy_idx]

    return X, y


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_prepared_data(dataset_type="standard"):
    if dataset_type == "standard":
        X, y = make_classification(
            n_samples=500, n_features=2, n_redundant=0,
            n_informative=2, random_state=42, n_clusters_per_class=1
        )
    elif dataset_type == "xor":
        X = np.random.randn(500, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
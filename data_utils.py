import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def get_prepared_data():
    # 1. Генерация набора данных
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

    # 2. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 3. Стандартизация (Z-score)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
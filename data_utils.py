import numpy as np
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split


def get_prepared_data(dataset_type="standard"):
    # 1. Генерация данных в зависимости от типа
    if dataset_type == "standard" or dataset_type == "linear":
        # Линейно разделимые данные (База)
        X, y = make_classification(
            n_samples=500, n_features=2, n_redundant=0,
            n_informative=2, random_state=42, n_clusters_per_class=1
        )
    elif dataset_type == "xor":
        # Нелинейный датасет XOR (Бонус)
        X = np.random.randn(500, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    elif dataset_type == "circle":
        # Нелинейный датасет Круги (Бонус)
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    else:
        raise ValueError("Неизвестный тип датасета. Выберите 'standard', 'xor' или 'circle'.")

    # 2. Разбиение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3. Z-score нормализация (очень важно для градиентного спуска!)
    # Считаем среднее и отклонение ТОЛЬКО по обучающей выборке во избежание утечки данных
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
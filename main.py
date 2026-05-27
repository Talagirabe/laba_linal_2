import numpy as np
from data_utils import get_prepared_data, accuracy
from perceptron import Perceptron


# Функция для бонусного задания: K-Fold Кросс-валидация
def run_k_fold(X, y, k=5):
    print(f"\n--- Бонус: {k}-Fold Кросс-валидация ---")
    fold_size = len(X) // k
    accuracies = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val = X[start:end]
        y_val = y[start:end]

        # Собираем обучающую выборку из оставшихся блоков
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        # Обучаем модель на текущем фолде
        model = Perceptron(n_features=2)
        model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=16)

        # Считаем точность
        acc = accuracy(y_val, model.predict(X_val))
        accuracies.append(acc)
        print(f"Фолд {i + 1}: Точность = {acc * 100:.2f}%")

    print(f"Средняя точность (CV): {np.mean(accuracies) * 100:.2f}%")


# Функция для тестирования разных датасетов
def test_dataset(dataset_name):
    X_train, X_test, y_train, y_test = get_prepared_data(dataset_type=dataset_name)
    model = Perceptron(n_features=2)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=16)
    return accuracy(y_test, model.predict(X_test))


if __name__ == "__main__":
    print("=== ЗАПУСК ЛАБОРАТОРНОЙ РАБОТЫ №1 ===")

    # 1. Базовое обучение
    X_train, X_test, y_train, y_test = get_prepared_data("standard")
    print("\n--- 1. Базовая модель ---")
    base_model = Perceptron(n_features=2)
    base_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    print(f"Точность (Train): {accuracy(y_train, base_model.predict(X_train)) * 100:.2f}%")
    print(f"Точность (Test): {accuracy(y_test, base_model.predict(X_test)) * 100:.2f}%")

    # 2. Кросс-валидация
    # Для кросс-валидации объединяем все данные обратно в один массив
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    run_k_fold(X_all, y_all, k=5)

    # 3. Тесты нелинейности (Доказательство для защиты)
    print("\n--- 2. Тесты архитектуры на 3 датасетах ---")

    acc_linear = test_dataset("standard")
    print(f"Линейный датасет (Linear): Точность = {acc_linear * 100:.1f}%")

    acc_xor = test_dataset("xor")
    print(f"Датасет XOR (Исключающее ИЛИ): Точность = {acc_xor * 100:.1f}%")

    acc_circle = test_dataset("circle")
    print(f"Датасет Окружность (Circle): Точность = {acc_circle * 100:.1f}%\n")

    print("=== ИТОГОВЫЙ ВЫВОД ===")
    print("Однослойный перцептрон способен строить только прямые линии.")
    print("Для решения нелинейных задач (XOR и Circle) абсолютно необходимо")
    print("использование многослойных сетей (скрытых слоев).")
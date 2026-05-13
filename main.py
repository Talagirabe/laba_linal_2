import os
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_prepared_data, accuracy
from perceptron import Perceptron


def run_base_training():
    # Создаем папку для графиков, если её еще нет
    os.makedirs('plots', exist_ok=True)

    X_train, X_test, y_train, y_test = get_prepared_data()

    print("--- БАЗОВОЕ ОБУЧЕНИЕ ---")
    model = Perceptron(n_features=2, init_type='small')
    train_loss, val_loss = model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    train_acc = accuracy(y_train, model.predict(X_train))
    test_acc = accuracy(y_test, model.predict(X_test))
    print(f"Точность на обучающей выборке: {train_acc:.4f}")
    print(f"Точность на тестовой выборке: {test_acc:.4f}")

    # Визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Кривая обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (BCE)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.7)
    x_values = np.array([np.min(X_test[:, 0]), np.max(X_test[:, 0])])
    y_values = -(model.w[0] * x_values + model.b) / model.w[1]
    plt.plot(x_values, y_values, color='black', label='Decision Boundary')
    plt.title('Разделяющая граница')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()

    plt.tight_layout()

    # СОХРАНЕНИЕ ГРАФИКА
    filepath = 'plots/base_training.png'
    plt.savefig(filepath, dpi=300)  # dpi=300 для высокого качества
    plt.close()  # Закрываем фигуру, чтобы освободить память
    print(f"График успешно сохранен: {filepath}")


if __name__ == "__main__":
    run_base_training()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from data_utils import get_prepared_data
from perceptron import Perceptron

os.makedirs('plots', exist_ok=True)


# 1. Функция для генерации базового двойного графика (Линия + Loss)
def plot_base_training(losses, X_test, y_test, model, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Левый график: Кривая падения ошибки
    ax1.plot(losses, color='blue', label='Train Loss')
    ax1.set_title("Кривая обучения (Loss)")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # Правый график: Разделяющая гиперплоскость
    ax2.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label="Class 0", alpha=0.6)
    ax2.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label="Class 1", alpha=0.6)

    w1, w2, b = model.w[0], model.w[1], model.b
    if abs(w2) > 1e-6:
        x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        x1 = np.linspace(x1_min, x1_max, 100)
        x2 = -(w1 * x1 + b) / w2
        ax2.plot(x1, x2, color='black', label="Граница (Decision Boundary)")

    ax2.set_xlim(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
    ax2.set_ylim(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1)
    ax2.set_title("Разделяющая гиперплоскость")
    ax2.grid(True)
    ax2.legend()

    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


# 2. Функция для отрисовки ошибочных точек для 3-х датасетов
def plot_dataset_errors(X_test, y_test, y_pred, title, filename):
    incorrect = y_test != y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[~incorrect, 0], X_test[~incorrect, 1], c="blue", alpha=0.5, label="Верно")
    plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c="red", s=100, edgecolors="black", label="Ошибка")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


def run_all_experiments():
    print("Генерация ВСЕХ графиков начата. Пожалуйста, подождите...")
    X_train, X_test, y_train, y_test = get_prepared_data("standard")

    # 0. БАЗОВОЕ ОБУЧЕНИЕ (Самый важный график)
    print("1/9: Базовый график...")
    base_model = Perceptron(n_features=2)
    base_losses = base_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    plot_base_training(base_losses, X_test, y_test, base_model, 'image_152e87.jpg')

    # 1. Скорость обучения
    print("2/9: Скорость обучения...")
    plt.figure(figsize=(10, 5))
    for lr in [0.001, 0.01, 0.1, 1.0]:
        m = Perceptron(n_features=2)
        losses = m.fit(X_train, y_train, X_test, y_test, epochs=50, lr=lr)
        plt.plot(losses, label=f'lr={lr}')
    plt.title('Learning Rate Experiment')
    plt.legend()
    plt.savefig('plots/image_1531a7.png', dpi=300)
    plt.close()

    # 2. Размер батча
    print("3/9: Размер батча...")
    plt.figure(figsize=(10, 5))
    for b in [1, 16, 64, 256]:
        m = Perceptron(n_features=2)
        losses = m.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=b)
        plt.plot(losses, label=f'batch={b}')
    plt.title('Batch Size Experiment')
    plt.legend()
    plt.savefig('plots/image_152ea3.png', dpi=300)
    plt.close()

    # 3. Инициализация весов
    print("4/9: Инициализация весов...")
    plt.figure(figsize=(10, 5))
    for init in ['zero', 'small', 'large']:
        m = Perceptron(n_features=2, init_type=init)
        losses = m.fit(X_train, y_train, X_test, y_test, epochs=50)
        plt.plot(losses, label=f'init={init}')
    plt.title('Weight Initialization')
    plt.legend()
    plt.savefig('plots/image_1531c6.png', dpi=300)
    plt.close()

    # 4. Momentum vs SGD
    print("5/9: Momentum...")
    plt.figure(figsize=(10, 5))
    sgd_model = Perceptron(n_features=2, optimizer="sgd")
    plt.plot(sgd_model.fit(X_train, y_train, X_test, y_test, epochs=100), label="SGD")
    for beta in [0.5, 0.9, 0.99]:
        mom_model = Perceptron(n_features=2, optimizer="momentum", beta=beta)
        plt.plot(mom_model.fit(X_train, y_train, X_test, y_test, epochs=100), label=f"Momentum beta={beta}")
    plt.title('SGD vs Momentum')
    plt.legend()
    plt.savefig('plots/bonus_momentum.png', dpi=300)
    plt.close()

    # 5. Hinge Loss
    print("6/9: Hinge Loss...")
    y_train_hinge = np.where(y_train == 0, -1, 1)
    y_test_hinge = np.where(y_test == 0, -1, 1)
    ce_model = Perceptron(n_features=2, loss_type="cross_entropy")
    hinge_model = Perceptron(n_features=2, loss_type="hinge")
    plt.figure(figsize=(10, 5))
    plt.plot(ce_model.fit(X_train, y_train, X_test, y_test, epochs=100), label="Cross Entropy")
    plt.plot(hinge_model.fit(X_train, y_train_hinge, X_test, y_test_hinge, epochs=100), label="Hinge Loss")
    plt.title('Hinge vs Cross Entropy')
    plt.legend()
    plt.savefig('plots/bonus_hinge.png', dpi=300)
    plt.close()

    # 6. L2 Regularization
    print("7/9: L2 Регуляризация...")
    plt.figure(figsize=(10, 5))
    for l2 in [0.0, 0.01, 1.0]:
        m = Perceptron(n_features=2, l2_lambda=l2)
        plt.plot(m.fit(X_train, y_train, X_test, y_test, epochs=100), label=f"L2={l2}")
    plt.title('L2 Regularization')
    plt.legend()
    plt.savefig('plots/bonus_l2.png', dpi=300)
    plt.close()

    # 7. ROC Curve
    print("8/9: ROC Кривая...")
    m = Perceptron(n_features=2)
    m.fit(X_train, y_train, X_test, y_test, epochs=100)
    y_probs = m.forward(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test, y_probs):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('plots/bonus_roc_curve.png', dpi=300)
    plt.close()

    # 8. Графики ошибок для ВСЕХ 3-х датасетов
    print("9/9: Графики нелинейных датасетов...")
    datasets_to_test = [
        ("standard", "Линейный датасет", "bonus_linear_errors.png"),
        ("xor", "Датасет XOR", "bonus_xor_errors.png"),
        ("circle", "Датасет Окружность", "bonus_circle_errors.png")
    ]
    for ds_name, ds_title, ds_filename in datasets_to_test:
        X_tr, X_te, y_tr, y_te = get_prepared_data(dataset_type=ds_name)
        model = Perceptron(n_features=2)
        model.fit(X_tr, y_tr, X_te, y_te, epochs=100, lr=0.1, batch_size=16)
        y_pr = model.predict(X_te)
        plot_dataset_errors(X_te, y_te, y_pr, ds_title, ds_filename)

    print("\nГОТОВО! Все графики успешно сохранены в папку 'plots/'!")


if __name__ == "__main__":
    run_all_experiments()
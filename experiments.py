import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from data_utils import get_prepared_data
from perceptron import Perceptron

os.makedirs('plots', exist_ok=True)
X_train, X_test, y_train, y_test = get_prepared_data()

def run_all_experiments():
    print("Генерация графиков начата. Пожалуйста, подождите...")

    # 1. Скорость обучения
    plt.figure(figsize=(10, 5))
    for lr in [0.001, 0.01, 0.5, 1.0]:
        m = Perceptron(n_features=2)
        losses = m.fit(X_train, y_train, X_test, y_test, epochs=50, lr=lr)
        plt.plot(losses, label=f'lr={lr}')
    plt.title('Learning Rate Experiment')
    plt.legend()
    plt.savefig('plots/image_1531a7.png', dpi=300)
    plt.close()

    # 2. Размер батча
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
    plt.figure(figsize=(10, 5))
    for l2 in [0.0, 0.01, 1.0]:
        m = Perceptron(n_features=2, l2_lambda=l2)
        plt.plot(m.fit(X_train, y_train, X_test, y_test, epochs=100), label=f"L2={l2}")
    plt.title('L2 Regularization')
    plt.legend()
    plt.savefig('plots/bonus_l2.png', dpi=300)
    plt.close()

    # 7. ROC Curve
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

    # 8. Ошибки на XOR
    print("Генерация графика ошибок XOR...")
    # Загружаем датасет XOR (он уже есть в обновленном data_utils.py)
    X_train_xor, X_test_xor, y_train_xor, y_test_xor = get_prepared_data(dataset_type="xor")

    # Обучаем модель
    xor_model = Perceptron(n_features=2)
    xor_model.fit(X_train_xor, y_train_xor, X_test_xor, y_test_xor, epochs=100)

    # Делаем предсказания и ищем ошибки
    y_pred_xor = xor_model.predict(X_test_xor)
    incorrect = y_test_xor != y_pred_xor

    # Строим график
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_xor[~incorrect, 0], X_test_xor[~incorrect, 1], c="blue", alpha=0.5, label="Correct (Верно)")
    plt.scatter(X_test_xor[incorrect, 0], X_test_xor[incorrect, 1], c="red", s=100, edgecolors="black",
                label="Misclassified (Ошибка)")
    plt.title("Ошибки классификации на датасете XOR")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/bonus_xor_errors.png', dpi=300)
    plt.close()

    print("Все графики успешно сохранены в папку 'plots/'!")

if __name__ == "__main__":
    run_all_experiments()
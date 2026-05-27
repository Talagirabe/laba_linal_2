import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from data_utils import get_prepared_data
from perceptron import Perceptron

# Создаем папку, если её нет
os.makedirs('plots', exist_ok=True)
X_train, X_test, y_train, y_test = get_prepared_data()

print("=== БОНУС 3: РАСШИРЕННЫЕ МЕТРИКИ ===")
model = Perceptron(n_features=2)
model.fit(X_train, y_train, X_test, y_test, epochs=100)

y_pred = model.predict(X_test)
y_probs = model.forward(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred):.4f}")

# График ROC-кривой
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")
plt.legend()
plt.grid(True)
plt.savefig('plots/bonus_roc_curve.png', dpi=300)
plt.close()

print("\n=== БОНУС 4: MOMENTUM VS SGD ===")
betas = [0.5, 0.9, 0.99]
plt.figure(figsize=(10, 6))

# Обычный SGD
sgd_model = Perceptron(n_features=2, optimizer="sgd")
sgd_losses, _ = sgd_model.fit(X_train, y_train, X_test, y_test, epochs=100)
plt.plot(sgd_losses, label="Обычный SGD", linewidth=2)

# Momentum с разными коэффициентами
for beta in betas:
    m_model = Perceptron(n_features=2, optimizer="momentum", beta=beta)
    m_losses, _ = m_model.fit(X_train, y_train, X_test, y_test, epochs=100)
    plt.plot(m_losses, label=f"Momentum (beta={beta})")

plt.title("Сравнение скорости сходимости: SGD против Momentum")
plt.xlabel("Эпоха")
plt.ylabel("Loss (Train)")
plt.legend()
plt.grid(True)
plt.savefig('plots/bonus_momentum.png', dpi=300)
plt.close()

print("\nВсе бонусные графики успешно сохранены в папку 'plots/'!")
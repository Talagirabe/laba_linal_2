import os
import matplotlib.pyplot as plt
from data_utils import get_prepared_data
from perceptron import Perceptron

# Создаем папку для графиков, если её еще нет
os.makedirs('plots', exist_ok=True)

X_train, X_test, y_train, y_test = get_prepared_data()

print("Запуск экспериментов...")

# 1. Эксперимент со скоростью обучения
print("1/3 Обучение с разным learning rate...")
lrs = [0.001, 0.01, 0.5, 1.0]
plt.figure(figsize=(10, 5))
for lr in lrs:
    m = Perceptron(n_features=2)
    t_loss, _ = m.fit(X_train, y_train, X_test, y_test, epochs=50, lr=lr)
    plt.plot(t_loss, label=f'lr={lr}')
plt.title('Влияние скорости обучения (Train Loss)')
plt.legend()
plt.savefig('plots/exp_learning_rate.png', dpi=300)
plt.close()

# 2. Эксперимент с размером батча
print("2/3 Обучение с разным batch size...")
batches = [1, 16, 64, 256]
plt.figure(figsize=(10, 5))
for b in batches:
    m = Perceptron(n_features=2)
    t_loss, _ = m.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=b)
    plt.plot(t_loss, label=f'batch={b}')
plt.title('Влияние размера батча (Train Loss)')
plt.legend()
plt.savefig('plots/exp_batch_size.png', dpi=300)
plt.close()

# 3. Эксперимент с инициализацией весов
print("3/3 Обучение с разной инициализацией весов...")
inits = ['zero', 'small', 'large']
plt.figure(figsize=(10, 5))
for init in inits:
    m = Perceptron(n_features=2, init_type=init)
    t_loss, _ = m.fit(X_train, y_train, X_test, y_test, epochs=50)
    plt.plot(t_loss, label=f'init={init}')
plt.title('Влияние инициализации весов (Train Loss)')
plt.legend()
plt.savefig('plots/exp_weight_init.png', dpi=300)
plt.close()

print("Все эксперименты завершены. Графики сохранены в папку 'plots/'.")
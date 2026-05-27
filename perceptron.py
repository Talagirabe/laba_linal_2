import numpy as np


class Perceptron:
    def __init__(self, n_features, init_type='small', loss_type="cross_entropy", l2_lambda=0.0, optimizer="sgd",
                 beta=0.9):
        # Инициализация весов
        if init_type == 'small':
            self.w = np.random.randn(n_features) * 0.01
        elif init_type == 'zero':
            self.w = np.zeros(n_features)
        elif init_type == 'large':
            self.w = np.random.randn(n_features) * 10
        self.b = 0.0

        # Гиперпараметры (включая настройки для бонусных заданий)
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer
        self.beta = beta

        # Векторы инерции для метода Momentum
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0.0

    def sigmoid(self, z):
        # Ограничиваем z, чтобы избежать ошибки переполнения (overflow) в экспоненте
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def linear_output(self, X):
        return np.dot(X, self.w) + self.b

    def forward(self, X):
        return self.sigmoid(self.linear_output(X))

    def compute_loss(self, y_true, y_pred):
        if self.loss_type == "cross_entropy":
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            # Добавляем L2-штраф
            loss += self.l2_lambda * np.sum(self.w ** 2)
            return loss

        elif self.loss_type == "hinge":
            return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        train_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Перемешивание данных перед каждой эпохой (Stochastic)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                m = X_batch.shape[0]

                # --- 1. Расчет градиентов ---
                if self.loss_type == "cross_entropy":
                    y_pred = self.forward(X_batch)
                    error = y_pred - y_batch
                    # Градиент с учетом производной L2-регуляризации
                    grad_w = (np.dot(X_batch.T, error) / m) + (2 * self.l2_lambda * self.w)
                    grad_b = np.sum(error) / m

                elif self.loss_type == "hinge":
                    scores = self.linear_output(X_batch)
                    margins = y_batch * scores
                    condition = margins < 1
                    if np.any(condition):
                        grad_w = -np.mean(X_batch[condition] * y_batch[condition][:, np.newaxis], axis=0)
                        grad_b = -np.mean(y_batch[condition])
                    else:
                        grad_w = np.zeros_like(self.w)
                        grad_b = 0.0

                # --- 2. Обновление весов ---
                if self.optimizer == "sgd":
                    self.w -= lr * grad_w
                    self.b -= lr * grad_b

                elif self.optimizer == "momentum":
                    self.v_w = self.beta * self.v_w - lr * grad_w
                    self.v_b = self.beta * self.v_b - lr * grad_b
                    self.w += self.v_w
                    self.b += self.v_b

            # Сохранение истории ошибки (нужно для графиков)
            if self.loss_type == "cross_entropy":
                train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            elif self.loss_type == "hinge":
                train_losses.append(self.compute_loss(y_train, self.linear_output(X_train)))

        # Возвращаем ровно 1 массив, чтобы не сломать matplotlib
        return train_losses

    def predict(self, X):
        if self.loss_type == "cross_entropy":
            return (self.forward(X) >= 0.5).astype(int)
        elif self.loss_type == "hinge":
            return np.where(self.linear_output(X) >= 0, 1, -1)
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

        # Гиперпараметры бонусов
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer
        self.beta = beta

        # Векторы скорости для Momentum
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0.0

    def sigmoid(self, z):
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
            # Добавление L2-регуляризации
            loss += self.l2_lambda * np.sum(self.w ** 2)
            return loss

        elif self.loss_type == "hinge":
            return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        self.train_losses = []
        self.val_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                m = X_batch.shape[0]

                # Расчет градиентов в зависимости от функции потерь
                if self.loss_type == "cross_entropy":
                    y_pred = self.forward(X_batch)
                    error = y_pred - y_batch
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

                # Обновление весов (SGD или Momentum)
                if self.optimizer == "sgd":
                    self.w -= lr * grad_w
                    self.b -= lr * grad_b

                elif self.optimizer == "momentum":
                    self.v_w = self.beta * self.v_w - lr * grad_w
                    self.v_b = self.beta * self.v_b - lr * grad_b
                    self.w += self.v_w
                    self.b += self.v_b

            # Сохранение лоссов для графиков
            if self.loss_type == "cross_entropy":
                self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
                self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))
            elif self.loss_type == "hinge":
                self.train_losses.append(self.compute_loss(y_train, self.linear_output(X_train)))
                self.val_losses.append(self.compute_loss(y_val, self.linear_output(X_val)))

        return self.train_losses, self.val_losses

    def predict(self, X):
        if self.loss_type == "cross_entropy":
            return (self.forward(X) >= 0.5).astype(int)
        elif self.loss_type == "hinge":
            return np.where(self.linear_output(X) >= 0, 1, -1)

import numpy as np

class Perceptron:
    def __init__(self, n_features, init_type='small', loss_type="cross_entropy", l2_lambda=0.0, optimizer="sgd", beta=0.9):
        if init_type == 'small':
            self.w = np.random.randn(n_features) * 0.01
        elif init_type == 'zero':
            self.w = np.zeros(n_features)
        elif init_type == 'large':
            self.w = np.random.randn(n_features) * 10
        self.b = 0.0

        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer
        self.beta = beta
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0.0

    def sigmoid(self, z):
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
            loss += self.l2_lambda * np.sum(self.w ** 2)
            return loss
        elif self.loss_type == "hinge":
            return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        train_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                m = X_batch.shape[0]

                if self.loss_type == "cross_entropy":
                    y_pred = self.forward(X_batch)
                    error = y_pred - y_batch
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

                if self.optimizer == "sgd":
                    self.w -= lr * grad_w
                    self.b -= lr * grad_b
                elif self.optimizer == "momentum":
                    self.v_w = self.beta * self.v_w - lr * grad_w
                    self.v_b = self.beta * self.v_b - lr * grad_b
                    self.w += self.v_w
                    self.b += self.v_b

            if self.loss_type == "cross_entropy":
                train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            elif self.loss_type == "hinge":
                train_losses.append(self.compute_loss(y_train, self.linear_output(X_train)))

        return train_losses

    def predict(self, X):
        if self.loss_type == "cross_entropy":
            return (self.forward(X) >= 0.5).astype(int)
        elif self.loss_type == "hinge":
            return np.where(self.linear_output(X) >= 0, 1, -1)
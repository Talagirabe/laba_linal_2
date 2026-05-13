import numpy as np

class Perceptron:
    def __init__(self, n_features, init_type='small'):
        if init_type == 'small':
            self.w = np.random.randn(n_features) * 0.01
        elif init_type == 'zero':
            self.w = np.zeros(n_features)
        elif init_type == 'large':
            self.w = np.random.randn(n_features) * 10
        self.b = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        train_losses = []
        val_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(X_batch)

                dw = np.dot(X_batch.T, (y_pred - y_batch)) / X_batch.shape[0]
                db = np.sum(y_pred - y_batch) / X_batch.shape[0]

                self.w -= lr * dw
                self.b -= lr * db

            train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

        return train_losses, val_losses

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
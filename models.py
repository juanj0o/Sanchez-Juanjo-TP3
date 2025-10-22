import numpy as np
import pandas as pd

class MLP:
    def __init__(self, n_layers, layer_sizes, batch_size, init='random'):
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

        self.weights = None 
        self.biases = None
        self.init = init
        self.initialize_parameters()

    def initialize_parameters(self):
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            if self.init == 'random':
                # Inicialización aleatoria pequeña
                W = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * 0.01
            else:  # glorot (Xavier initialization)
                # Glorot/Xavier: mejor para redes profundas
                limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
                W = np.random.uniform(-limit, limit, (self.layer_sizes[i+1], self.layer_sizes[i]))

            b = np.zeros((self.layer_sizes[i+1], 1))

            self.weights.append(W)
            self.biases.append(b)
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_gradient(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.hidden_outs = [X]
        self.activations = [X]

        for i in range(self.n_layers):
            z = self.weights[i] @ self.activations[-1] + self.biases[i]
            self.hidden_outs.append(z)

            if i == self.n_layers - 1:
                a = self.softmax(z)
            else:
                a = self.relu(z)
            self.activations.append(a)

        return a

    def backpropagation(self, X, y):
        m = X.shape[1]  # Número de ejemplos en el batch

        # Inicializar listas para los gradientes
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        # Gradiente de la capa de salida (softmax + cross-entropy)
        # La derivada de cross-entropy con softmax se simplifica a: y_pred - y_true
        dz = self.activations[-1] - y

        # Backpropagation desde la última capa hacia la primera
        for i in range(self.n_layers - 1, -1, -1):
            # Gradientes de pesos y biases
            dW[i] = (1 / m) * (dz @ self.activations[i].T)
            db[i] = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            # Si no estamos en la primera capa, calcular gradiente para la capa anterior
            if i > 0:
                dz = (self.weights[i].T @ dz) * self.relu_gradient(self.hidden_outs[i])

        return dW, db



    def train(self, X, y, epochs, learning_rate):
        losses = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Calcular loss (cross-entropy)
            m = X.shape[1]
            loss = -(1 / m) * np.sum(y * np.log(output + 1e-8))
            losses.append(loss)

            # Backward pass
            dW, db = self.backpropagation(X, y)

            # Actualizar parámetros
            for i in range(self.n_layers):
                self.weights[i] -= learning_rate * dW[i]
                self.biases[i] -= learning_rate * db[i]

            # Mostrar progreso cada 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        # Hacer forward pass sin guardar estados intermedios
        output = self.forward(X)
        # Retornar la clase con mayor probabilidad
        return np.argmax(output, axis=0)




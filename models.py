import numpy as np
import pandas as pd
import cupy as cp

class MLP:
    def __init__(self, n_layers, layer_sizes, batch_size, init='random', optimizer='sgd'):
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

        self.weights = None 
        self.biases = None
        self.init = init
        self.initialize_parameters()

        self.optimizer = optimizer
        
        # Para Adam
        self.m_w = None  # primer momento (momentum)
        self.v_w = None  # segundo momento (velocity)
        self.m_b = None
        self.v_b = None
        self.t = 0  # timestep para Adam

    def initialize_parameters(self):
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            if self.init == 'random':
                W = cp.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * 0.01
            else:           
                limit = cp.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
                W = cp.random.uniform(-limit, limit, (self.layer_sizes[i+1], self.layer_sizes[i]))

            b = cp.zeros((self.layer_sizes[i+1], 1))

            self.weights.append(W)
            self.biases.append(b)

    def initialize_adam(self):
        self.m_w = [cp.zeros_like(w) for w in self.weights]
        self.v_w = [cp.zeros_like(w) for w in self.weights]
        self.m_b = [cp.zeros_like(b) for b in self.biases]
        self.v_b = [cp.zeros_like(b) for b in self.biases]
        self.t = 0

    def adam_update(self, dW, db, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        
        for i in range(self.n_layers):
            # Update momentum
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dW[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db[i]
            
            # Update velocity
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dW[i] ** 2)
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db[i] ** 2)
            
            # Bias correction
            m_w_corrected = self.m_w[i] / (1 - beta1 ** self.t)
            m_b_corrected = self.m_b[i] / (1 - beta1 ** self.t)
            v_w_corrected = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_corrected = self.v_b[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= learning_rate * m_w_corrected / (cp.sqrt(v_w_corrected) + epsilon)
            self.biases[i] -= learning_rate * m_b_corrected / (cp.sqrt(v_b_corrected) + epsilon)
        
    def softmax(self, x):
        e_x = cp.exp(x - cp.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)
    
    def relu(self, x):
        return cp.maximum(0, x)
    
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

    def backpropagation(self, X, y, lambda_reg=0):
        m = X.shape[1]
        
        dW = [None] * self.n_layers
        db = [None] * self.n_layers
        
        dz = self.activations[-1] - y
        
        for i in range(self.n_layers - 1, -1, -1):
            dW[i] = (1 / m) * (dz @ self.activations[i].T)
            
            # Agregar gradiente de L2
            if lambda_reg > 0:
                dW[i] += (lambda_reg / m) * self.weights[i]
            
            db[i] = (1 / m) * cp.sum(dz, axis=1, keepdims=True)
            
            if i > 0:
                dz = (self.weights[i].T @ dz) * self.relu_gradient(self.hidden_outs[i])
        
        return dW, db

    def linear_lr_schedule(self, epoch, initial_lr, final_lr, total_epochs):
        """Learning rate decay lineal con saturación"""
        if epoch >= total_epochs:
            return final_lr
        return initial_lr - (initial_lr - final_lr) * (epoch / total_epochs)

    def exponential_lr_schedule(self, epoch, initial_lr, decay_rate):
        """Learning rate decay exponencial"""
        return initial_lr * (decay_rate ** epoch)

    def create_mini_batches(self, X, y):
        """
        Crea mini-batches de manera eficiente.

        OPTIMIZACIÓN: En lugar de copiar todo el dataset shuffled,
        solo generamos índices shuffled y luego creamos batches directamente.
        Esto evita copiar ~3.5GB de datos en memoria.
        """
        m = X.shape[1]
        mini_batches = []

        # Verificar si X está en CPU (numpy) o GPU (cupy)
        is_numpy = isinstance(X, np.ndarray)

        # Generar permutación de índices (rápido, solo ~566K enteros = ~2MB)
        if is_numpy:
            permutation = np.random.permutation(m)
        else:
            permutation = cp.random.permutation(m)

        num_complete_batches = m // self.batch_size

        # Crear batches usando indexación directa (sin copiar todo el dataset)
        for k in range(num_complete_batches):
            batch_indices = permutation[k * self.batch_size:(k + 1) * self.batch_size]
            mini_batch_X = X[:, batch_indices]
            mini_batch_y = y[:, batch_indices]
            mini_batches.append((mini_batch_X, mini_batch_y))

        # Si hay un último batch incompleto, agregarlo también
        if m % self.batch_size != 0:
            batch_indices = permutation[num_complete_batches * self.batch_size:]
            mini_batch_X = X[:, batch_indices]
            mini_batch_y = y[:, batch_indices]
            mini_batches.append((mini_batch_X, mini_batch_y))

        return mini_batches

    def compute_loss(self, y_true, y_pred, lambda_reg=0):
        m = y_true.shape[1]
        cross_entropy = -(1 / m) * cp.sum(y_true * cp.log(y_pred + 1e-8))
        
        if lambda_reg > 0:
            l2_penalty = 0
            for w in self.weights:
                l2_penalty += cp.sum(w ** 2)
            l2_penalty *= (lambda_reg / (2 * m))
            return cross_entropy + l2_penalty
    
        return cross_entropy

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100,
          learning_rate=0.01, lr_schedule=None, lr_schedule_params=None,
          lambda_reg=0, early_stopping_patience=None,
          beta1=0.9, beta2=0.999, verbose=True, use_gpu_cache=False):

        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None,
            'learning_rates': []
        }

        # Inicializar Adam si es necesario
        if self.optimizer == 'adam':
            self.initialize_adam()

        # Inicializar early stopping si es necesario
        early_stopper = None
        if early_stopping_patience is not None and X_val is not None:
            early_stopper = EarlyStopping(patience=early_stopping_patience)

        is_numpy = isinstance(X_train, np.ndarray)

        # OPTIMIZACIÓN OPCIONAL: Cachear datos en GPU si el usuario lo solicita
        # ADVERTENCIA: Solo funciona si los datos caben en VRAM
        if use_gpu_cache and is_numpy:
            try:
                if verbose:
                    print("⚠️  EXPERIMENTAL: Intentando cachear datos en GPU...")
                    print(f"   Tamaño estimado: {(X_train.nbytes + y_train.nbytes) / 1024**3:.2f} GB")
                X_train = cp.asarray(X_train)
                y_train = cp.asarray(y_train)
                if X_val is not None and y_val is not None:
                    X_val = cp.asarray(X_val)
                    y_val = cp.asarray(y_val)
                is_numpy = False  # Ahora están en GPU
                if verbose:
                    print("   ✅ Datos cacheados en GPU exitosamente!\n")
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error al cachear en GPU: {e}")
                    print("   → Continuando con transferencias por batch...\n")

        # Guardar el mejor modelo para early stopping
        best_weights = None
        best_biases = None
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Calcular learning rate actual
            current_lr = learning_rate

            if lr_schedule == 'linear' and lr_schedule_params:
                current_lr = self.linear_lr_schedule(
                    epoch,
                    learning_rate,
                    lr_schedule_params.get('final_lr', learning_rate * 0.01),
                    lr_schedule_params.get('total_epochs', epochs)
                )
            elif lr_schedule == 'exponential' and lr_schedule_params:
                current_lr = self.exponential_lr_schedule(
                    epoch,
                    learning_rate,
                    lr_schedule_params.get('decay_rate', 0.96)
                )

            history['learning_rates'].append(float(current_lr))

            # Crear mini-batches
            mini_batches = self.create_mini_batches(X_train, y_train)

            epoch_loss = 0

            # Entrenar en cada mini-batch
            for mini_batch_X, mini_batch_y in mini_batches:
                if is_numpy:
                    mini_batch_X_gpu = cp.asarray(mini_batch_X)
                    mini_batch_y_gpu = cp.asarray(mini_batch_y)
                else:
                    mini_batch_X_gpu = mini_batch_X
                    mini_batch_y_gpu = mini_batch_y
                # Forward pass
                output = self.forward(mini_batch_X_gpu)

                # Calcular loss con regularización L2
                batch_loss = self.compute_loss(mini_batch_y_gpu, output, lambda_reg)
                epoch_loss += batch_loss

                # Backward pass
                dW, db = self.backpropagation(mini_batch_X_gpu, mini_batch_y_gpu, lambda_reg)

                # Update parameters
                if self.optimizer == 'adam':
                    self.adam_update(dW, db, current_lr, beta1, beta2)
                else:  # SGD
                    for i in range(self.n_layers):
                        self.weights[i] -= current_lr * dW[i]
                        self.biases[i] -= current_lr * db[i]
            
            # Calcular loss promedio de la época
            avg_train_loss = epoch_loss / len(mini_batches)
            history['train_loss'].append(float(avg_train_loss))

            # Validación
            if X_val is not None and y_val is not None:
                if is_numpy:
                    val_loss = self._compute_validation_loss_batched(X_val, y_val, lambda_reg)
                else:
                    val_output = self.forward(X_val)
                    val_loss = self.compute_loss(y_val, val_output, lambda_reg)
                
                history['val_loss'].append(float(val_loss))
                
                # Guardar mejor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                
                # Early stopping
                if early_stopper is not None:
                    early_stopper(val_loss)
                    if early_stopper.early_stop:
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                            print(f"Restoring best weights from epoch {epoch - early_stopper.patience}")
                        # Restaurar mejor modelo
                        self.weights = best_weights
                        self.biases = best_biases
                        break
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - LR: {current_lr:.6f} - "
                        f"Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - LR: {current_lr:.6f} - "
                        f"Train Loss: {avg_train_loss:.4f}")
        
        # Al final del entrenamiento, restaurar el mejor modelo si hubo validación
        if X_val is not None and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
        
        return history

    def _compute_validation_loss_batched(self, X_val, y_val, lambda_reg=0):
        m = X_val.shape[1]
        total_loss = 0
        
        for i in range(0, m, self.batch_size):
            end_idx = min(i + self.batch_size, m)
            X_batch = cp.asarray(X_val[:, i:end_idx])
            y_batch = cp.asarray(y_val[:, i:end_idx])
            
            output = self.forward(X_batch)
            batch_loss = self.compute_loss(y_batch, output, lambda_reg)
            total_loss += batch_loss * (end_idx - i)
        
        return total_loss / m

    def predict(self, X):
        output = self.forward(X)
        return cp.argmax(output, axis=0)



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
import numpy as np
import pandas as pd
import cupy as cp
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

        start_time = time.time()

        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None,
            'learning_rates': []
        }

        if self.optimizer == 'adam':
            self.initialize_adam()

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

        best_weights = None
        best_biases = None
        best_val_loss = float('inf')

        for epoch in range(epochs):
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

            mini_batches = self.create_mini_batches(X_train, y_train)

            epoch_loss = 0

            for mini_batch_X, mini_batch_y in mini_batches:
                if is_numpy:
                    mini_batch_X_gpu = cp.asarray(mini_batch_X)
                    mini_batch_y_gpu = cp.asarray(mini_batch_y)
                else:
                    mini_batch_X_gpu = mini_batch_X
                    mini_batch_y_gpu = mini_batch_y
                output = self.forward(mini_batch_X_gpu)

                batch_loss = self.compute_loss(mini_batch_y_gpu, output, lambda_reg)
                epoch_loss += batch_loss

                dW, db = self.backpropagation(mini_batch_X_gpu, mini_batch_y_gpu, lambda_reg)

                if self.optimizer == 'adam':
                    self.adam_update(dW, db, current_lr, beta1, beta2)
                else: 
                    for i in range(self.n_layers):
                        self.weights[i] -= current_lr * dW[i]
                        self.biases[i] -= current_lr * db[i]
            
            avg_train_loss = epoch_loss / len(mini_batches)
            history['train_loss'].append(float(avg_train_loss))

            if X_val is not None and y_val is not None:
                if is_numpy:
                    val_loss = self._compute_validation_loss_batched(X_val, y_val, lambda_reg)
                else:
                    val_output = self.forward(X_val)
                    val_loss = self.compute_loss(y_val, val_output, lambda_reg)
                
                history['val_loss'].append(float(val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                
                if early_stopper is not None:
                    early_stopper(val_loss)
                    if early_stopper.early_stop:
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                            print(f"Restoring best weights from epoch {epoch - early_stopper.patience}")
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
        
        if X_val is not None and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")

        end_time = time.time()
        total_time = end_time - start_time
        epochs_trained = len(history['train_loss'])
        time_per_epoch = total_time / epochs_trained if epochs_trained > 0 else 0

        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING TIME SUMMARY")
            print(f"{'='*60}")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
            print(f"Epochs trained: {epochs_trained}")
            print(f"Time per epoch: {time_per_epoch:.2f}s")
            print(f"{'='*60}\n")

        history['training_time'] = {
            'total_seconds': total_time,
            'total_minutes': total_time / 60,
            'epochs_trained': epochs_trained,
            'seconds_per_epoch': time_per_epoch
        }

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


class MLP_PyTorch(nn.Module):

    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP_PyTorch, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Add ReLU activation for all layers except the last one
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLP_PyTorch_Advanced(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation='relu', dropout_rate=0.0):
        super(MLP_PyTorch_Advanced, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        activation_dict = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'swish': nn.SiLU()
        }

        act_fn = activation_dict.get(activation.lower(), nn.ReLU())

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_fn)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchTrainer:

    def __init__(self, model, device='cuda'):

        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, X_train, y_train, X_val, y_val, batch_size=128):

        X_train_torch = torch.tensor(X_train.astype(np.float32)).to(self.device)
        y_train_torch = torch.tensor(y_train.astype(np.int64)).to(self.device)

        X_val_torch = torch.tensor(X_val.astype(np.float32)).to(self.device)
        y_val_torch = torch.tensor(y_val.astype(np.int64)).to(self.device)

        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        val_dataset = TensorDataset(X_val_torch, y_val_torch)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, criterion):
        """Evaluate model on given data loader."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)

    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001,
              weight_decay=0.001, lr_schedule=None, lr_schedule_params=None,
              early_stopping_patience=None, verbose=True):

        start_time = time.time()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Setup learning rate scheduler
        scheduler = None
        if lr_schedule == 'exponential' and lr_schedule_params:
            decay_rate = lr_schedule_params.get('decay_rate', 0.95)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        # Setup early stopping
        early_stopper = None
        if early_stopping_patience is not None:
            early_stopper = EarlyStopping(patience=early_stopping_patience)

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)

            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Early stopping check
            if early_stopper is not None:
                early_stopper(val_loss)
                if early_stopper.early_stop:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        print(f"Restoring best weights from epoch {epoch - early_stopper.patience}")
                    self.model.load_state_dict(best_model_state)
                    break

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

            if verbose and (epoch % 10 == 0 or epoch == 0):
                print(f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}, "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")

        end_time = time.time()
        total_time = end_time - start_time
        epochs_trained = len(history['train_loss'])
        time_per_epoch = total_time / epochs_trained if epochs_trained > 0 else 0

        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING TIME SUMMARY")
            print(f"{'='*60}")
            if total_time >= 60:
                mins = int(total_time // 60)
                secs = int(total_time % 60)
                print(f"Total time: {mins}min {secs}s ({total_time:.2f}s)")
            else:
                print(f"Total time: {total_time:.2f}s")
            print(f"Epochs trained: {epochs_trained}")
            print(f"Time per epoch: {time_per_epoch:.2f}s")
            print(f"{'='*60}\n")

        history['training_time'] = {
            'total_seconds': total_time,
            'total_minutes': total_time / 60,
            'epochs_trained': epochs_trained,
            'seconds_per_epoch': time_per_epoch
        }

        return history

    def predict(self, X):
        self.model.eval()
        X_torch = torch.tensor(X.astype(np.float32)).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_torch)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()
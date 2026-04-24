"""Deep Learning models for heat risk forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score


class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM architecture for time series classification.
    
    Processes monthly climate features through:
    1. Convolutional feature extraction (Conv1D)
    2. Sequential modeling (LSTM)
    3. Dense classification head
    """
    
    def __init__(self, input_size: int, num_classes: int = 4, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: (batch_size, seq_len, input_size) tensor
        
        Returns:
            logits: (batch_size, num_classes) tensor
        """
        # Conv expects (batch, channels, length)
        x = x.transpose(1, 2)  # (B, input_size, seq_len)
        
        # CNN blocks
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Back to (B, seq_len, channels)
        x = x.transpose(1, 2)  # (B, seq_len, 128)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last output
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class DLForecastModel:
    """PyTorch wrapper for training and inference."""
    
    def __init__(self, input_size: int, num_classes: int = 4, hidden_size: int = 64, 
                 learning_rate: float = 0.001, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = HybridCNNLSTM(input_size, num_classes, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        self.input_size = input_size
        self.num_classes = num_classes
    
    def _get_class_weights(self, y):
        """Compute class weights for imbalanced data."""
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts)
        return torch.FloatTensor(weights).to(self.device)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 60, 
            batch_size: int = 16, patience: int = 10):
        """Train the model.
        
        Args:
            X_train: (n_samples, seq_len, n_features) - training features
            y_train: (n_samples,) - training labels
            X_val: validation features (optional)
            y_val: validation labels (optional)
            epochs: max epochs
            batch_size: batch size
            patience: early stopping patience
        
        Returns:
            history: dict with train/val metrics
        """
        # Fit scaler and normalize
        n_samples = X_train.shape[0]
        X_train_2d = X_train.reshape(n_samples, -1)
        self.scaler.fit(X_train_2d)
        X_train = self._normalize(X_train)
        
        if X_val is not None:
            X_val = self._normalize(X_val)
        
        # Prepare dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Class weights
        class_weights = self._get_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        history = {"train_loss": [], "val_f1": []}
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_f1, _ = self._evaluate(X_val, y_val)
                history["val_f1"].append(val_f1)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        return history
    
    def _normalize(self, X):
        """Normalize features using fitted scaler."""
        n_samples, seq_len, n_features = X.shape
        X_2d = X.reshape(n_samples, -1)
        X_2d = self.scaler.transform(X_2d)
        return X_2d.reshape(n_samples, seq_len, n_features)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Handles both 2D (sklearn-style) and 3D (sequence) input.
        - 2D input (n_samples, n_features): treated as single timestep
        - 3D input (n_samples, seq_len, n_features): used as-is
        """
        # Handle 2D input: reshape to 3D with seq_len=1
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)  # (n_samples, n_features) -> (n_samples, 1, n_features)
        
        X = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba
    
    def _evaluate(self, X, y):
        """Compute metrics on validation/test set."""
        y_pred = self.predict(X)
        f1 = f1_score(y, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y, y_pred)
        return f1, acc
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        f1, acc = self._evaluate(X_test, y_test)
        return {"macro_f1": float(f1), "accuracy": float(acc)}

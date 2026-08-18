import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score

class HeartMLP(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=[64, 32, 16], dropout_rate=0.2):
        super(HeartMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MLPClassifier:
    def __init__(self, input_dim=13, hidden_dims=[64, 32, 16], lr=1e-3, weight_decay=1e-3, epochs=200, batch_size=32, device=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HeartMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
    def fit(self, X_train, y_train, X_val, y_val, patience=30):
        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_va_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_val_auc = 0.0
        best_weights = None
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch_x.size(0)
                
            train_loss = running_loss / len(X_train)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_va_t)
                val_loss = self.criterion(val_out, y_va_t).item()
                val_probs = torch.sigmoid(val_out).cpu().numpy().flatten()
                
            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except Exception:
                val_auc = 0.5
                
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
        if best_weights:
            self.model.load_state_dict(best_weights)
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
        return np.column_stack((1 - probs, probs))
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, filepath="models/mlp_model.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'history': self.history
        }, filepath)
        print(f"Saved MLP model to {filepath}")
        
    def load(self, filepath="models/mlp_model.pth"):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.model = HeartMLP(input_dim=self.input_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.history = checkpoint.get('history', {})
        self.model.eval()
        print(f"Loaded MLP model from {filepath}")
        return self

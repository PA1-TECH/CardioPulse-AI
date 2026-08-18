import numpy as np
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier as PyTorchTabNetClassifier

class TabNetModel:
    def __init__(self, n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-3, seed=42):
        self.model = PyTorchTabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-4),
            scheduler_params=dict(step_size=20, gamma=0.7),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax',
            seed=seed,
            verbose=0
        )
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, max_epochs=200, patience=30, batch_size=64):
        eval_set = []
        eval_name = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            eval_name = ['val']
            
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=['auc', 'logloss'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=16,
            num_workers=0,
            drop_last=False
        )
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importances(self):
        return self.model.feature_importances_

    def save(self, filepath="models/tabnet_model"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # TabNet automatically appends .zip
        saved_path = self.model.save_model(filepath)
        print(f"Saved TabNet model to {saved_path}")
        return saved_path
        
    def load(self, filepath="models/tabnet_model.zip"):
        if filepath.endswith(".zip"):
            filepath = filepath[:-4]
        self.model.load_model(f"{filepath}.zip")
        print(f"Loaded TabNet model from {filepath}.zip")
        return self

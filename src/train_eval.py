import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.preprocessing import prepare_data, save_preprocessor, FEATURE_COLUMNS
from src.dnn import DNNClassifier
from src.tabnet import TabNetModel
from src.mlp import MLPClassifier

def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0

def evaluate_model(model_name, model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    spec = compute_specificity(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    return {
        'model_name': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': cm.tolist(),
        'probabilities': probs,
        'predictions': preds
    }

def main():
    print("==================================================")
    print("Heart Disease Model Training & Evaluation Pipeline")
    print("==================================================")
    
    output_dir = os.path.abspath("output")
    models_dir = os.path.abspath("models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Prepare Data
    print("\n[1/6] Loading and Preprocessing Data...")
    data = prepare_data(data_path="dataset/heart_clean.csv", test_size=0.2, val_size=0.15, random_state=42)
    
    X_train_scaled = data['X_train_scaled']
    y_train = data['y_train'].values
    X_val_scaled = data['X_val_scaled']
    y_val = data['y_val'].values
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test'].values
    scaler = data['scaler']
    
    save_preprocessor(scaler, output_dir=models_dir)
    print(f"Data split sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # 2. Train DNN
    print("\n[2/6] Training Deep Neural Network (DNN)...")
    dnn = DNNClassifier(input_dim=len(FEATURE_COLUMNS), lr=1e-3, epochs=250, batch_size=32)
    dnn.fit(X_train_scaled, y_train, X_val_scaled, y_val, patience=35)
    dnn.save(os.path.join(models_dir, "dnn_model.pth"))
    
    # 3. Train TabNet
    print("\n[3/6] Training TabNet Model...")
    tabnet = TabNetModel(n_d=16, n_a=16, n_steps=4, gamma=1.3)
    tabnet.fit(X_train_scaled, y_train, X_val_scaled, y_val, max_epochs=250, patience=35, batch_size=64)
    tabnet.save(os.path.join(models_dir, "tabnet_model"))
    
    # 4. Train MLP
    print("\n[4/6] Training Multi-Layer Perceptron (MLP)...")
    mlp = MLPClassifier(input_dim=len(FEATURE_COLUMNS), hidden_dims=[64, 32, 16], lr=1e-3, epochs=250, batch_size=32)
    mlp.fit(X_train_scaled, y_train, X_val_scaled, y_val, patience=35)
    mlp.save(os.path.join(models_dir, "mlp_model.pth"))
    
    # 5. Evaluate All Models
    print("\n[5/6] Evaluating Models on Test Set...")
    eval_results = []
    
    dnn_res = evaluate_model("DNN", dnn, X_test_scaled, y_test)
    tabnet_res = evaluate_model("TabNet", tabnet, X_test_scaled, y_test)
    mlp_res = evaluate_model("MLP", mlp, X_test_scaled, y_test)
    
    # Ensemble Model Evaluation
    ensemble_probs = (dnn_res['probabilities'] + tabnet_res['probabilities'] + mlp_res['probabilities']) / 3.0
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    ens_acc = accuracy_score(y_test, ensemble_preds)
    ens_prec = precision_score(y_test, ensemble_preds, zero_division=0)
    ens_rec = recall_score(y_test, ensemble_preds, zero_division=0)
    ens_spec = compute_specificity(y_test, ensemble_preds)
    ens_f1 = f1_score(y_test, ensemble_preds, zero_division=0)
    ens_auc = roc_auc_score(y_test, ensemble_probs)
    ens_cm = confusion_matrix(y_test, ensemble_preds)
    
    ensemble_res = {
        'model_name': 'Ensemble (DNN+TabNet+MLP)',
        'accuracy': round(ens_acc, 4),
        'precision': round(ens_prec, 4),
        'recall': round(ens_rec, 4),
        'specificity': round(ens_spec, 4),
        'f1_score': round(ens_f1, 4),
        'roc_auc': round(ens_auc, 4),
        'confusion_matrix': ens_cm.tolist(),
        'probabilities': ensemble_probs,
        'predictions': ensemble_preds
    }
    
    models_metrics = [dnn_res, tabnet_res, mlp_res, ensemble_res]
    
    # Create Comparison DataFrame
    metrics_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall (Sensitivity)': m['recall'],
            'Specificity': m['specificity'],
            'F1 Score': m['f1_score'],
            'ROC-AUC': m['roc_auc']
        } for m in models_metrics
    ])
    
    print("\n=========================================")
    print("      MODEL PERFORMANCE COMPARISON       ")
    print("=========================================")
    print(metrics_df.to_string(index=False))
    
    # Save Metrics JSON
    json_path = os.path.join(output_dir, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump([
            {k: v for k, v in m.items() if k not in ['probabilities', 'predictions']}
            for m in models_metrics
        ], f, indent=2)
    print(f"\nSaved metrics comparison JSON to {json_path}")
    
    # 6. Generate Visualizations & Plots
    print("\n[6/6] Generating Performance Charts and Plots...")
    
    # Plot 1: ROC Curves
    plt.figure(figsize=(8, 6))
    for m in models_metrics:
        fpr, tpr, _ = roc_curve(y_test, m['probabilities'])
        plt.plot(fpr, tpr, label=f"{m['model_name']} (AUC = {m['roc_auc']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Heart Disease Prediction Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()
    
    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, m in enumerate(models_metrics):
        sns.heatmap(np.array(m['confusion_matrix']), annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['No Disease (0)', 'Disease (1)'],
                    yticklabels=['No Disease (0)', 'Disease (1)'])
        axes[idx].set_title(f"{m['model_name']} Confusion Matrix", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # Plot 3: Metrics Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    plot_df = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 4: TabNet Feature Importance
    try:
        importances = tabnet.get_feature_importances()
        feat_imp = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=True)
        plt.figure(figsize=(8, 6))
        feat_imp.plot(kind='barh', color='#2563EB')
        plt.title('TabNet Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Relative Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tabnet_feature_importance.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot TabNet feature importances: {e}")
        
    print(f"\nAll plots saved to {output_dir}/")
    print("Training and Evaluation pipeline complete!")

if __name__ == "__main__":
    main()

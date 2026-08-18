import os
import sys
import numpy as np
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_preprocessor, FEATURE_COLUMNS
from src.dnn import DNNClassifier
from src.tabnet import TabNetModel
from src.mlp import MLPClassifier
from src.recommendation import LifestyleRecommendationEngine

class HeartRiskAgent:
    def __init__(self, models_dir="models"):
        self.models_dir = os.path.abspath(models_dir)
        self.scaler, self.feature_names = load_preprocessor(self.models_dir)
        self.recommendation_engine = LifestyleRecommendationEngine()
        
        # Load DNN
        dnn_path = os.path.join(self.models_dir, "dnn_model.pth")
        if os.path.exists(dnn_path):
            self.dnn = DNNClassifier().load(dnn_path)
        else:
            self.dnn = None
            
        # Load TabNet
        tabnet_path = os.path.join(self.models_dir, "tabnet_model.zip")
        if os.path.exists(tabnet_path):
            self.tabnet = TabNetModel().load(os.path.join(self.models_dir, "tabnet_model"))
        else:
            self.tabnet = None
            
        # Load MLP
        mlp_path = os.path.join(self.models_dir, "mlp_model.pth")
        if os.path.exists(mlp_path):
            self.mlp = MLPClassifier().load(mlp_path)
        else:
            self.mlp = None

    def _prepare_input_array(self, patient_dict):
        """Converts dict into scaled numpy array matching feature columns."""
        df_input = pd.DataFrame([patient_dict])[self.feature_names]
        raw_arr = df_input.values
        scaled_arr = self.scaler.transform(df_input)
        return raw_arr, scaled_arr

    def predict(self, patient_dict, model_choice="ensemble"):
        raw_arr, scaled_arr = self._prepare_input_array(patient_dict)
        
        results = {}
        probs = []
        
        # DNN Prediction
        if self.dnn is not None:
            dnn_prob = float(self.dnn.predict_proba(scaled_arr)[0, 1])
            results['DNN'] = {
                'probability': round(dnn_prob, 4),
                'risk_percentage': round(dnn_prob * 100, 2),
                'class': int(dnn_prob >= 0.5)
            }
            probs.append(dnn_prob)
            
        # TabNet Prediction
        if self.tabnet is not None:
            tabnet_prob = float(self.tabnet.predict_proba(scaled_arr)[0, 1])
            results['TabNet'] = {
                'probability': round(tabnet_prob, 4),
                'risk_percentage': round(tabnet_prob * 100, 2),
                'class': int(tabnet_prob >= 0.5)
            }
            probs.append(tabnet_prob)
            
        # MLP Prediction
        if self.mlp is not None:
            mlp_prob = float(self.mlp.predict_proba(scaled_arr)[0, 1])
            results['MLP'] = {
                'probability': round(mlp_prob, 4),
                'risk_percentage': round(mlp_prob * 100, 2),
                'class': int(mlp_prob >= 0.5)
            }
            probs.append(mlp_prob)
            
        # Ensemble Average
        ensemble_prob = float(np.mean(probs)) if probs else 0.5
        results['Ensemble'] = {
            'probability': round(ensemble_prob, 4),
            'risk_percentage': round(ensemble_prob * 100, 2),
            'class': int(ensemble_prob >= 0.5)
        }
        
        # Select target model probability for recommendations
        selected_choice = model_choice.capitalize()
        if selected_choice not in results:
            selected_choice = "Ensemble"
            
        selected_prob = results[selected_choice]['probability']
        
        recommendations = self.recommendation_engine.generate_recommendations(patient_dict, selected_prob)
        
        return {
            'patient_input': patient_dict,
            'selected_model': selected_choice,
            'model_predictions': results,
            'selected_prediction': results[selected_choice],
            'recommendations': recommendations
        }

if __name__ == "__main__":
    import sys
    print("Testing HeartRiskAgent module...")

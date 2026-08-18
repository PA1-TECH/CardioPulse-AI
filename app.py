from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import HeartRiskAgent

app = Flask(__name__, template_folder='templates', static_folder='static')
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = HeartRiskAgent(models_dir="models")
    return agent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/benchmark')
def benchmark():
    return render_template('benchmark.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patient_data = data.get('patient', {})
        model_choice = data.get('model_choice', 'ensemble')
        
        current_agent = get_agent()
        result = current_agent.predict(patient_data, model_choice=model_choice)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = os.path.abspath("output/model_comparison.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify({'status': 'success', 'metrics': data})
    else:
        return jsonify({'status': 'error', 'message': 'Metrics not generated yet. Run training script first.'}), 444

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(os.path.abspath('output'), filename)

if __name__ == '__main__':
    print("Starting Heart Disease Risk Prediction & Lifestyle Recommendation Server...")
    get_agent()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

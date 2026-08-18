import http.server
import socketserver
import json
import os
import sys
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import HeartRiskAgent

PORT = 5000
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = HeartRiskAgent(models_dir="models")
    return agent

class HeartRiskRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == "/" or path == "/index.html":
            self.path = "/templates/index.html"
            return super().do_GET()
            
        elif path == "/benchmark" or path == "/benchmark.html":
            self.path = "/templates/benchmark.html"
            return super().do_GET()
            
        elif path == "/api/metrics":
            metrics_path = os.path.join(PROJECT_ROOT, "output", "model_comparison.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'metrics': metrics_data}).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Metrics not generated yet.'}).encode("utf-8"))
            return
            
        elif path.startswith("/output/") or path.startswith("/static/"):
            return super().do_GET()
            
        else:
            return super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
                patient_data = payload.get('patient', {})
                model_choice = payload.get('model_choice', 'ensemble')
                
                curr_agent = get_agent()
                result = curr_agent.predict(patient_data, model_choice=model_choice)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'data': result}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode("utf-8"))
            return
            
        self.send_response(404)
        self.end_headers()

def run_server():
    get_agent()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), HeartRiskRequestHandler) as httpd:
        print(f"CardioPulse AI Web Dashboard running at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()

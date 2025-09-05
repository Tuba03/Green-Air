from http.server import BaseHTTPRequestHandler
import json
import joblib
import numpy as np
import librosa
import io

# Load model at cold start
model = joblib.load("api/green_air_classifier.joblib")

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            # Load audio from request body
            y, sr = librosa.load(io.BytesIO(post_data), sr=None)

            # Extract features
            features = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0).reshape(1, -1)

            # Predict
            prediction = model.predict(features)

            # Respond
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"prediction": prediction.tolist()}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

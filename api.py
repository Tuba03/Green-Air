# api.py
import os
import tempfile
import time
import joblib
import librosa
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# --- Load env ---
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")

# --- Load ML model ---
MODEL_FILE = "green_air_classifier.joblib"
model = None
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("⚠️ Could not load model:", e)

# --- In-memory DB ---
community_data = []

# Convert to wav
def convert_to_wav(src_path, dst_path):
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio.export(dst_path, format="wav")

# Feature extraction
def extract_features(file_path, duration=3):
    y, sr = librosa.load(file_path, sr=None, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features, y, sr

# Estimate dB SPL-like
def estimate_db(y):
    rms = np.sqrt(np.mean(y**2))
    if rms <= 0:
        return 0.0
    return round(float(20 * np.log10(rms) + 94), 2)

# AQI fetch
def get_aqi(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        )
        res = requests.get(url, timeout=8)
        data = res.json()
        if res.status_code == 200 and data.get("list"):
            entry = data["list"][0]
            aqi = int(entry["main"]["aqi"])
            comp = entry["components"]
            return {
                "aqi": aqi,
                "level": {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}.get(aqi, "Unknown"),
                "components": {
                    "pm2_5": float(comp.get("pm2_5", 0)),
                    "pm10": float(comp.get("pm10", 0)),
                    "no2": float(comp.get("no2", 0)),
                    "co": float(comp.get("co", 0)),
                },
            }
        return {"error": "AQI unavailable"}
    except Exception:
        return {"error": "AQI fetch failed"}

@app.route("/analyze_sound", methods=["POST"])
def analyze_uploaded_audio():
    try:
        if "audio_file" not in request.files:  # ✅ match frontend key
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio_file"]
        lat = request.form.get("lat")
        lon = request.form.get("lon")

        # Save temp
        suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
        tmp_src = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_src.write(audio_file.read())
        tmp_src.close()

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        convert_to_wav(tmp_src.name, tmp_wav.name)

        # Extract features
        features, y, sr = extract_features(tmp_wav.name)
        noise_db = estimate_db(y)
        predicted_label = None
        if model is not None:
            try:
                predicted_label = model.predict([features])[0]
            except Exception as e:
                print("Prediction error:", e)

        # AQI lookup
        air_quality = {}
        if lat and lon:
            air_quality = get_aqi(lat, lon)

        # Entry in community
        entry = {
            "timestamp": time.time(),
            "location": {"lat": float(lat), "lon": float(lon)},
            "sound_analysis": {
                "type": str(predicted_label or "Unknown"),
                "noise_level_db": float(noise_db or 0),
            },
            "air_quality": air_quality,
        }
        community_data.append(entry)

        return jsonify(entry), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/community-data", methods=["GET"])
def get_community_data():
    limit = int(request.args.get("limit", 50))
    return jsonify({"measurements": community_data[-limit:]})

if __name__ == "__main__":
    print("🚀 Starting GreenAir backend...")
    app.run(host="0.0.0.0", port=5000, debug=True)
# """    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             if data.get('list'):
#                 aqi = data['list'][0]['main']['aqi']
#                 components = data['list'][0]['components']
#                 return {'aqi': aqi, 'components': components}
#         print(f"Failed to fetch AQI: {response.status_code} {response.text}")
#     except Exception as e:
#         print(f"Error fetching AQI: {e}")
#     return None
# # Example usage:
# # result = fetch_air_quality(19.076, 72.8777)
# # print(result)     
import os
import flask
from flask import request, jsonify
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import requests

# --- 1. Load the trained model and other necessary files ---
MODEL_FILE = 'green_air_classifier.joblib'
FEATURES_FILE = 'urban_sound_features.joblib'
AQI_API_KEY = "OPENWEATHER_API_KEY" # Replace with your actual API key

app = flask.Flask(__name__)
# Enable CORS to allow the React frontend to communicate with this backend
CORS(app)

try:
    model = joblib.load(MODEL_FILE)
    features_df = joblib.load(FEATURES_FILE)
    # Get the unique classes from the features file
    model_classes = features_df['label'].unique().tolist()
    print("Model and features loaded successfully.")
except FileNotFoundError:
    print(f"Error: Required files not found. Please run 'train.py' first.")
    model = None
    model_classes = []

# --- 2. Feature Extraction Function (re-used from train.py) ---
def extract_features(file_path):
    """Extracts MFCC features for a single audio file."""
    try:
        y, sr = librosa.load(file_path, duration=3)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# --- 3. Air Quality Index (AQI) API Integration ---
def get_aqi(lat, lon):
    """Fetches real-time AQI data from a public API."""
    try:
        # Using OpenWeatherMap's Air Pollution API as a free option
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={AQI_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and data['list']:
            aqi = data['list'][0]['main']['aqi']
            # Map AQI code to a description
            aqi_mapping = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
            return aqi, aqi_mapping.get(aqi, 'Unknown')
        else:
            return None, "Data not available"
    except Exception as e:
        print(f"Error fetching AQI data: {e}")
        return None, "API call failed"

# --- 4. The Main API Endpoint ---
@app.route('/analyze_sound', methods=['POST'])
def analyze_uploaded_audio():
    if model is None:
        return jsonify({"error": "Model not loaded. Please contact the administrator."}), 500
    
    # Check if a file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    # Save the uploaded file temporarily
    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)

    # Extract features from the temporary file
    features = extract_features(temp_path)

    if features is None:
        os.remove(temp_path)
        return jsonify({"error": "Failed to process audio file."}), 500

    # Make a prediction using the loaded model
    # The model expects a 2D array, so we reshape the features
    prediction_label = model.predict([features])[0]

    # Clean up the temporary file
    os.remove(temp_path)

    # Fetch AQI data if location is provided
    aqi, aqi_description = None, "Location data not provided"
    if lat and lon:
        aqi, aqi_description = get_aqi(lat, lon)
    
    # Return the analysis results as a JSON response
    return jsonify({
        'sound_type': prediction_label,
        'noise_level_estimate_db': '70-90 dB', # Placeholder, requires a more complex model
        'air_quality_index': aqi,
        'air_quality_description': aqi_description,
        'recommendation': 'Based on this data, consider using public transport to reduce both noise and air pollution.',
        'message': 'Analysis successful.'
    })

# --- 5. Run the Flask app ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
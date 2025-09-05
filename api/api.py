import os
import tempfile
import joblib
import librosa
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
AQI_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# Global variables for model (will be loaded when needed)
model = None
model_classes = []

def load_model():
    """Load the trained model and features if not already loaded."""
    global model, model_classes
    if model is None:
        try:
            # In Vercel, you might need to adjust paths or load from external storage
            # For now, this assumes the model files are in the same directory
            model = joblib.load('green_air_classifier.joblib')
            features_df = joblib.load('urban_sound_features.joblib')
            model_classes = features_df['label'].unique().tolist()
            print("Model and features loaded successfully.")
        except FileNotFoundError:
            print("Error: Model files not found.")
            return False
    return True

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

def get_aqi(lat, lon):
    """Fetches real-time AQI data from OpenWeatherMap API."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={AQI_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and data['list']:
            aqi = data['list'][0]['main']['aqi']
            aqi_mapping = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
            return aqi, aqi_mapping.get(aqi, 'Unknown')
        else:
            return None, "Data not available"
    except Exception as e:
        print(f"Error fetching AQI data: {e}")
        return None, "API call failed"

@app.route('/')
def home():
    """Health check endpoint."""
    return jsonify({"message": "Green Air API is running", "status": "healthy"})

@app.route('/analyze_sound', methods=['POST'])
def analyze_uploaded_audio():
    """Main endpoint for audio analysis."""
    # Load model if not already loaded
    if not load_model():
        return jsonify({"error": "Model not available. Please contact the administrator."}), 500
    
    # Check if a file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_path = temp_file.name
        audio_file.save(temp_path)

    try:
        # Extract features from the temporary file
        features = extract_features(temp_path)

        if features is None:
            return jsonify({"error": "Failed to process audio file."}), 500

        # Make a prediction using the loaded model
        prediction_label = model.predict([features])[0]

        # Fetch AQI data if location is provided
        aqi, aqi_description = None, "Location data not provided"
        if lat and lon:
            aqi, aqi_description = get_aqi(lat, lon)
        
        # Return the analysis results as a JSON response
        return jsonify({
            'sound_type': prediction_label,
            'noise_level_estimate_db': '70-90 dB',  # Placeholder
            'air_quality_index': aqi,
            'air_quality_description': aqi_description,
            'recommendation': 'Based on this data, consider using public transport to reduce both noise and air pollution.',
            'message': 'Analysis successful.'
        })

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# This is the main handler for Vercel
def handler(request):
    """Vercel serverless function handler."""
    return app(request.environ, start_response)

# For local development
if __name__ == '__main__':
    app.run(debug=True, port=5000)
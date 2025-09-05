import os
import tempfile
import joblib
import librosa
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
from io import BytesIO

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
AQI_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# External storage URLs - replace with your actual URLs
MODEL_URL = os.environ.get("MODEL_URL", "")  # e.g., S3, Google Drive, GitHub releases
FEATURES_URL = os.environ.get("FEATURES_URL", "")

# Global variables for model - lazy loading
model = None
model_classes = []
model_loaded = False
model_load_attempted = False

def download_file_from_url(url, local_path):
    """Download a file from URL to local path."""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, local_path)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def load_model_from_external():
    """Load model from external storage (S3, Google Drive, etc.)."""
    global model, model_classes, model_loaded, model_load_attempted
    
    if model_load_attempted:
        return model is not None
    
    model_load_attempted = True
    
    try:
        if not MODEL_URL or not FEATURES_URL:
            print("External storage URLs not configured")
            return False
        
        # Create temp directory for model files
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model.joblib")
        features_path = os.path.join(temp_dir, "features.joblib")
        
        # Download model files
        if download_file_from_url(MODEL_URL, model_path) and \
           download_file_from_url(FEATURES_URL, features_path):
            
            # Load the downloaded files
            model = joblib.load(model_path)
            features_df = joblib.load(features_path)
            model_classes = features_df['label'].unique().tolist()
            
            print("Model loaded successfully from external storage.")
            model_loaded = True
            
            # Clean up temp files
            os.remove(model_path)
            os.remove(features_path)
            os.rmdir(temp_dir)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error loading model from external storage: {e}")
        return False

def load_local_model():
    """Try to load model from local files (for testing)."""
    global model, model_classes, model_loaded
    
    try:
        current_dir = os.path.dirname(__file__)
        model_file = os.path.join(current_dir, 'green_air_classifier.joblib')
        features_file = os.path.join(current_dir, 'urban_sound_features.joblib')
        
        if os.path.exists(model_file) and os.path.exists(features_file):
            model = joblib.load(model_file)
            features_df = joblib.load(features_file)
            model_classes = features_df['label'].unique().tolist()
            print("Model loaded from local files.")
            model_loaded = True
            return True
        else:
            print("Local model files not found.")
            return False
            
    except Exception as e:
        print(f"Error loading local model: {e}")
        return False

def load_model():
    """Load model from local files or external storage."""
    if model_loaded:
        return model is not None
    
    # Try local first, then external
    return load_local_model() or load_model_from_external()

def extract_features(file_path):
    """Extracts MFCC features for a single audio file."""
    try:
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

def get_aqi(lat, lon):
    """Fetches real-time AQI data from OpenWeatherMap API."""
    if not AQI_API_KEY:
        return None, "API key not configured"
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={AQI_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200 and 'list' in data and data['list']:
            aqi = data['list'][0]['main']['aqi']
            aqi_mapping = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
            return aqi, aqi_mapping.get(aqi, 'Unknown')
        else:
            return None, "Data not available"
    except Exception as e:
        print(f"Error fetching AQI data: {e}")
        return None, "API call failed"

@app.route('/')
@app.route('/api')
def home():
    """Health check endpoint."""
    return jsonify({
        "message": "Green Air API is running", 
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_available": model is not None,
        "external_urls_configured": bool(MODEL_URL and FEATURES_URL)
    })

@app.route('/api/analyze_sound', methods=['POST', 'OPTIONS'])
def analyze_uploaded_audio():
    """Main endpoint for audio analysis."""
    if request.method == 'OPTIONS':
        return '', 200
    
    # Load model if not already loaded
    if not load_model():
        return jsonify({
            "error": "Model not available",
            "details": "Configure MODEL_URL and FEATURES_URL environment variables for external storage",
            "help": "Set these to public URLs of your joblib files (S3, Google Drive, GitHub releases, etc.)"
        }), 500
    
    # Check if a file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    
    # Check file size
    if audio_file.content_length and audio_file.content_length > 4 * 1024 * 1024:
        return jsonify({"error": "Audio file too large. Maximum size is 4MB."}), 413
    
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)

        # Extract features
        features = extract_features(temp_path)
        if features is None:
            return jsonify({"error": "Failed to process audio file."}), 500

        # Make prediction
        prediction_label = model.predict([features])[0]
        prediction_proba = model.predict_proba([features])[0]
        confidence = float(max(prediction_proba))

        # Get AQI data
        aqi, aqi_description = None, "Location data not provided"
        if lat and lon:
            try:
                aqi, aqi_description = get_aqi(float(lat), float(lon))
            except ValueError:
                aqi_description = "Invalid location coordinates"
        
        # Generate recommendation
        recommendation = generate_recommendation(prediction_label, aqi)
        
        return jsonify({
            'sound_type': prediction_label,
            'confidence': round(confidence * 100, 2),
            'noise_level_estimate_db': estimate_noise_level(prediction_label),
            'air_quality_index': aqi,
            'air_quality_description': aqi_description,
            'recommendation': recommendation,
            'message': 'Analysis successful.'
        })

    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def estimate_noise_level(sound_type):
    """Estimate noise level based on sound type."""
    noise_levels = {
        'air_conditioner': '50-60 dB',
        'car_horn': '100-110 dB', 
        'children_playing': '60-70 dB',
        'dog_bark': '80-90 dB',
        'drilling': '85-100 dB',
        'engine_idling': '40-50 dB',
        'gun_shot': '140+ dB',
        'jackhammer': '100-110 dB',
        'siren': '100-120 dB',
        'street_music': '70-85 dB'
    }
    return noise_levels.get(sound_type, '70-90 dB')

def generate_recommendation(sound_type, aqi):
    """Generate recommendation based on analysis."""
    high_noise_sources = ['car_horn', 'drilling', 'gun_shot', 'jackhammer', 'siren']
    
    if sound_type in high_noise_sources:
        base_rec = "High noise levels detected. Consider ear protection."
    else:
        base_rec = "Moderate noise levels detected."
    
    if aqi and aqi >= 4:
        return base_rec + " Air quality is poor - consider staying indoors."
    elif aqi and aqi >= 3:
        return base_rec + " Air quality is moderate - limit outdoor activities."
    else:
        return base_rec + " Air quality is acceptable."

# WSGI handler for Vercel
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
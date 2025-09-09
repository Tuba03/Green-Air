import pandas as pd
import librosa
import numpy as np

import os
from tqdm import tqdm
import joblib
import requests


# --- 1. Configuration ---
# File paths for the dataset
CSV_FILE = 'UrbanSound8K.csv'
AUDIO_DIR = 'audio'
FEATURES_FILE = 'urban_sound_features.joblib'
MODEL_FILE = 'green_air_classifier.joblib'

# OpenWeatherMap API configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'YOUR_API_KEY_HERE')  # Set your API key in env or replace

# --- Air Quality Fetch Function ---
def fetch_air_quality(lat, lon):
    """
    Fetches air quality data from OpenWeatherMap API for the given latitude and longitude.
    Returns a dict with AQI and components, or None if failed.
    """
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('list'):
                aqi = data['list'][0]['main']['aqi']
                components = data['list'][0]['components']
                return {'aqi': aqi, 'components': components}
        print(f"Failed to fetch AQI: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error fetching AQI: {e}")
    return None

# Example usage:
# result = fetch_air_quality(19.076, 72.8777)
# print(result)

# --- 2. Feature Extraction Function ---
def extract_features(file_path, duration=3):
    """
    Extracts MFCC features from an audio file.
    
    Args:
        file_path (str): The full path to the audio file.
        duration (int): The number of seconds to process from the start of the file.
    
    Returns:
        np.array: A 1D array of mean MFCCs, or None if an error occurs.
    """
    try:
        # Load the audio file, resample, and limit to the specified duration
        y, sr = librosa.load(file_path, duration=duration)
        
        # Extract Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Take the mean of the MFCCs to get a single feature vector
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 3. Main Feature Extraction and Saving Logic ---
def process_and_save_features():
    """
    Reads the metadata, processes all audio files, and saves the features.
    """
    if os.path.exists(FEATURES_FILE):
        print(f"Features file '{FEATURES_FILE}' already exists. Loading it.")
        features_df = joblib.load(FEATURES_FILE)
        return features_df

    print("Starting feature extraction. This may take a while...")
    
    # Load the metadata CSV
    metadata = pd.read_csv(CSV_FILE)
    
    all_features = []
    
    # Iterate over each row in the metadata DataFrame with a progress bar
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting Features"):
        file_name = row['slice_file_name']
        fold_num = row['fold']
        label = row['class']
        
        # Construct the full path to the audio file
        file_path = os.path.join(AUDIO_DIR, f"fold{fold_num}", file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            continue
        
        # Extract features and append to our list
        data = extract_features(file_path)
        if data is not None:
            all_features.append({'features': data, 'label': label})
    
    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save the DataFrame to a file using joblib
    joblib.dump(features_df, FEATURES_FILE)
    
    print(f"\nFeature extraction complete. Saved to '{FEATURES_FILE}'.")
    return features_df

# --- 4. Main Training and Saving Logic ---
def train_and_save_model(features_df):
    """
    Trains a classifier and saves it to a file.
    """
    # Import scikit-learn modules here to keep them scoped
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    print("Starting model training...")

    # Split the DataFrame into features (X) and labels (y)
    X = np.array(features_df['features'].tolist())
    y = np.array(features_df['label'].tolist())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel training complete.")
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved as '{MODEL_FILE}'.")

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # First, process the data or load the existing features
    features_df = process_and_save_features()
    
    if features_df is not None and not features_df.empty:
        # Then, train the model
        train_and_save_model(features_df)
    else:
        print("No features were extracted. Cannot train the model.")
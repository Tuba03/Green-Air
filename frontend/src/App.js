import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [location, setLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Function to handle file selection from the user's computer
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // Function to get the user's location using the browser's Geolocation API
  const getLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
          });
          console.log("Location found:", position.coords.latitude, position.coords.longitude);
        },
        (err) => {
          console.error("Location error:", err);
          alert("Unable to retrieve your location. Analysis will be sound-only.");
          setLocation(null);
        }
      );
    } else {
      alert("Geolocation is not supported by your browser.");
      setLocation(null);
    }
  };

  // Function to handle the form submission and send data to the Flask API
  const handleUpload = async () => {
    setError(null);
    setResults(null);
    if (!selectedFile) {
      setError("Please select an audio file first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('audio_file', selectedFile);
    if (location) {
      formData.append('lat', location.lat);
      formData.append('lon', location.lon);
    }

    try {
      // Connect to your Flask API. Make sure the Flask API is running!
      const response = await fetch('http://127.0.0.1:5000/analyze_sound', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to connect to the server or process the file.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Render the UI
  return (
    <div className="App">
      <header className="App-header">
        <h1>GreenAir - Urban Sound & Air Quality Analysis</h1>
        <p>Upload an audio file to classify the sound and check the local air quality.</p>

        <div className="controls">
          <input type="file" onChange={handleFileChange} accept="audio/*" />
          <button onClick={getLocation} className="location-btn">
            Get My Location
          </button>
          <button onClick={handleUpload} disabled={loading} className="analyze-btn">
            {loading ? 'Analyzing...' : 'Analyze Audio'}
          </button>
        </div>

        {location && (
          <div className="location-info">
            <p>Location found: Lat {location.lat.toFixed(4)}, Lon {location.lon.toFixed(4)}</p>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {results && (
          <div className="results-box">
            <h2>Analysis Results</h2>
            <p><strong>Sound Type:</strong> {results.sound_type}</p>
            <p><strong>Noise Level:</strong> {results.noise_level_estimate_db}</p>
            <p><strong>Air Quality:</strong> {results.air_quality_description} ({results.air_quality_index})</p>
            <p className="recommendation"><strong>Recommendation:</strong> {results.recommendation}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
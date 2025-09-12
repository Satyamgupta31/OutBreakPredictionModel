import logging
from flask import Flask, jsonify, request
import os

# Assuming ml_pipeline.py is in the same directory
try:
    from ml_pipeline import OutbreakPredictionAPI
except ImportError:
    # Handle the case where the file isn't found
    logging.error("Could not import OutbreakPredictionAPI. Make sure ml_pipeline.py is in the same directory.")
    OutbreakPredictionAPI = None

# Configure logging for the Flask application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the Outbreak Prediction System
# This will be done once when the app starts
api = None
try:
    # Use a local MongoDB connection string
    mongodb_url = "mongodb://localhost:27017/"
    api = OutbreakPredictionAPI(mongodb_url)
    init_status = api.initialize_system()
    if init_status['status'] == 'success':
        logger.info("API initialized successfully: " + init_status['message'])
    else:
        logger.error("Failed to initialize API: " + init_status['message'])
        api = None
except Exception as e:
    logger.error(f"An error occurred during API initialization: {e}")
    api = None


@app.route('/')
def home():
    """Health check endpoint to ensure the API is running."""
    return "Outbreak Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict_outbreak():
    """
    Endpoint to get a single outbreak prediction based on input data.
    
    Expected JSON payload:
    {
        "village_id": "VILLAGE_001",
        "date": "2025-09-12",
        "diarrhea_cases": 5,
        "vomiting_cases": 3,
        "fever_cases": 8,
        "ph_level": 6.2,
        "turbidity": 4.5,
        "tds": 450,
        "rainfall": 15.2,
        "temperature": 32.5,
        "humidity": 75,
        "season": "monsoon"
    }
    """
    if api is None:
        return jsonify({"error": "API is not initialized. Check server logs."}), 500
        
    try:
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Call the prediction method from the OutbreakPredictionAPI
        prediction_result = api.predict_single(data)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # To run this in a production environment, use a WSGI server like Gunicorn or Waitress.
    # For local testing, you can run it directly.
    app.run(host='0.0.0.0', port=5000)

# app.py
import logging
import os
from flask import Flask, jsonify, request

# ========================
# IMPORT ML PIPELINE
# ========================
try:
    from ml_pipeline import OutbreakPredictionAPI
except ImportError:
    logging.error("Could not import OutbreakPredictionAPI. Ensure ml_pipeline.py is in the same directory.")
    OutbreakPredictionAPI = None

# ========================
# IMPORT CHATBOT
# ========================
try:
    from chatbot import ask_chatbot
except ImportError:
    logging.error("Could not import ask_chatbot. Ensure chatbot.py is in the same directory.")
    ask_chatbot = None

# ========================
# CONFIGURE LOGGING
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# INITIALIZE FLASK APP
# ========================
app = Flask(__name__)

# ========================
# INITIALIZE OUTBREAK API
# ========================
api = None
try:
    mongodb_url = "mongodb://localhost:27017/"
    api = OutbreakPredictionAPI(mongodb_url)
    init_status = api.initialize_system()
    if init_status['status'] == 'success':
        logger.info("✅ Outbreak Prediction API initialized successfully")
    else:
        logger.error("❌ Failed to initialize API: " + init_status['message'])
        api = None
except Exception as e:
    logger.error(f"❌ Error initializing OutbreakPredictionAPI: {e}")
    api = None

# ========================
# HEALTH CHECK
# ========================
@app.route('/')
def home():
    return "🚑 Outbreak Prediction & Chatbot API is running!"

# ========================
# OUTBREAK PREDICTION ENDPOINT
# ========================
@app.route('/predict', methods=['POST'])
def predict_outbreak():
    if api is None:
        return jsonify({"error": "ML API is not initialized. Check server logs."}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        prediction_result = api.predict_single(data)
        return jsonify(prediction_result)

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================
# CHATBOT ENDPOINT
# ========================
@app.route('/chat', methods=['POST'])
def chat():
    if ask_chatbot is None:
        return jsonify({"error": "Chatbot function is not available. Check server logs."}), 500

    try:
        data = request.get_json(force=True)
        user_type = data.get("user_type")
        village_id = data.get("village_id")
        user_query = data.get("user_query")

        if not all([user_type, village_id, user_query]):
            return jsonify({"error": "Missing required fields: user_type, village_id, user_query"}), 400

        # Generate chatbot response
        response = ask_chatbot(user_type=user_type, village_id=village_id, user_query=user_query)
        return jsonify({"reply": response})

    except Exception as e:
        logger.error(f"❌ Chatbot error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================
# RUN FLASK APP
# ========================
if __name__ == '__main__':
    # For production, use a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000)

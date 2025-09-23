# app.py - Updated with Windows compatibility
import logging
import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS

# Fix Windows encoding issues
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Import ML pipeline
try:
    from ml_pipeline import OutbreakPredictionAPI
except ImportError:
    logger.error("Could not import OutbreakPredictionAPI. Ensure ml_pipeline.py is in the same directory.")
    OutbreakPredictionAPI = None

# Import multilingual chatbot
try:
    from multilingual_chatbot import ask_multilingual_chatbot, get_supported_languages, MultilingualHealthChatbot
except ImportError:
    logger.error("Could not import multilingual chatbot. Ensure multilingual_chatbot.py is in the same directory.")
    ask_multilingual_chatbot = None
    get_supported_languages = None
    MultilingualHealthChatbot = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize systems
api = None
multilingual_chatbot = None

# Initialize Outbreak Prediction API
try:
    mongodb_url = "mongodb://localhost:27017/"
    api = OutbreakPredictionAPI(mongodb_url)
    init_status = api.initialize_system()
    if init_status['status'] == 'success':
        logger.info("Outbreak Prediction API initialized successfully")
    else:
        logger.error(f"Failed to initialize API: {init_status['message']}")
        api = None
except Exception as e:
    logger.error(f"Error initializing OutbreakPredictionAPI: {e}")
    api = None

# Initialize Multilingual Chatbot
try:
    if MultilingualHealthChatbot:
        multilingual_chatbot = MultilingualHealthChatbot()
        logger.info("Multilingual Health Chatbot initialized successfully")
    else:
        logger.error("Multilingual chatbot not available")
except Exception as e:
    logger.error(f"Error initializing Multilingual Chatbot: {e}")
    multilingual_chatbot = None

# Health check
@app.route('/')
def home():
    return jsonify({
        "message": "Multilingual Outbreak Prediction & Chatbot API",
        "status": "running",
        "services": {
            "ml_prediction": api is not None,
            "multilingual_chat": multilingual_chatbot is not None
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "services": {
            "outbreak_prediction": {
                "available": api is not None,
                "status": "ready" if api else "unavailable"
            },
            "multilingual_chatbot": {
                "available": multilingual_chatbot is not None,
                "status": "ready" if multilingual_chatbot else "unavailable",
                "supported_languages": len(get_supported_languages()) if get_supported_languages else 0
            }
        }
    })

# Language support endpoints
@app.route('/languages', methods=['GET'])
def get_languages():
    if not get_supported_languages:
        return jsonify({"error": "Language support not available"}), 500
    
    try:
        languages = get_supported_languages()
        return jsonify({
            "supported_languages": languages,
            "total_count": len(languages)
        })
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect-language', methods=['POST'])
def detect_language():
    if not multilingual_chatbot:
        return jsonify({"error": "Multilingual chatbot not available"}), 500
    
    try:
        data = request.get_json(force=True)
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        detected_lang = multilingual_chatbot.detect_language(text)
        language_info = multilingual_chatbot.supported_languages.get(detected_lang)
        
        return jsonify({
            "detected_language": detected_lang,
            "language_name": language_info['name'] if language_info else "Unknown",
            "local_name": language_info['local_name'] if language_info else "Unknown",
            "confidence": "auto-detected"
        })
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return jsonify({"error": str(e)}), 500

# Outbreak prediction endpoints
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
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    if api is None:
        return jsonify({"error": "ML API is not initialized"}), 500

    try:
        data = request.get_json(force=True)
        villages = data.get("villages")
        date = data.get("date")
        
        predictions = api.predict_batch(villages, date)
        
        return jsonify({
            "predictions": predictions,
            "total_count": len(predictions),
            "high_risk_count": sum(1 for p in predictions if p.get('risk_level') == 'HIGH'),
            "medium_risk_count": sum(1 for p in predictions if p.get('risk_level') == 'MEDIUM'),
            "low_risk_count": sum(1 for p in predictions if p.get('risk_level') == 'LOW')
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# Multilingual chatbot endpoints
@app.route('/chat', methods=['POST'])
def chat():
    if not multilingual_chatbot:
        return jsonify({"error": "Multilingual chatbot is not available. Check server logs."}), 500

    try:
        data = request.get_json(force=True)
        user_type = data.get("user_type")
        village_id = data.get("village_id")
        user_query = data.get("user_query")
        preferred_language = data.get("preferred_language")

        if not all([user_type, village_id, user_query]):
            return jsonify({
                "error": "Missing required fields: user_type, village_id, user_query"
            }), 400

        # Log the request for debugging
        logger.info(f"Chat request: user_type={user_type}, village_id={village_id}, language={preferred_language}")

        response = multilingual_chatbot.generate_response(
            user_type=user_type,
            village_id=village_id,
            user_query=user_query,
            preferred_language=preferred_language
        )

        # Ensure all required fields are present
        result = {
            "response": response.get('response', 'Sorry, no response generated.'),
            "detected_language": response.get('detected_language', 'en'),
            "language_name": response.get('language_name', 'English'),
            "has_local_data": response.get('has_local_data', False),
            "user_query": user_query,
            "village_id": village_id,
            "user_type": user_type,
            "status": response.get('status', 'success')
        }

        # Add error field if present
        if 'error' in response:
            result['error_details'] = response['error']

        logger.info(f"Chat response: language={result['language_name']}, status={result['status']}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "response": "I'm sorry, I encountered an error. Please try again.",
            "detected_language": "en",
            "language_name": "English",
            "has_local_data": False,
            "status": "error"
        }), 500

@app.route('/chat/translate', methods=['POST'])
def translate_text():
    if not multilingual_chatbot:
        return jsonify({"error": "Translation service not available"}), 500

    try:
        data = request.get_json(force=True)
        text = data.get("text")
        target_language = data.get("target_language")
        source_language = data.get("source_language", "auto")

        if not text or not target_language:
            return jsonify({
                "error": "Missing required fields: text, target_language"
            }), 400

        translated_text = multilingual_chatbot.translate_text(
            text=text,
            target_lang=target_language,
            source_lang=source_language
        )

        return jsonify({
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language
        })

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/village/<village_id>/data', methods=['GET'])
def get_village_data(village_id):
    if not multilingual_chatbot:
        return jsonify({"error": "Service not available"}), 500

    try:
        village_data = multilingual_chatbot.get_village_data(village_id)
        
        if not village_data:
            return jsonify({
                "message": "No data found for this village",
                "village_id": village_id,
                "data": None
            }), 404
        
        return jsonify({
            "village_id": village_id,
            "data": village_data,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Village data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test/multilingual', methods=['POST'])
def test_multilingual():
    if not multilingual_chatbot:
        return jsonify({"error": "Multilingual chatbot not available"}), 500

    try:
        test_queries = [
            {
                'user_type': 'villager',
                'village_id': 'VILLAGE_001',
                'query': 'আমার জ্বর আছে, কি করবো?',
                'language': 'Bengali'
            },
            {
                'user_type': 'asha_worker',
                'village_id': 'VILLAGE_002',
                'query': 'How to prevent diarrhea in children?',
                'language': 'English'
            },
            {
                'user_type': 'villager',
                'village_id': 'VILLAGE_003',
                'query': 'पानी साफ करने का तरीका क्या है?',
                'language': 'Hindi'
            }
        ]

        results = []
        for test in test_queries:
            result = multilingual_chatbot.generate_response(
                user_type=test['user_type'],
                village_id=test['village_id'],
                user_query=test['query']
            )
            
            results.append({
                'test_query': test['query'],
                'expected_language': test['language'],
                'detected_language': result['detected_language'],
                'language_name': result['language_name'],
                'response_preview': result['response'][:100] + '...',
                'success': 'error' not in result
            })

        return jsonify({
            "test_results": results,
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r['success'])
        })

    except Exception as e:
        logger.error(f"Multilingual test error: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /",
            "GET /health",
            "GET /languages",
            "POST /detect-language",
            "POST /predict",
            "POST /predict/batch",
            "POST /chat",
            "POST /chat/translate",
            "GET /village/<village_id>/data",
            "POST /test/multilingual"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500

if __name__ == '__main__':
    print("Starting Multilingual Health Monitoring API...")
    print("Supported Languages:")
    
    if get_supported_languages:
        languages = get_supported_languages()
        for code, info in languages.items():
            print(f"  - {info['name']} ({info['local_name']}): {code}")
    
    print("\nAPI Endpoints:")
    print("  - Health Check: GET /health")
    print("  - Languages: GET /languages")
    print("  - Chat: POST /chat")
    print("  - Predictions: POST /predict")
    print("  - Translation: POST /chat/translate")
    print("  - Test: POST /test/multilingual")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
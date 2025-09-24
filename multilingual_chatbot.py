# multilingual_chatbot.py - Gemini-Integrated AI Mitra Health Assistant
import os
import logging
from typing import Dict, Optional
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import google.generativeai as genai
import json
import re

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "healthmonitoring")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "USE_YOUR_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified language support
SUPPORTED_LANGUAGES = {
    'hi': {
        'name': 'Hindi',
        'local_name': 'हिंदी',
        'script_chars': 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
    },
    'bn': {
        'name': 'Bengali', 
        'local_name': 'বাংলা',
        'script_chars': 'আইউঊএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ'
    },
    'as': {
        'name': 'Assamese',
        'local_name': 'অসমীয়া',
        'script_chars': 'অসমীয়াআইউঊএঐওঔকগঙচছজঝঞটঠডঢণতথদধনপফবভমযৰরলৱশষসহ'
    },
    'en': {
        'name': 'English',
        'local_name': 'English',
        'script_chars': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    }
}

class DatabaseConnection:
    """MongoDB connection handler"""
    _instance = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def get_database(self):
        """Get database connection"""
        if self._db is not None:
            return self._db
            
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
            self._db = client[DB_NAME]
            logger.info("Connected to MongoDB")
            return self._db
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return None

db_connection = DatabaseConnection()

class MultilingualHealthChatbot:
    def __init__(self):
        self.supported_languages = SUPPORTED_LANGUAGES
        self.default_language = 'en'
        
    def detect_language(self, text: str) -> str:
        """Enhanced language detection based on script analysis"""
        if not text or len(text.strip()) < 2:
            return self.default_language
            
        # Count characters for each script
        script_scores = {}
        
        for lang_code, lang_info in self.supported_languages.items():
            if lang_code == 'en':
                continue
                
            script_chars = set(lang_info['script_chars'])
            score = sum(1 for char in text if char in script_chars)
            if score > 0:
                script_scores[lang_code] = score
        
        # Return language with highest script match
        if script_scores:
            detected_lang = max(script_scores, key=script_scores.get)
            # Special handling for Assamese vs Bengali
            if detected_lang == 'bn':
                assamese_indicators = ['ৰ', 'ৱ', 'ক্ষ']
                if any(indicator in text for indicator in assamese_indicators):
                    return 'as'
            return detected_lang
        
        return self.default_language
    
    def get_village_data(self, village_id: str) -> Optional[Dict]:
        """Get comprehensive village health data"""
        try:
            db = db_connection.get_database()
            if db is None:
                return None
                
            # Get latest symptom report
            latest_symptom = db.symptom_reports.find_one(
                {"village_id": village_id},
                sort=[("date", -1)]
            )
            
            if not latest_symptom:
                return None
            
            latest_date = latest_symptom.get("date")
            
            # Get water quality data
            water_quality = db.water_quality.find_one({
                "village_id": village_id,
                "date": latest_date
            })
            
            # Get outbreak predictions
            prediction = db.outbreak_predictions.find_one({
                "village_id": village_id,
                "date": latest_date
            })
            
            # Get recent trends (last 7 entries)
            recent_reports = list(db.symptom_reports.find({
                "village_id": village_id
            }).sort("date", -1).limit(7))
            
            return {
                "village_id": village_id,
                "latest_date": latest_date,
                "current_symptoms": latest_symptom,
                "water_quality": water_quality,
                "outbreak_prediction": prediction,
                "recent_trends": recent_reports,
                "data_available": True
            }
            
        except Exception as e:
            logger.error(f"Error getting village data: {e}")
            return None
    
    def create_health_prompt(self, user_type: str, village_id: str, user_query: str, 
                            user_language: str, village_data: Optional[Dict] = None) -> str:
        """Create comprehensive health consultation prompt for Gemini"""
        
        language_info = self.supported_languages.get(user_language, self.supported_languages['en'])
        language_name = language_info['name']
        
        prompt = f"""
You are AI Mitra, a knowledgeable health assistant specializing in rural healthcare in Northeast India. You have expertise in:
- Common diseases in Northeast India (malaria, dengue, diarrhea, respiratory infections)
- Rural healthcare protocols and referral systems
- Cultural sensitivity for Northeast Indian communities
- ASHA worker guidance and government health programs

CRITICAL RESPONSE REQUIREMENTS:
- Respond ONLY in {language_name} language - no English mixed in
- Be empathetic, practical, and medically accurate
- Provide actionable advice appropriate for rural settings
- Include specific warning signs that require immediate medical attention
- Reference local healthcare resources (ASHA workers, PHCs, CHCs)
- Be culturally sensitive to Northeast Indian practices

USER CONTEXT:
- User Type: {user_type}
- Village: {village_id}
- Language: {language_name}
- Query: "{user_query}"

RESPONSE GUIDELINES:
1. Acknowledge their concern empathetically
2. Provide immediate actionable steps (2-3 specific actions)
3. Explain when to seek urgent medical care (warning signs)
4. Suggest appropriate healthcare contacts (ASHA worker, PHC, emergency)
5. Include relevant preventive advice
6. End with reassurance and support

MEDICAL FOCUS AREAS:
- Fever management and malaria/dengue recognition
- Diarrhea treatment with ORS preparation
- Respiratory symptoms and TB screening
- Maternal health and pregnancy care
- Child health and nutrition
- Water safety and hygiene practices
"""
        
        # Add user-type specific guidance
        if user_type == 'asha_worker':
            prompt += """

ASHA WORKER SPECIFIC GUIDANCE:
- Provide clinical protocols and assessment guidelines
- Include referral criteria for PHC/CHC/District Hospital
- Mention documentation and reporting requirements
- Reference government health programs and schemes
- Include medication guidance where appropriate
- Suggest community health education points
"""
        elif user_type == 'health_official':
            prompt += """

HEALTH OFFICIAL GUIDANCE:
- Include surveillance and monitoring recommendations
- Suggest population-level interventions
- Reference policy guidelines and reporting requirements
- Provide epidemiological insights
- Include outbreak investigation protocols
"""
        else:  # villager
            prompt += """

VILLAGER GUIDANCE:
- Use simple, non-medical language
- Focus on home-based care and family support
- Include cost-effective and locally available solutions
- Emphasize when professional medical care is essential
- Provide reassurance while maintaining medical accuracy
"""
        
        # Add village health data context
        if village_data and village_data.get('data_available'):
            current_symptoms = village_data.get('current_symptoms', {})
            water_quality = village_data.get('water_quality', {})
            prediction = village_data.get('outbreak_prediction', {})
            recent_trends = village_data.get('recent_trends', [])
            
            prompt += f"""

LOCAL HEALTH INTELLIGENCE (Village {village_id}):
- Latest Report Date: {village_data.get('latest_date')}
- Current Symptoms in Village: {current_symptoms}
- Water Quality Status: {water_quality}
- Disease Outbreak Risk: {prediction}
- Recent Health Trends: {len(recent_trends)} reports available

IMPORTANT: Consider this local health context when providing advice. If there are concerning patterns:
- Multiple fever cases → Consider malaria/dengue outbreak protocols
- Water quality issues → Emphasize water purification and waterborne disease prevention  
- High outbreak risk → Include community-wide preventive measures
- Recent symptom trends → Tailor advice based on local disease patterns

Mention the availability of local health data and how it informs your recommendations.
"""
        else:
            prompt += """

GENERAL HEALTH CONTEXT:
No specific local health data available. Focus on:
- Common health challenges in Northeast India
- Seasonal health risks (monsoon diseases, vector-borne illnesses)
- General preventive measures and health maintenance
- Building community health awareness
"""
        
        return prompt
    
    def generate_response(self, user_type: str, village_id: str, 
                         user_query: str, preferred_language: str = None) -> Dict[str, str]:
        """Generate intelligent multilingual health response using Gemini AI"""
        
        try:
            # Language detection and validation
            detected_language = preferred_language if preferred_language else self.detect_language(user_query)
            if detected_language not in self.supported_languages:
                detected_language = 'en'
            
            # Get village health data
            village_data = None
            try:
                village_data = self.get_village_data(village_id)
                logger.info(f"Village data retrieved for {village_id}: {village_data is not None}")
            except Exception as e:
                logger.warning(f"Could not fetch village data: {e}")
            
            # Create comprehensive health prompt
            prompt = self.create_health_prompt(
                user_type=user_type,
                village_id=village_id,
                user_query=user_query,
                user_language=detected_language,
                village_data=village_data
            )
            
            # Generate response using Gemini AI
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Configure generation settings for better healthcare responses
                generation_config = genai.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more focused medical advice
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1000,
                )
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                ai_response = response.text.strip()
                
                # Basic quality check
                if len(ai_response) < 50:
                    raise Exception("Response too short")
                
                # Check for language consistency
                if not self._check_language_consistency(ai_response, detected_language):
                    logger.warning("Language consistency issue detected")
                
            except Exception as e:
                logger.error(f"Gemini AI generation failed: {e}")
                # Fallback to basic response
                ai_response = self._get_fallback_response(user_query, detected_language)
            
            # Get language info
            language_info = self.supported_languages.get(detected_language)
            
            return {
                'response': ai_response,
                'detected_language': detected_language,
                'language_name': language_info['name'],
                'has_local_data': village_data is not None,
                'user_query': user_query,
                'village_id': village_id,
                'user_type': user_type,
                'ai_generated': True,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            
            # Comprehensive error handling
            fallback_language = preferred_language if preferred_language in self.supported_languages else 'en'
            language_info = self.supported_languages.get(fallback_language)
            
            error_response = self._get_error_response(fallback_language)
            
            return {
                'response': error_response,
                'detected_language': fallback_language,
                'language_name': language_info['name'],
                'has_local_data': False,
                'user_query': user_query,
                'village_id': village_id,
                'user_type': user_type,
                'ai_generated': False,
                'status': 'error',
                'error': str(e)
            }
    
    def _check_language_consistency(self, response: str, expected_language: str) -> bool:
        """Check if response is in expected language"""
        if expected_language == 'en':
            return True  # English responses are acceptable as fallback
        
        expected_chars = set(self.supported_languages[expected_language]['script_chars'])
        response_chars = set(response)
        
        # Check if response contains expected script characters
        overlap = len(expected_chars.intersection(response_chars))
        return overlap > 5  # Reasonable threshold
    
    def _get_fallback_response(self, query: str, language: str) -> str:
        """Generate fallback response when AI fails"""
        
        fallback_responses = {
            'hi': """AI मित्र यहाँ है आपकी सेवा में।

आपकी स्वास्थ्य संबंधी समस्या के लिए:

तत्काल सुझाव:
• यदि बुखार है तो आराम करें और पानी पिएं
• पेट की समस्या में ORS का घोल लें  
• सांस की तकलीफ में तुरंत डॉक्टर दिखाएं

⚠️ गंभीर लक्षणों में तुरंत चिकित्सा सहायता लें:
• तेज बुखार (103°F से ऊपर)
• सांस लेने में कठिनाई
• लगातार उल्टी या दस्त
• बेहोशी या चक्कर

🏥 संपर्क करें:
• आशा कार्यकर्ता
• प्राथमिक स्वास्थ्य केंद्र (PHC)
• आपातकाल: 108

स्वस्थ रहें और नियमित जांच कराते रहें।""",

            'bn': """AI মিত্র এখানে আপনার সেবায়।

আপনার স্বাস্থ্য সমস্যার জন্য:

তাৎক্ষণিক পরামর্শ:
• জ্বর হলে বিশ্রাম নিন এবং পানি পান করুন
• পেটের সমস্যায় ORS এর দ্রবণ নিন
• শ্বাসকষ্টে তুরন্ত ডাক্তার দেখান

⚠️ গুরুতর লক্ষণে তুরন্ত চিকিৎসা নিন:
• তীব্র জ্বর (১০৩°ফা এর বেশি)
• শ্বাস নিতে কষ্ট
• ক্রমাগত বমি বা ডায়রিয়া
• অজ্ঞান হওয়া বা চক্কর

🏥 যোগাযোগ করুন:
• আশা কর্মী
• প্রাথমিক স্বাস্থ্য কেন্দ্র
• জরুরি: ১০৮

সুস্থ থাকুন এবং নিয়মিত চেকআপ করান।""",

            'as': """AI মিত্র ইয়াত আছে আপোনাৰ সেৱাত।

আপোনাৰ স্বাস্থ্য সমস্যাৰ বাবে:

তাৎক্ষণিক পৰামৰ্শ:
• জ্বৰ হ'লে বিশ্ৰাম লওক আৰু পানী খাওক
• পেটৰ সমস্যাত ORS ৰ দ্ৰৱণ লওক
• উশাহত কষ্ট হ'লে তৎক্ষণাৎ ডাক্তৰক দেখুৱাওক

⚠️ গুৰুতৰ লক্ষণত তৎক্ষণাৎ চিকিৎসা লওক:
• তীব্ৰ জ্বৰ (১০৩°ফাৰ বেছি)
• উশাহ লোৱাত কষ্ট
• অবিৰাম বমি বা ডায়েৰিয়া
• অজ্ঞান হোৱা বা মূৰ ঘূৰোৱা

🏥 যোগাযোগ কৰক:
• আশা কৰ্মী
• প্ৰাথমিক স্বাস্থ্য কেন্দ্ৰ
• জৰুৰীকালীন: ১০৮

সুস্থ থাকক আৰু নিয়মিত পৰীক্ষা কৰাওক।""",

            'en': """AI Mitra is here to assist you.

For your health concern:

Immediate Advice:
• If fever - rest and drink plenty of water
• For stomach issues - take ORS solution
• For breathing difficulties - see doctor immediately

⚠️ Seek immediate medical help for severe symptoms:
• High fever (above 103°F)
• Difficulty breathing
• Continuous vomiting or diarrhea
• Unconsciousness or dizziness

🏥 Contact:
• ASHA Worker
• Primary Health Center (PHC)
• Emergency: 108

Stay healthy and get regular check-ups."""
        }
        
        return fallback_responses.get(language, fallback_responses['en'])
    
    def _get_error_response(self, language: str) -> str:
        """Get error response in appropriate language"""
        
        error_responses = {
            'hi': 'क्षमा करें, तकनीकी समस्या हो रही है। कृपया फिर से कोशिश करें। आपातकाल में 108 पर कॉल करें।',
            'bn': 'দুঃখিত, প্রযুক্তিগত সমস্যা হচ্ছে। অনুগ্রহ করে আবার চেষ্টা করুন। জরুরি অবস্থায় ১০৮ নম্বরে কল করুন।',
            'as': 'দুঃখিত, প্ৰযুক্তিগত সমস্যা হৈছে। অনুগ্ৰহ কৰি আকৌ চেষ্টা কৰক। জৰুৰীকালীন অৱস্থাত ১০৮ নম্বৰত কল কৰক।',
            'en': 'Sorry, technical issue occurred. Please try again. For emergencies, call 108.'
        }
        
        return error_responses.get(language, error_responses['en'])

# Convenience functions
def ask_multilingual_chatbot(user_type: str, village_id: str, 
                           user_query: str, preferred_language: str = None) -> Dict[str, str]:
    """Main function to interact with AI Mitra"""
    chatbot = MultilingualHealthChatbot()
    return chatbot.generate_response(user_type, village_id, user_query, preferred_language)

def get_supported_languages() -> Dict[str, Dict[str, str]]:
    """Get supported languages information"""
    return {
        code: {
            'name': info['name'],
            'local_name': info['local_name']
        }
        for code, info in SUPPORTED_LANGUAGES.items()
    }

if __name__ == "__main__":
    print("🤖 AI Mitra - Multilingual Health Assistant")
    print("=" * 50)
    
    chatbot = MultilingualHealthChatbot()
    
    # Test with various health queries
    test_cases = [
        {
            'user_type': 'villager',
            'village_id': 'VILLAGE_001',
            'query': 'मुझे बुखार और सिरदर्द है',
            'language': 'hi'
        },
        {
            'user_type': 'villager',
            'village_id': 'VILLAGE_002', 
            'query': 'আমার বাচ্চার ডায়রিয়া হয়েছে',
            'language': 'bn'
        },
        {
            'user_type': 'asha_worker',
            'village_id': 'VILLAGE_003',
            'query': 'Multiple fever cases reported in village',
            'language': 'en'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['query']}")
        result = chatbot.generate_response(
            test['user_type'], 
            test['village_id'], 
            test['query'],
            test['language']
        )
        print(f"Language: {result['language_name']}")
        print(f"AI Generated: {result['ai_generated']}")
        print(f"Status: {result['status']}")
        print(f"Has Local Data: {result['has_local_data']}")
        print(f"Response length: {len(result['response'])} characters")
        print(f"Preview: {result['response'][:100]}...")
    
    print("\n✅ AI Mitra ready for deployment!")

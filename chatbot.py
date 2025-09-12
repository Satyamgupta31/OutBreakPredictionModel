import os
from pymongo import MongoClient
import google.generativeai as genai

# ========================
# CONFIGURATION
# ========================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "outbreak_monitoring")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAoxaxF3PysMG3YIA4KTecD305at1IQl3o")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ========================
# MONGO CONNECTION
# ========================
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    client.admin.command('ping')
    print(f"✅ Connected to MongoDB database: {DB_NAME}")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# ========================
# FETCH LATEST DATA
# ========================
def get_village_data(village_id: str):
    try:
        latest_symptom = db.symptom_reports.find_one(
            {"village_id": village_id}, sort=[("date", -1)]
        )
        if not latest_symptom:
            return None

        latest_date = latest_symptom["date"]

        symptoms = latest_symptom
        water = db.water_quality.find_one({"village_id": village_id, "date": latest_date})
        prediction = db.outbreak_predictions.find_one({"village_id": village_id, "date": latest_date})

        return {
            "date": latest_date,
            "symptoms": symptoms or {},
            "water_quality": water or {},
            "prediction": prediction or {}
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ========================
# GENERATE RESPONSE
# ========================
def ask_chatbot(user_type: str, village_id: str, user_query: str):
    data = get_village_data(village_id)

    if data:
        prompt = f"""
        You are a helpful chatbot for rural health monitoring. 
        User: {user_type}
        Village: {village_id}, Date: {data['date']}

        Symptom reports: {data['symptoms']}
        Water quality: {data['water_quality']}
        Outbreak prediction: {data['prediction']}

        Question: "{user_query}"
        """
    else:
        prompt = f"""
        The user is a {user_type} from a rural village asking:
        "{user_query}"

        No local health or water data is available. 
        Please answer helpfully using general knowledge about safe water, hygiene, and disease prevention.
        """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ instead of gemini-pro
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"

# ========================
# SIMPLE INTERFACE
# ========================
def main():
    print("Welcome to the Village Health Chatbot!")
    user_type = input("Are you a villager or ASHA worker? ").strip().lower()
    village_id = input("Enter your village ID (e.g., VILLAGE_001): ").strip()

    while True:
        user_query = input(f"{user_type.title()}: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye! Stay safe.")
            break

        reply = ask_chatbot(user_type, village_id, user_query)
        print(f"Chatbot: {reply}\n")


if __name__ == "__main__":
    main()

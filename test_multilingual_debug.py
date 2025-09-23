# test_multilingual_debug.py
import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:5000"
    
    print("=== Testing Multilingual Health Chatbot API ===\n")
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Get Languages
    print("2. Testing Languages Endpoint...")
    try:
        response = requests.get(f"{base_url}/languages")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Supported Languages:")
            for code, info in data['supported_languages'].items():
                print(f"  {code}: {info['name']} ({info['local_name']})")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Language Detection
    print("3. Testing Language Detection...")
    test_texts = [
        "Hello, how are you?",
        "আমার জ্বর আছে",
        "मुझे बुखार है",
        "মোৰ জ্বৰ আছে"
    ]
    
    for text in test_texts:
        try:
            response = requests.post(f"{base_url}/detect-language", 
                                   json={"text": text})
            print(f"Text: '{text}'")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Detected: {result['detected_language']} ({result['language_name']})")
            else:
                print(f"Error: {response.text}")
            print()
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Chat Endpoint
    print("4. Testing Chat Endpoint...")
    test_queries = [
        {
            "user_type": "asha_worker",
            "village_id": "VILLAGE_001", 
            "user_query": "hi",
            "preferred_language": "en"
        },
        {
            "user_type": "villager",
            "village_id": "VILLAGE_001",
            "user_query": "আমার জ্বর আছে, কি করবো?",
            "preferred_language": "bn"
        },
        {
            "user_type": "asha_worker",
            "village_id": "VILLAGE_001",
            "user_query": "हमारी तब्यत खराब है",
            "preferred_language": "hi"
        }
    ]
    
    for i, query_data in enumerate(test_queries, 1):
        print(f"Test Chat {i}:")
        print(f"Query: '{query_data['user_query']}'")
        print(f"Language: {query_data['preferred_language']}")
        
        try:
            response = requests.post(f"{base_url}/chat", json=query_data)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response Language: {result.get('language_name', 'Unknown')}")
                print(f"Response: {result.get('response', 'No response')[:100]}...")
                print(f"Has Local Data: {result.get('has_local_data', False)}")
            else:
                print(f"Error Response: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "-"*30 + "\n")
    
    print("=== Testing Complete ===")

def test_direct_chatbot():
    """Test the multilingual chatbot directly without Flask"""
    print("=== Testing Direct Multilingual Chatbot ===\n")
    
    try:
        from multilingual_chatbot import MultilingualHealthChatbot
        
        chatbot = MultilingualHealthChatbot()
        
        test_queries = [
            {
                'user_type': 'asha_worker',
                'village_id': 'VILLAGE_001',
                'query': 'hi',
                'language': 'en'
            },
            {
                'user_type': 'villager',
                'village_id': 'VILLAGE_001',
                'query': 'আমার জ্বর আছে',
                'language': 'bn'
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"Direct Test {i}:")
            print(f"Query: '{test['query']}'")
            
            try:
                result = chatbot.generate_response(
                    user_type=test['user_type'],
                    village_id=test['village_id'],
                    user_query=test['query'],
                    preferred_language=test['language']
                )
                
                print("Result keys:", list(result.keys()))
                print(f"Language: {result.get('language_name', 'Missing')}")
                print(f"Response: {result.get('response', 'Missing')[:100]}...")
                print(f"Status: {result.get('status', 'Missing')}")
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                    
            except Exception as e:
                print(f"Direct test error: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "-"*30 + "\n")
            
    except ImportError as e:
        print(f"Cannot import multilingual_chatbot: {e}")
    except Exception as e:
        print(f"Direct test failed: {e}")

if __name__ == "__main__":
    # Test direct chatbot first
    test_direct_chatbot()
    
    print("\n" + "="*70 + "\n")
    
    # Test API endpoints
    test_api_endpoints()
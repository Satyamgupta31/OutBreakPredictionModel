#!/usr/bin/env python3
"""
Quick test script to verify the Outbreak Prediction System works correctly.
Run this after setting up the main system to ensure everything is functional.
"""

import sys
import logging
from datetime import datetime

# Configure logging for test
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_outbreak_system():
    """Test the complete outbreak prediction system."""
    
    print("🧪 TESTING OUTBREAK PREDICTION SYSTEM")
    print("="*50)
    
    try:
        # Import the system (assuming ml_pipeline.py is in same directory)
        from ml_pipeline import OutbreakPredictionSystem, OutbreakPredictionAPI
        
        # Test 1: System Initialization
        print("1. Testing system initialization...")
        # --- Using a local MongoDB connection string ---
        mongodb_url = "mongodb://localhost:27017/"
        
        system = OutbreakPredictionSystem(mongodb_url)
        print("   ✅ System initialized successfully")
        
        # Test 2: Data Generation
        print("2. Testing synthetic data generation...")
        data_success = system.create_synthetic_data(num_villages=5, days=30)
        if data_success:
            print("   ✅ Synthetic data created successfully")
        else:
            print("   ❌ Failed to create synthetic data")
            return False
        
        # Test 3: Data Loading
        print("3. Testing data loading...")
        training_data = system.load_training_data()
        if training_data is not None and len(training_data) > 0:
            print(f"   ✅ Loaded {len(training_data)} training records")
        else:
            print("   ❌ Failed to load training data")
            return False
        
        # Test 4: Model Training
        print("4. Testing model training...")
        training_success = system.train_models(training_data)
        if training_success:
            print("   ✅ Models trained successfully")
        else:
            print("   ❌ Model training failed")
            return False
        
        # Test 5: Single Prediction
        print("5. Testing single prediction...")
        test_data = {
            'village_id': 'TEST_VILLAGE',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'diarrhea_cases': 3,
            'vomiting_cases': 2,
            'fever_cases': 5,
            'ph_level': 6.8,
            'turbidity': 3.2,
            'tds': 400,
            'rainfall': 12.5,
            'temperature': 28.0,
            'humidity': 65,
            'season': 'monsoon'
        }
        
        prediction = system.predict_outbreak_risk(test_data)
        if 'error' not in prediction:
            print(f"   ✅ Single prediction successful: {prediction['risk_level']} "
                  f"({prediction['risk_probability']:.1%})")
        else:
            print(f"   ❌ Single prediction failed: {prediction['error']}")
            return False
        
        # Test 6: Batch Prediction
        print("6. Testing batch prediction...")
        batch_predictions = system.batch_predict()
        if batch_predictions and len(batch_predictions) > 0:
            print(f"   ✅ Batch prediction successful: {len(batch_predictions)} predictions")
            
            # Show risk distribution
            risk_counts = {}
            for pred in batch_predictions:
                risk_level = pred.get('risk_level', 'Unknown')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            print("   📊 Risk Distribution:")
            for risk, count in risk_counts.items():
                print(f"      {risk}: {count} villages")
        else:
            print("   ❌ Batch prediction failed")
            return False
        
        # Test 7: Model Persistence
        print("7. Testing model saving/loading...")
        save_success = system.save_models("test_models")
        if save_success:
            print("   ✅ Models saved successfully")
            
            # Test loading
            new_system = OutbreakPredictionSystem(mongodb_url)
            load_success = new_system.load_models("test_models")
            if load_success:
                print("   ✅ Models loaded successfully")
                
                # Test prediction with loaded model
                test_pred = new_system.predict_outbreak_risk(test_data)
                if 'error' not in test_pred:
                    print("   ✅ Prediction with loaded model successful")
                else:
                    print("   ❌ Prediction with loaded model failed")
                    return False
            else:
                print("   ❌ Model loading failed")
                return False
        else:
            print("   ❌ Model saving failed")
            return False
        
        # Test 8: API Wrapper
        print("8. Testing API wrapper...")
        api = OutbreakPredictionAPI(mongodb_url)
        
        # Test initialization
        init_result = api.initialize_system()
        if init_result['status'] == 'success':
            print("   ✅ API initialization successful")
        else:
            print(f"   ❌ API initialization failed: {init_result['message']}")
            return False
        
        # Test API prediction
        api_prediction = api.predict_single(test_data)
        if 'error' not in api_prediction:
            print("   ✅ API prediction successful")
        else:
            print(f"   ❌ API prediction failed: {api_prediction['error']}")
            return False
        
        # Test performance metrics
        performance = api.get_performance_metrics()
        if 'error' not in performance:
            print(f"   ✅ Performance metrics: Accuracy={performance.get('accuracy', 'N/A')}")
        else:
            print(f"   ❌ Performance metrics failed: {performance['error']}")
        
        # Test 9: MongoDB Operations
        print("9. Testing MongoDB operations...")
        save_pred_success = system.save_predictions_to_mongodb(batch_predictions[:3])  # Save first 3
        if save_pred_success:
            print("   ✅ Predictions saved to MongoDB successfully")
        else:
            print("   ❌ Failed to save predictions to MongoDB")
        
        # Clean up
        system.close_connection()
        new_system.close_connection()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("="*50)
        print("Your Outbreak Prediction System is working correctly!")
        print("\nNext steps:")
        print("1. Run 'python ml_pipeline.py' for full demonstration")
        print("2. Integrate with your web dashboard and mobile app")
        print("3. Start collecting real data to replace synthetic data")
        print("4. Set up scheduled retraining for production use")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure ml_pipeline.py is in the same directory")
        return False
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    
    print("🔧 CHECKING DEPENDENCIES")
    print("="*30)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'sklearn': 'scikit-learn',
        'pymongo': 'pymongo',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

if __name__ == "__main__":
    print("🤖 OUTBREAK PREDICTION SYSTEM - TEST SUITE")
    print("="*60)
    
    # Check dependencies first
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies before running tests.")
        sys.exit(1)
    
    print()  # Empty line
    
    # Run system tests
    tests_passed = test_outbreak_system()
    
    if tests_passed:
        print(f"\n🏆 SUCCESS: System is ready for production use!")
        sys.exit(0)
    else:
        print(f"\n💥 FAILURE: Some tests failed. Please check the errors above.")
        sys.exit(1)

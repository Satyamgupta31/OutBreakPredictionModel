#!/usr/bin/env python3
"""
Complete AI/ML Outbreak Prediction System for Rural Health Monitoring

This system predicts disease outbreaks using machine learning by analyzing:
- Symptom reports from villages
- Water quality measurements
- Environmental conditions

Features:
- MongoDB integration for data storage
- RandomForest for outbreak prediction
- Isolation Forest for anomaly detection
- Automated model training and evaluation
- Batch and real-time predictions
- Model persistence and retraining

Author: AI Health Monitoring System
Date: 2025
"""

import logging
import warnings
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pickle

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outbreak_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OutbreakPredictionSystem:
    """
    Complete ML system for predicting disease outbreaks in rural areas.

    This system integrates with MongoDB to load training data from multiple
    collections, trains ML models, and provides predictions for outbreak risks.
    """

    def __init__(self, mongodb_url: str, database_name: str = "healthmonitoring"):
        """
        Initialize the outbreak prediction system.

        Args:
            mongodb_url (str): MongoDB connection string
            database_name (str): Name of the database to use
        """
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.client = None
        self.db = None

        # ML models
        self.outbreak_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.anomaly_model = IsolationForest(
            contamination=0.1,
            random_state=42
        )

        # Preprocessing components
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoder = LabelEncoder()

        # Model status
        self.models_trained = False
        self.feature_columns = []

        # Connect to MongoDB
        self.connect_to_mongodb()

    def connect_to_mongodb(self) -> bool:
        """
        Establish connection to MongoDB.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(
                self.mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )

            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]

            logger.info(f"Successfully connected to MongoDB database: {self.database_name}")
            return True

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            return False

    def create_synthetic_data(self, num_villages: int = 20, days: int = 90) -> bool:
        """
        Generate synthetic data for testing and demonstration with more realistic correlations.
        
        Args:
            num_villages (int): Number of villages to generate data for
            days (int): Number of days of historical data

        Returns:
            bool: True if data created successfully
        """
        try:
            logger.info("Generating improved synthetic data...")

            village_ids = [f"VILLAGE_{i:03d}" for i in range(1, num_villages + 1)]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            collections = {
                'symptom_reports': [],
                'water_quality': [],
                'environment': [],
                'training_data': []
            }

            np.random.seed(42)

            for village_id in village_ids:
                # Base factors for a more nuanced risk calculation
                base_symptoms = np.random.uniform(1, 5)
                base_water_quality = np.random.uniform(0.1, 0.5)
                base_environment = np.random.uniform(0, 0.2)

                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Generate random noise for each day
                    noise = np.random.normal(0, 0.1)

                    # --- Generate data based on a more complex, correlated model ---

                    # A. Symptoms (core driver of outbreaks)
                    symptom_risk_factor = np.clip(base_symptoms + np.random.normal(0, 1), 1, 15)
                    diarrhea = int(np.random.poisson(symptom_risk_factor * 0.5))
                    vomiting = int(np.random.poisson(symptom_risk_factor * 0.3))
                    fever = int(np.random.poisson(symptom_risk_factor * 0.8))

                    # B. Water Quality (increases risk, especially with poor quality)
                    # This now has a more independent relationship to the final outbreak probability
                    water_risk_factor = np.clip(base_water_quality + np.random.normal(0, 0.2), 0.1, 1.0)
                    ph_level = np.clip(7.0 - (water_risk_factor * 2) + np.random.normal(0, 0.5), 5.0, 9.0)
                    turbidity = np.clip(water_risk_factor * 15 + np.random.normal(0, 2), 1, 20)
                    tds = np.clip(water_risk_factor * 800 + np.random.normal(0, 100), 200, 1500)

                    # C. Environment (monsoon season, high humidity/temp increase risk)
                    season_map = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring',
                                  5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
                                  9: 'monsoon', 10: 'monsoon', 11: 'monsoon', 12: 'winter'}
                    season = season_map.get(date.month, 'winter')

                    env_risk_factor = base_environment + noise
                    if season == 'monsoon':
                        env_risk_factor += 0.5 # Significant risk increase in monsoon
                        rainfall = np.random.uniform(15, 60)
                        humidity = np.random.uniform(80, 100)
                        temperature = np.random.uniform(25, 35)
                    else:
                        rainfall = np.random.uniform(0, 15)
                        humidity = np.random.uniform(40, 80)
                        temperature = np.random.uniform(15, 45)

                    # D. Final Outbreak Probability (the target variable)
                    # A more complex combination of factors, including the latest symptom data
                    symptom_weight = np.log1p(diarrhea + vomiting + fever) * 0.3
                    water_weight = water_risk_factor * 0.2
                    env_weight = env_risk_factor * 0.1
                    
                    # Add more complexity and lessen the direct link between water quality and risk
                    outbreak_prob = np.clip(symptom_weight + water_weight + env_weight + np.random.normal(0, 0.05), 0, 1)

                    outbreak_occurred = 1 if np.random.random() < outbreak_prob else 0

                    # Append to collections
                    collections['symptom_reports'].append({
                        'village_id': village_id, 'date': date_str, 'diarrhea_cases': diarrhea,
                        'vomiting_cases': vomiting, 'fever_cases': fever, 'reported_by': f"health_worker_{np.random.randint(1, 6)}"
                    })
                    collections['water_quality'].append({
                        'village_id': village_id, 'date': date_str, 'ph_level': round(ph_level, 2),
                        'turbidity': round(turbidity, 2), 'tds': round(tds, 2), 'sensor_id': f"sensor_{village_id}_water"
                    })
                    collections['environment'].append({
                        'village_id': village_id, 'date': date_str, 'rainfall': round(rainfall, 2),
                        'season': season, 'temperature': round(temperature, 2), 'humidity': round(humidity, 2)
                    })
                    collections['training_data'].append({
                        'village_id': village_id, 'date': date_str, 'outbreak_occurred': outbreak_occurred,
                        'notes': f"Generated data - prob: {outbreak_prob:.2f}"
                    })

            # Insert data into MongoDB
            for collection_name, data in collections.items():
                if data:
                    self.db[collection_name].delete_many({})
                    result = self.db[collection_name].insert_many(data)
                    logger.info(f"Inserted {len(result.inserted_ids)} records into {collection_name}")

            logger.info("Synthetic data generation completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            return False

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """
        Load and combine training data from MongoDB collections.

        Returns:
            pd.DataFrame: Combined training dataset or None if error
        """
        try:
            logger.info("Loading training data from MongoDB...")

            # Load data from all collections and remove the '_id' column immediately
            symptom_data = list(self.db.symptom_reports.find())
            df_symptoms = pd.DataFrame(symptom_data).drop(columns=['_id'], errors='ignore')
            
            water_data = list(self.db.water_quality.find())
            df_water = pd.DataFrame(water_data).drop(columns=['_id'], errors='ignore')
            
            env_data = list(self.db.environment.find())
            df_env = pd.DataFrame(env_data).drop(columns=['_id'], errors='ignore')
            
            training_labels = list(self.db.training_data.find())
            df_labels = pd.DataFrame(training_labels).drop(columns=['_id'], errors='ignore')

            if not all([symptom_data, water_data, env_data, training_labels]):
                logger.warning("Some collections are empty. Generating synthetic data...")
                self.create_synthetic_data()
                return self.load_training_data()  # Recursive call after data generation

            # Merge data on village_id and date
            df_combined = df_symptoms.merge(
                df_water, on=['village_id', 'date'], how='outer'
            ).merge(
                df_env, on=['village_id', 'date'], how='outer'
            ).merge(
                df_labels, on=['village_id', 'date'], how='outer'
            )

            logger.info(f"Loaded {len(df_combined)} training records")
            return df_combined

        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return None

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess the data for machine learning.

        Args:
            df (pd.DataFrame): Raw data
            is_training (bool): Whether this is training data

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Processed features and labels
        """
        try:
            logger.info("Preprocessing data...")

            # Make a copy to avoid modifying original
            df_processed = df.copy()

            # Convert date to datetime and extract features
            df_processed['date'] = pd.to_datetime(df_processed['date'])
            df_processed['month'] = df_processed['date'].dt.month
            df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
            df_processed['weekday'] = df_processed['date'].dt.weekday

            # Encode categorical variables
            if 'season' in df_processed.columns:
                season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'monsoon': 3}
                df_processed['season_encoded'] = df_processed['season'].map(season_mapping)

            # Create derived features
            # Total symptom cases
            symptom_cols = ['diarrhea_cases', 'vomiting_cases', 'fever_cases']
            if all(col in df_processed.columns for col in symptom_cols):
                df_processed['total_symptom_cases'] = df_processed[symptom_cols].sum(axis=1)
                df_processed['symptom_diversity'] = (df_processed[symptom_cols] > 0).sum(axis=1)

            # Water quality score (higher = worse quality)
            if all(col in df_processed.columns for col in ['ph_level', 'turbidity', 'tds']):
                df_processed['water_quality_score'] = (
                    abs(df_processed['ph_level'] - 7.0) +  # Deviation from neutral pH
                    df_processed['turbidity'] / 10 +  # Normalized turbidity
                    df_processed['tds'] / 1000  # Normalized TDS
                )

            # Environmental risk factors
            if 'rainfall' in df_processed.columns:
                df_processed['high_rainfall'] = (df_processed['rainfall'] > 20).astype(int)

            if 'humidity' in df_processed.columns:
                df_processed['high_humidity'] = (df_processed['humidity'] > 80).astype(int)

            # Select feature columns for ML
            feature_cols = [
                'diarrhea_cases', 'vomiting_cases', 'fever_cases',
                'ph_level', 'turbidity', 'tds',
                'rainfall', 'temperature', 'humidity',
                'month', 'day_of_year', 'weekday', 'season_encoded',
                'total_symptom_cases', 'symptom_diversity', 'water_quality_score',
                'high_rainfall', 'high_humidity'
            ]

            # Filter existing columns only
            available_cols = [col for col in feature_cols if col in df_processed.columns]
            X = df_processed[available_cols].copy()

            # Handle missing values
            if is_training:
                X_imputed = pd.DataFrame(
                    self.imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                self.feature_columns = X.columns.tolist()
            else:
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X),
                    columns=X.columns,
                    index=X.index
                )

            # Get labels if training
            y = None
            if is_training and 'outbreak_occurred' in df_processed.columns:
                y = df_processed['outbreak_occurred'].fillna(0).astype(int)

            logger.info(f"Preprocessed {len(X_imputed)} samples with {len(X_imputed.columns)} features")

            return X_imputed, y

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def train_models(self, df: pd.DataFrame) -> bool:
        """
        Train the machine learning models.

        Args:
            df (pd.DataFrame): Training dataset

        Returns:
            bool: True if training successful
        """
        try:
            logger.info("Starting model training...")

            # Preprocess data
            X, y = self.preprocess_data(df, is_training=True)

            if y is None:
                logger.error("No target labels found in training data")
                return False

            # Check for class imbalance
            outbreak_ratio = y.mean()
            logger.info(f"Outbreak ratio in training data: {outbreak_ratio:.2%}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train outbreak prediction model
            logger.info("Training outbreak prediction model...")
            self.outbreak_model.fit(X_train_scaled, y_train)

            # Evaluate outbreak model
            train_score = self.outbreak_model.score(X_train_scaled, y_train)
            test_score = self.outbreak_model.score(X_test_scaled, y_test)

            logger.info(f"Outbreak Model - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")

            # Cross-validation
            cv_scores = cross_val_score(self.outbreak_model, X_train_scaled, y_train, cv=5)
            logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            # Detailed evaluation
            y_pred = self.outbreak_model.predict(X_test_scaled)
            y_pred_proba = self.outbreak_model.predict_proba(X_test_scaled)[:, 1]

            logger.info("Classification Report:")
            logger.info("\n" + classification_report(y_test, y_pred))

            # ROC AUC Score
            if len(np.unique(y_test)) > 1:  # Check if both classes present in test set
                auc_score = roc_auc_score(y_test, y_pred_proba)
                logger.info(f"ROC AUC Score: {auc_score:.3f}")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.outbreak_model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Top 10 Most Important Features:")
            for _, row in feature_importance.head(10).iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.3f}")

            # Train anomaly detection model
            logger.info("Training anomaly detection model...")
            self.anomaly_model.fit(X_train_scaled)

            # Evaluate anomaly model
            anomaly_scores = self.anomaly_model.decision_function(X_test_scaled)
            anomaly_predictions = self.anomaly_model.predict(X_test_scaled)
            anomaly_ratio = (anomaly_predictions == -1).mean()

            logger.info(f"Anomaly detection - {anomaly_ratio:.1%} of test data flagged as anomalies")

            self.models_trained = True
            logger.info("Model training completed successfully!")

            return True

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return False

    def predict_outbreak_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict outbreak risk for a single village/date.

        Args:
            data (dict): Input data for prediction

        Returns:
            dict: Prediction results
        """
        try:
            if not self.models_trained:
                raise ValueError("Models must be trained before making predictions")

            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Preprocess
            X, _ = self.preprocess_data(df, is_training=False)

            # Ensure feature consistency
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0  # Fill missing features with 0

            X = X[self.feature_columns]  # Keep only training features in same order

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make predictions
            outbreak_prob = self.outbreak_model.predict_proba(X_scaled)[0, 1]
            anomaly_score = self.anomaly_model.decision_function(X_scaled)[0]
            is_anomaly = self.anomaly_model.predict(X_scaled)[0] == -1

            # Determine risk level
            if outbreak_prob >= 0.7:
                risk_level = "HIGH"
            elif outbreak_prob >= 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            result = {
                'village_id': data.get('village_id', 'Unknown'),
                'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'risk_probability': round(float(outbreak_prob), 3),
                'risk_level': risk_level,
                'anomaly_score': round(float(anomaly_score), 3),
                'is_anomaly': bool(is_anomaly),
                'prediction_timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'error': str(e),
                'village_id': data.get('village_id', 'Unknown'),
                'date': data.get('date', 'Unknown')
            }

    def batch_predict(self, villages: List[str] = None, date: str = None) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple villages.

        Args:
            villages (list): List of village IDs (None for all villages)
            date (str): Date for predictions (None for latest data)

        Returns:
            list: List of prediction results
        """
        try:
            logger.info("Starting batch prediction...")

            if not self.models_trained:
                logger.error("Models must be trained before making predictions")
                return []

            # Get latest data from MongoDB
            pipeline = []

            if villages:
                pipeline.append({'$match': {'village_id': {'$in': villages}}})

            if date:
                pipeline.append({'$match': {'date': date}})
            else:
                # Get most recent data for each village
                pipeline.extend([
                    {'$sort': {'village_id': 1, 'date': -1}},
                    {'$group': {
                        '_id': '$village_id',
                        'latest_doc': {'$first': '$$ROOT'}
                    }},
                    {'$replaceRoot': {'newRoot': '$latest_doc'}}
                ])

            # Get data from each collection
            symptom_data = list(self.db.symptom_reports.aggregate(pipeline))
            water_data = list(self.db.water_quality.aggregate(pipeline))
            env_data = list(self.db.environment.aggregate(pipeline))

            if not symptom_data:
                logger.warning("No data found for batch prediction")
                return []

            # Combine data
            predictions = []

            for symptom_record in symptom_data:
                village_id = symptom_record['village_id']
                record_date = symptom_record['date']

                # Find matching records
                water_record = next((r for r in water_data
                                     if r['village_id'] == village_id and r['date'] == record_date), {})
                env_record = next((r for r in env_data
                                   if r['village_id'] == village_id and r['date'] == record_date), {})

                # Combine all data
                combined_data = {**symptom_record, **water_record, **env_record}

                # Make prediction
                prediction = self.predict_outbreak_risk(combined_data)
                predictions.append(prediction)

            logger.info(f"Completed batch prediction for {len(predictions)} villages")
            return predictions

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return []

    def save_predictions_to_mongodb(self, predictions: List[Dict[str, Any]]) -> bool:
        """
        Save predictions to MongoDB.

        Args:
            predictions (list): List of prediction results

        Returns:
            bool: True if successful
        """
        try:
            if not predictions:
                logger.warning("No predictions to save")
                return False

            # Add timestamp to all predictions
            for pred in predictions:
                pred['created_at'] = datetime.now()

            # Insert into MongoDB
            result = self.db.predictions.insert_many(predictions)
            logger.info(f"Saved {len(result.inserted_ids)} predictions to MongoDB")

            return True

        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            return False

    def save_models(self, model_dir: str = "models") -> bool:
        """
        Save trained models to disk.

        Args:
            model_dir (str): Directory to save models

        Returns:
            bool: True if successful
        """
        try:
            if not self.models_trained:
                logger.error("No trained models to save")
                return False

            os.makedirs(model_dir, exist_ok=True)

            # Save models and preprocessing components
            joblib.dump(self.outbreak_model, f"{model_dir}/outbreak_model.pkl")
            joblib.dump(self.anomaly_model, f"{model_dir}/anomaly_model.pkl")
            joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
            joblib.dump(self.imputer, f"{model_dir}/imputer.pkl")

            # Save feature columns
            with open(f"{model_dir}/feature_columns.json", 'w') as f:
                json.dump(self.feature_columns, f)

            logger.info(f"Models saved to {model_dir}/")
            return True

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False

    def load_models(self, model_dir: str = "models") -> bool:
        """
        Load trained models from disk.

        Args:
            model_dir (str): Directory containing saved models

        Returns:
            bool: True if successful
        """
        try:
            if not os.path.exists(model_dir):
                logger.error(f"Model directory {model_dir} does not exist")
                return False

            # Load models and preprocessing components
            self.outbreak_model = joblib.load(f"{model_dir}/outbreak_model.pkl")
            self.anomaly_model = joblib.load(f"{model_dir}/anomaly_model.pkl")
            self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
            self.imputer = joblib.load(f"{model_dir}/imputer.pkl")

            # Load feature columns
            with open(f"{model_dir}/feature_columns.json", 'r') as f:
                self.feature_columns = json.load(f)

            self.models_trained = True
            logger.info(f"Models loaded from {model_dir}/")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics of trained models.

        Returns:
            dict: Performance metrics
        """
        try:
            if not self.models_trained:
                return {"error": "Models not trained"}

            # Load test data for evaluation
            df = self.load_training_data()
            if df is None:
                return {"error": "Could not load training data"}

            X, y = self.preprocess_data(df, is_training=True)

            # Split data the same way as training
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_test_scaled = self.scaler.transform(X_test)

            # Get predictions
            y_pred = self.outbreak_model.predict(X_test_scaled)
            y_pred_proba = self.outbreak_model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = self.outbreak_model.score(X_test_scaled, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None

            return {
                "accuracy": round(accuracy, 3),
                "auc_score": round(auc_score, 3) if auc_score else None,
                "test_samples": len(y_test),
                "outbreak_ratio": round(y_test.mean(), 3),
                "models_trained": True
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {"error": str(e)}

    def retrain_models(self) -> bool:
        """
        Retrain models with latest data from MongoDB.

        Returns:
            bool: True if successful
        """
        try:
            logger.info("Starting model retraining...")

            # Load fresh data
            df = self.load_training_data()
            if df is None:
                logger.error("Could not load training data for retraining")
                return False

            # Reset preprocessing components
            self.scaler = StandardScaler()
            self.imputer = SimpleImputer(strategy='mean')

            # Retrain models
            success = self.train_models(df)

            if success:
                logger.info("Model retraining completed successfully")
            else:
                logger.error("Model retraining failed")

            return success

        except Exception as e:
            logger.error(f"Error during model retraining: {str(e)}")
            return False

    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def demonstrate_system():
    """
    Demonstration function showing complete system usage.
    """
    logger.info("="*60)
    logger.info("OUTBREAK PREDICTION SYSTEM DEMONSTRATION")
    logger.info("="*60)

    # MongoDB connection string (set to local database)
    mongodb_url = "mongodb://localhost:27017/"

    try:
        # Initialize system
        logger.info("1. Initializing Outbreak Prediction System...")
        system = OutbreakPredictionSystem(mongodb_url)

        # Generate synthetic data for demonstration
        logger.info("2. Creating synthetic training data...")
        system.create_synthetic_data(num_villages=15, days=60)

        # Load and train models
        logger.info("3. Loading training data...")
        training_data = system.load_training_data()

        if training_data is not None:
            logger.info("4. Training machine learning models...")
            training_success = system.train_models(training_data)

            if training_success:
                # Save models
                logger.info("5. Saving trained models...")
                system.save_models()

                # Get model performance
                logger.info("6. Model Performance Metrics:")
                performance = system.get_model_performance()
                for key, value in performance.items():
                    logger.info(f"   {key}: {value}")

                # Single prediction example
                logger.info("7. Making single village prediction...")
                sample_data = {
                    'village_id': 'VILLAGE_001',
                    'date': '2025-09-12',
                    'diarrhea_cases': 5,
                    'vomiting_cases': 3,
                    'fever_cases': 8,
                    'ph_level': 6.2,
                    'turbidity': 4.5,
                    'tds': 450,
                    'rainfall': 15.2,
                    'temperature': 32.5,
                    'humidity': 75,
                    'season': 'monsoon'
                }

                prediction = system.predict_outbreak_risk(sample_data)
                logger.info("   Single Prediction Result:")
                for key, value in prediction.items():
                    logger.info(f"    {key}: {value}")

                # Batch predictions
                logger.info("8. Making batch predictions for all villages...")
                batch_predictions = system.batch_predict()
                logger.info(f"   Generated {len(batch_predictions)} predictions")

                # Show sample batch predictions
                if batch_predictions:
                    logger.info("   Sample Batch Predictions:")
                    for i, pred in enumerate(batch_predictions[:3]):  # Show first 3
                        logger.info(f"      Village {pred['village_id']}: {pred['risk_level']} "
                                    f"({pred['risk_probability']:.1%} probability)")

                # Save predictions to MongoDB
                logger.info("9. Saving predictions to MongoDB...")
                save_success = system.save_predictions_to_mongodb(batch_predictions)

                if save_success:
                    logger.info("      Predictions saved successfully!")

                # Demonstrate model loading
                logger.info("10. Testing model loading from disk...")
                new_system = OutbreakPredictionSystem(mongodb_url)
                load_success = new_system.load_models()

                if load_success:
                    logger.info("      Models loaded successfully!")

                    # Test prediction with loaded models
                    test_prediction = new_system.predict_outbreak_risk(sample_data)
                    logger.info("      Test prediction with loaded models successful!")

                logger.info("="*60)
                logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
                logger.info("="*60)

                # Summary statistics
                high_risk_count = sum(1 for p in batch_predictions if p.get('risk_level') == 'High')
                medium_risk_count = sum(1 for p in batch_predictions if p.get('risk_level') == 'Medium')
                low_risk_count = sum(1 for p in batch_predictions if p.get('risk_level') == 'Low')

                logger.info("SUMMARY STATISTICS:")
                logger.info(f"   High Risk Villages: {high_risk_count}")
                logger.info(f"   Medium Risk Villages: {medium_risk_count}")
                logger.info(f"   Low Risk Villages: {low_risk_count}")
                logger.info(f"   Total Villages Analyzed: {len(batch_predictions)}")

            else:
                logger.error("Model training failed!")
        else:
            logger.error("Failed to load training data!")

        # Clean up
        system.close_connection()

    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Main execution function - runs complete demonstration
    """
    try:
        demonstrate_system()
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        import traceback
        traceback.print_exc()


# Example usage for integration with other systems
class OutbreakPredictionAPI:
    """
    Simple API wrapper for easy integration with web applications
    """

    def __init__(self, mongodb_url: str):
        self.system = OutbreakPredictionSystem(mongodb_url)

    def initialize_system(self) -> dict:
        """Initialize system and train models if needed"""
        try:
            # Try to load existing models
            if not self.system.load_models():
                logger.info("No existing models found. Training new models...")

                # Load training data
                training_data = self.system.load_training_data()
                if training_data is None:
                    # Create synthetic data for first-time setup
                    self.system.create_synthetic_data()
                    training_data = self.system.load_training_data()

                # Train models
                if training_data is not None:
                    success = self.system.train_models(training_data)
                    if success:
                        self.system.save_models()
                        return {"status": "success", "message": "System initialized and models trained"}
                    else:
                        return {"status": "error", "message": "Model training failed"}
                else:
                    return {"status": "error", "message": "Could not load training data"}
            else:
                return {"status": "success", "message": "System initialized with existing models"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def predict_single(self, village_data: dict) -> dict:
        """Make prediction for single village"""
        try:
            return self.system.predict_outbreak_risk(village_data)
        except Exception as e:
            return {"error": str(e)}

    def predict_batch(self, villages: list = None, date: str = None) -> list:
        """Make batch predictions"""
        try:
            return self.system.batch_predict(villages, date)
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return []

    def get_performance_metrics(self) -> dict:
        """Get model performance metrics"""
        return self.system.get_model_performance()

    def retrain(self) -> dict:
        """Retrain models with latest data"""
        try:
            success = self.system.retrain_models()
            if success:
                self.system.save_models()
                return {"status": "success", "message": "Models retrained successfully"}
            else:
                return {"status": "error", "message": "Model retraining failed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

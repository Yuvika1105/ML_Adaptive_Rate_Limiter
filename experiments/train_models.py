"""
Model Training Pipeline
Train LSTM and Isolation Forest models on generated data
"""

import yaml
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lstm_predictor import LSTMTrafficPredictor
from src.anomaly_detector import TrafficAnomalyDetector
from src.feature_extractor import TrafficFeatureEngineer
import numpy as np
import pandas as pd


def train_all_models(config_path='config.yaml'):
    """Train both LSTM and Anomaly Detection models"""
    
    print("="*60)
    print(" STARTING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # === STEP 1: Feature Engineering ===
    print("\n" + "="*60)
    print("STEP 1: Feature Engineering")
    print("="*60)
    
    feature_engineer = TrafficFeatureEngineer(
        config=config,
        window_seconds=config['ml_models']['feature_engineering']['window_seconds']
    )
    
    X, y, df_features = feature_engineer.process_pipeline(
        raw_data_path='data/raw/traffic_data.csv',
        save_path='data/processed/features.csv'
    )
    
    traffic_data = df_features['request_count'].values
    
    # === STEP 2: Train LSTM Predictor ===
    print("\n" + "="*60)
    print("STEP 2: Training LSTM Traffic Predictor")
    print("="*60)
    
    lstm_config = config['ml_models']['lstm']
    lstm_predictor = LSTMTrafficPredictor(lstm_config)
    
    lstm_results = lstm_predictor.train(
        traffic_data, 
        validation_split=config['experiments']['validation_split']
    )
    
    lstm_predictor.save_model('data/models/lstm_predictor.h5')
    
    # === STEP 3: Train Anomaly Detector ===
    print("\n" + "="*60)
    print("STEP 3: Training Isolation Forest Anomaly Detector")
    print("="*60)
    
    anomaly_config = config['ml_models']['isolation_forest']
    anomaly_detector = TrafficAnomalyDetector(anomaly_config)
    
    anomaly_results = anomaly_detector.train(X, y)
    
    anomaly_detector.save_model('data/models/anomaly_detector.pkl')
    
    # === STEP 4: Feature Importance Analysis ===
    print("\n" + "="*60)
    print("STEP 4: Feature Importance Analysis")
    print("="*60)
    
    importance_df = anomaly_detector.analyze_feature_importance(
        X[:500], 
        feature_engineer.feature_names
    )
    
    print("\n Top 15 Most Important Features:")
    print(importance_df.head(15).to_string())
    
    importance_df.to_csv('data/processed/feature_importance.csv', index=False)
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print(" TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    print(f"\n LSTM Predictor:")
    print(f"  Train MAE: {lstm_results['train_mae']:.2f}")
    print(f"  Val MAE: {lstm_results['val_mae']:.2f}")
    
    print(f"\n Anomaly Detector:")
    print(f"  Accuracy: {anomaly_results.get('accuracy', 0):.4f}")
    print(f"  Precision: {anomaly_results.get('precision', 0):.4f}")
    print(f"  Recall: {anomaly_results.get('recall', 0):.4f}")
    print(f"  False Positive Rate: {anomaly_results.get('false_positive_rate', 0):.4f}")
    
    print(f"\n Models Saved:")
    print(f"  LSTM: data/models/lstm_predictor.h5")
    print(f"  Anomaly Detector: data/models/anomaly_detector.pkl")
    print(f"  Feature Scaler: data/processed/features_scaler.pkl")
    
    return {
        'lstm_results': lstm_results,
        'anomaly_results': anomaly_results,
        'feature_importance': importance_df
    }


if __name__ == "__main__":
    results = train_all_models()
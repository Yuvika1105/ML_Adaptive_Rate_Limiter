"""
Isolation Forest Anomaly Detector
Detects abnormal traffic patterns indicating attacks/bots
Component #2 of the hybrid ML ensemble (PATENT)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os


class TrafficAnomalyDetector:
    """
    Isolation Forest-based anomaly detection
    
    Purpose: Identify suspicious traffic patterns in real-time
    Used by adaptive_engine.py to detect attacks and reduce limits
    
    Why Isolation Forest:
    - Works well with high-dimensional data
    - Fast real-time inference
    - No need for labeled attack data (unsupervised)
    - Effective at detecting outliers
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.contamination = config['contamination']
        self.n_estimators = config['n_estimators']
        self.model = None
        self.feature_importance = None
        
    def build_model(self) -> IsolationForest:
        """
        Build Isolation Forest model
        
        Parameters:
        - contamination: Expected proportion of anomalies (0.1 = 10%)
        - n_estimators: Number of isolation trees
        - max_samples: Samples per tree (smaller = faster)
        """
        
        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.config.get('max_samples', 256),
            random_state=self.config.get('random_state', 42),
            n_jobs=-1  # Use all CPU cores
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray = None) -> dict:
        """
        Train anomaly detector
        
        Args:
            X: Feature matrix (scaled)
            y: Labels (optional, used for evaluation only)
            
        Returns:
            Training metrics
        """
        
        print("🔄 Training Isolation Forest Anomaly Detector...")
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        
        # Build and train model
        self.model = self.build_model()
        self.model.fit(X)
        
        # Get anomaly scores
        scores = self.model.decision_function(X)
        predictions = self.model.predict(X)  # -1 = anomaly, 1 = normal
        
        # Convert to binary (1 = anomaly, 0 = normal)
        predictions_binary = (predictions == -1).astype(int)
        
        print(f"✅ Training Complete!")
        print(f"  Detected anomalies: {predictions_binary.sum()} / {len(X)} ({predictions_binary.mean()*100:.2f}%)")
        
        # Evaluate if labels provided
        results = {}
        if y is not None:
            print(f"\n📊 Evaluation Metrics:")
            print(classification_report(y, predictions_binary, 
                                       target_names=['Normal', 'Attack']))
            
            cm = confusion_matrix(y, predictions_binary)
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            # Calculate metrics
            results = {
                'accuracy': (cm[0,0] + cm[1,1]) / cm.sum(),
                'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
                'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
                'false_positive_rate': cm[0,1] / (cm[0,1] + cm[0,0]),
                'false_negative_rate': cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0,
            }
            
            # ROC-AUC (using anomaly scores)
            try:
                roc_auc = roc_auc_score(y, -scores)  # Negative scores (lower = more anomalous)
                results['roc_auc'] = roc_auc
                print(f"\n  ROC-AUC: {roc_auc:.4f}")
            except:
                pass
            
            print(f"\n  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if samples are anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
    
    def get_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (continuous values)
        
        Returns:
            Anomaly scores (lower = more anomalous)
            Typically in range [-0.5, 0.5]
        """
        
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        scores = self.model.decision_function(X)
        return scores
    
    def get_anomaly_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to probabilities [0, 1]
        
        Returns:
            Probabilities (higher = more likely to be anomaly)
        """
        
        scores = self.get_anomaly_score(X)
        
        # Convert to probabilities using sigmoid
        # Negative scores indicate anomalies
        probabilities = 1 / (1 + np.exp(scores))
        
        return probabilities
    
    def detect_single_request(self, features: np.ndarray) -> dict:
        """
        Analyze a single request in real-time
        
        Args:
            features: Feature vector for one request
            
        Returns:
            Dictionary with anomaly info
        """
        
        # Ensure 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get predictions
        is_anomaly = self.predict(features)[0]
        score = self.get_anomaly_score(features)[0]
        probability = self.get_anomaly_probability(features)[0]
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(score),
            'anomaly_probability': float(probability),
            'confidence': abs(score),  # Higher absolute value = more confident
            'severity': 'high' if probability > 0.8 else 'medium' if probability > 0.5 else 'low'
        }
    
    def analyze_feature_importance(self, X: np.ndarray, feature_names: list) -> pd.DataFrame:
        """
        Analyze which features are most important for anomaly detection
        
        Uses mean decrease in anomaly score
        """
        
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Get baseline scores
        baseline_scores = self.get_anomaly_score(X)
        
        importances = []
        
        for i, feature_name in enumerate(feature_names):
            # Shuffle this feature
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])
            
            # Get new scores
            shuffled_scores = self.get_anomaly_score(X_shuffled)
            
            # Calculate importance (change in anomaly detection)
            importance = np.abs(baseline_scores - shuffled_scores).mean()
            importances.append(importance)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        return importance_df
    
    def save_model(self, path: str):
        """Save trained model"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'feature_importance': self.feature_importance
            }, f)
        
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.config = data['config']
        self.feature_importance = data.get('feature_importance')
        
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed features
    X = np.load('data/processed/features_X.npy')
    y = np.load('data/processed/features_y.npy')
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Attacks: {y.sum()} ({y.mean()*100:.2f}%)")
    
    # Initialize and train
    detector = TrafficAnomalyDetector(config['ml_models']['isolation_forest'])
    results = detector.train(X, y)
    
    # Save model
    detector.save_model('data/models/anomaly_detector.pkl')
    
    # Test single prediction
    test_sample = X[0]
    result = detector.detect_single_request(test_sample)
    
    print(f"\n🎯 Single Request Analysis:")
    print(f"  Is Anomaly: {result['is_anomaly']}")
    print(f"  Probability: {result['anomaly_probability']:.4f}")
    print(f"  Severity: {result['severity']}")
    print(f"  Actual Label: {'Attack' if y[0] == 1 else 'Normal'}")
    
    # Feature importance
    if 'feature_names' in locals():
        print("\n📊 Top 10 Most Important Features:")
        importance_df = detector.analyze_feature_importance(X[:100], feature_names)
        print(importance_df.head(10))
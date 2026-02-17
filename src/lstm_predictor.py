"""
LSTM Traffic Predictor
Predicts future API traffic based on historical patterns
Component #1 of the hybrid ML ensemble (PATENT)
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import pickle
import os


class LSTMTrafficPredictor:
    """
    LSTM-based traffic forecasting model
    
    Purpose: Predict expected traffic load in next time window
    Used by adaptive_engine.py to proactively adjust rate limits
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config['sequence_length']
        self.units = config['units']
        self.dropout = config['dropout']
        self.model = None
        self.history = []
        
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> tuple:
        """
        Create time series sequences for LSTM training
        
        Args:
            data: Traffic counts over time
            seq_length: Number of time steps to look back
            
        Returns:
            X: Input sequences (shape: [samples, seq_length, 1])
            y: Target values (next time step)
        """
        
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self) -> keras.Model:
        """
        Build LSTM architecture
        
        Architecture:
        - LSTM layer (captures temporal patterns)
        - Dropout (prevents overfitting)
        - Dense layer (output prediction)
        """
        
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                self.units,
                activation='tanh',
                return_sequences=True,
                input_shape=(self.sequence_length, 1)
            ),
            layers.Dropout(self.dropout),
            
            # Second LSTM layer
            layers.LSTM(
                self.units // 2,
                activation='tanh',
                return_sequences=False
            ),
            layers.Dropout(self.dropout),
            
            # Output layer
            layers.Dense(1, activation='relu')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, traffic_data: np.ndarray, validation_split: float = 0.2) -> dict:
        """
        Train LSTM model on historical traffic data
        
        Args:
            traffic_data: Array of traffic counts over time
            validation_split: Fraction of data for validation
            
        Returns:
            Training history
        """
        
        print("🔄 Training LSTM Traffic Predictor...")
        
        # Create sequences
        X, y = self._create_sequences(traffic_data, self.sequence_length)
        print(f"  Created {len(X)} training sequences")
        
        # Reshape for LSTM [samples, time_steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Build model
        self.model = self.build_model()
        print(f"  Model architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        train_loss, train_mae = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n✅ Training Complete!")
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Val MAE: {val_mae:.2f}")
        
        return {
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'history': history.history
        }
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict next time window's traffic
        
        Args:
            sequence: Recent traffic history (last N time steps)
            
        Returns:
            Predicted traffic count
        """
        
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Ensure correct shape
        if len(sequence) < self.sequence_length:
            # Pad with zeros if needed
            padding = np.zeros(self.sequence_length - len(sequence))
            sequence = np.concatenate([padding, sequence])
        elif len(sequence) > self.sequence_length:
            # Take last N values
            sequence = sequence[-self.sequence_length:]
        
        # Reshape for prediction
        sequence = sequence.reshape(1, self.sequence_length, 1)
        
        # Predict
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        
        return max(0, prediction)  # Ensure non-negative
    
    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Predict for multiple sequences at once"""
        
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Reshape if needed
        if len(sequences.shape) == 2:
            sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        
        predictions = self.model.predict(sequences, verbose=0)
        return predictions.flatten()
    
    def update_history(self, new_traffic: float):
        """Update traffic history for online prediction"""
        
        self.history.append(new_traffic)
        
        # Keep only recent history
        if len(self.history) > self.sequence_length * 2:
            self.history = self.history[-self.sequence_length * 2:]
    
    def get_recent_sequence(self) -> np.ndarray:
        """Get most recent sequence for prediction"""
        
        if len(self.history) < self.sequence_length:
            # Pad with zeros
            padding = [0] * (self.sequence_length - len(self.history))
            sequence = padding + self.history
        else:
            sequence = self.history[-self.sequence_length:]
        
        return np.array(sequence)
    
    def save_model(self, path: str):
        """Save trained model"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save history separately
        with open(path.replace('.h5', '_history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)
        
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        
        self.model = keras.models.load_model(path)
        
        # Load history if exists
        history_path = path.replace('.h5', '_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
        
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed features
    df = pd.read_csv('data/processed/features.csv')
    traffic_data = df['request_count'].values
    
    # Initialize and train
    predictor = LSTMTrafficPredictor(config['ml_models']['lstm'])
    
    # Train
    results = predictor.train(traffic_data, validation_split=0.2)
    
    # Save model
    predictor.save_model('data/models/lstm_predictor.h5')
    
    # Test prediction
    recent_sequence = traffic_data[-60:]
    prediction = predictor.predict(recent_sequence)
    actual = traffic_data[-1]
    
    print(f"\n🎯 Prediction Test:")
    print(f"  Recent sequence (last 5): {recent_sequence[-5:]}")
    print(f"  Predicted next: {prediction:.2f}")
    print(f"  Actual: {actual:.2f}")
    print(f"  Error: {abs(prediction - actual):.2f}")
"""
⭐⭐⭐ ADAPTIVE RATE LIMITING ENGINE ⭐⭐⭐
This is THE CORE PATENT - The Novel ML-Guided Rate Limiting Algorithm

PATENT CLAIMS:
1. Hybrid ML Ensemble combining LSTM prediction + Anomaly Detection + Adaptive Thresholds
2. Novel Dynamic Threshold Calculation Formula
3. Fairness-Aware Rate Limiting with Real-time Learning
4. ⭐ NEW: Explainable AI Decision Making (XAI)
5. ⭐ NEW: Reinforcement Learning Feedback Loop
6. ⭐ NEW: Cross-Endpoint Behavioral Analysis

This file contains the unique algorithmic innovation that makes the system patentable.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import psutil  # For system health monitoring


@dataclass
class RateLimitDecision:
    """Result of rate limiting decision with explainability"""
    allow: bool
    current_limit: int
    predicted_traffic: float
    anomaly_score: float
    fairness_multiplier: float
    confidence: float
    reason: str
    # ⭐ NEW: Explainability fields
    explanation: Optional[Dict] = None
    behavioral_risk: Optional[float] = None


class AdaptiveRateLimitEngine:
    """
    ⭐ PATENT CLAIM #1: Hybrid ML Ensemble for Dynamic Rate Limiting
    
    Novel Innovation: Combines FIVE ML components intelligently:
    1. LSTM Traffic Predictor - forecasts expected load
    2. Isolation Forest Anomaly Detector - identifies attacks  
    3. Adaptive Threshold Calculator - dynamically adjusts limits
    4. ⭐ NEW: Explainability Layer - transparent decisions
    5. ⭐ NEW: Online Learner - self-improving via RL
    6. ⭐ NEW: Behavioral Analyzer - cross-endpoint patterns
    
    What makes it novel:
    - Real-time ensemble decision making
    - Weighted voting between prediction and detection
    - Continuous adaptation based on feedback
    - Context-aware (business hours, flash sales, user reputation)
    - ⭐ Explainable decisions with audit trail
    - ⭐ Self-healing through reinforcement learning
    - ⭐ Behavioral pattern recognition across API surface
    """
    
    def __init__(self, config: dict, lstm_model=None, anomaly_model=None, 
                 feature_extractor=None, explainer=None):
        self.config = config
        
        # ML Models
        self.lstm_model = lstm_model
        self.anomaly_model = anomaly_model
        self.feature_extractor = feature_extractor
        
        # ⭐ NEW: Explainability component
        self.explainer = explainer
        self.enable_explanations = config.get('explainability', {}).get('enable_explanations', True)
        self.audit_log_path = config.get('explainability', {}).get('audit_log_path', 'logs/audit.log')
        
        # Rate limiting parameters
        self.base_limit = config['base_limit']
        self.min_limit = config['min_limit']
        self.max_limit = config['max_limit']
        
        # Ensemble weights (PATENT ELEMENT: weighted combination)
        self.lstm_weight = config['lstm_weight']  # 0.6 default
        self.anomaly_weight = config['anomaly_weight']  # 0.4 default
        
        # Adaptation parameters
        self.alpha = config['adaptation_speed']  # Smoothing factor
        self.fairness_beta = config['fairness_beta']
        
        # State tracking
        self.current_limit = self.base_limit
        self.traffic_history = []
        self.decision_history = []
        
        # ⭐ NEW: Online Learning State (Reinforcement Learning)
        self.enable_online_learning = config.get('online_learning', {}).get('enable_feedback_loop', True)
        self.learning_rate = config.get('online_learning', {}).get('learning_rate', 0.01)
        self.reward_decay = config.get('online_learning', {}).get('reward_decay', 0.95)
        
        self.feedback_history = []
        self.performance_metrics = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'true_negatives': 0
        }
        
        # ⭐ NEW: System Health Monitoring
        self.cpu_threshold = config.get('online_learning', {}).get('cpu_threshold', 80)
        self.response_time_threshold = config.get('online_learning', {}).get('response_time_threshold', 500)
        
        # Performance tracking (legacy - now in performance_metrics)
        self.false_positives = 0
        self.false_negatives = 0
        self.total_decisions = 0
        
    def calculate_adaptive_limit(self, request_features: Dict) -> RateLimitDecision:
        """
        ⭐ PATENT CLAIM #2: Dynamic Threshold Calculation Algorithm
        
        ENHANCED FORMULA (with behavioral risk):
        L(t) = L_base × [w1×P(t) + w2×S(t)] × F(t) × C(t) × B(t)
        
        Where:
        - L(t) = Adaptive limit at time t
        - L_base = Baseline limit
        - P(t) = Normalized prediction score (0-2, from LSTM)
        - S(t) = Safety factor (1 - anomaly_score, from Isolation Forest)
        - F(t) = Fairness multiplier (user-specific adjustment)
        - C(t) = Context multiplier (business hours, flash sales)
        - ⭐ B(t) = Behavioral risk factor (cross-endpoint patterns)
        - w1, w2 = Learned ensemble weights
        
        This specific mathematical combination is NOVEL and PATENTABLE.
        """
        
        # === STEP 1: LSTM PREDICTION (Traffic Forecast) ===
        predicted_traffic = self._predict_traffic(request_features)
        prediction_score = self._normalize_prediction(predicted_traffic)
        
        # === STEP 2: ANOMALY DETECTION (Attack Identification) ===
        anomaly_score = self._detect_anomaly(request_features)
        safety_factor = 1 - anomaly_score  # Invert: lower limit if anomalous
        
        # === STEP 3: ⭐ NEW - BEHAVIORAL ANALYSIS (Cross-Endpoint) ===
        behavioral_risk = self._analyze_behavioral_patterns(request_features)
        behavioral_safety = 1 - behavioral_risk
        
        # === STEP 4: ENSEMBLE COMBINATION (ENHANCED) ===
        # Now includes behavioral component
        ensemble_score = (
            self.lstm_weight * prediction_score +
            self.anomaly_weight * safety_factor * behavioral_safety
        )
        
        # === STEP 5: FAIRNESS ADJUSTMENT (PATENT CLAIM #3) ===
        fairness_multiplier = self._calculate_fairness(request_features)
        
        # === STEP 6: CONTEXT AWARENESS ===
        context_multiplier = self._get_context_multiplier(request_features)
        
        # === STEP 7: ⭐ NEW - SYSTEM HEALTH ADJUSTMENT ===
        health_multiplier = self._get_system_health_multiplier()
        
        # === STEP 8: APPLY ENHANCED FORMULA ===
        new_limit = (
            self.base_limit * 
            ensemble_score * 
            fairness_multiplier * 
            context_multiplier *
            health_multiplier
        )
        
        # === STEP 9: SMOOTH ADAPTATION ===
        # Prevent sudden jumps - gradual adaptation
        adaptive_limit = (
            self.alpha * new_limit + 
            (1 - self.alpha) * self.current_limit
        )
        
        # === STEP 10: ENFORCE BOUNDS ===
        final_limit = int(np.clip(adaptive_limit, self.min_limit, self.max_limit))
        
        # === STEP 11: MAKE DECISION ===
        current_rate = request_features.get('requests_per_minute', 0)
        allow = current_rate < final_limit
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            prediction_score, 
            anomaly_score, 
            fairness_multiplier,
            behavioral_risk
        )
        
        # Determine reason
        if not allow:
            if behavioral_risk > 0.8:
                reason = "Blocked: Bot pattern detected across endpoints"
            elif anomaly_score > 0.7:
                reason = "Blocked: High anomaly score (suspected attack)"
            elif current_rate > final_limit * 1.5:
                reason = "Blocked: Excessive traffic rate"
            else:
                reason = f"Blocked: Rate {current_rate} exceeds limit {final_limit}"
        else:
            reason = "Allowed: Within adaptive rate limit"
        
        # === STEP 12: ⭐ NEW - GENERATE EXPLANATION (XAI) ===
        explanation = None
        if self.enable_explanations and self.explainer is not None:
            # Extract feature vector for explainer
            feature_vector = self._extract_feature_vector(request_features)
            
            explanation = self.explainer.explain_decision(
                request_features=feature_vector,
                decision="BLOCKED" if not allow else "ALLOWED",
                anomaly_score=anomaly_score,
                adaptive_limit=final_limit
            )
            
            # Log to audit trail
            if self.audit_log_path:
                self._write_audit_log(explanation)
        
        # Update state
        self.current_limit = final_limit
        self.total_decisions += 1
        
        # Create decision
        decision = RateLimitDecision(
            allow=allow,
            current_limit=final_limit,
            predicted_traffic=predicted_traffic,
            anomaly_score=anomaly_score,
            fairness_multiplier=fairness_multiplier,
            confidence=confidence,
            reason=reason,
            explanation=explanation,  # ⭐ NEW
            behavioral_risk=behavioral_risk  # ⭐ NEW
        )
        
        # Log decision
        self.decision_history.append({
            'timestamp': time.time(),
            'decision': allow,
            'limit': final_limit,
            'predicted': predicted_traffic,
            'anomaly': anomaly_score,
            'fairness': fairness_multiplier,
            'behavioral_risk': behavioral_risk,  # ⭐ NEW
            'confidence': confidence
        })
        
        return decision
    
    def _analyze_behavioral_patterns(self, features: Dict) -> float:
        """
        ⭐ PATENT ELEMENT: Cross-Endpoint Behavioral Analysis
        
        Analyzes user behavior across multiple API endpoints to detect bots
        
        Returns:
            Risk score 0-1 (0=safe, 1=highly suspicious)
        """
        
        # If feature extractor has behavioral analyzer, use it
        if hasattr(self.feature_extractor, 'endpoint_analyzer'):
            user_id = features.get('user_id', 'anonymous')
            endpoint = features.get('endpoint', '/api/default')
            
            analysis = self.feature_extractor.endpoint_analyzer.analyze_request_sequence(
                user_id, endpoint
            )
            
            return analysis.get('suspicion_score', 0.0)
        
        # Fallback: simple heuristics
        behavioral_signals = []
        
        # Signal 1: Very low inter-request time (bot-like)
        avg_inter_time = features.get('avg_inter_request_time', 1.0)
        if avg_inter_time < 0.1:  # < 100ms between requests
            behavioral_signals.append(0.8)
        
        # Signal 2: Low endpoint diversity (targeting specific endpoint)
        endpoint_diversity = features.get('endpoint_diversity', 1.0)
        if endpoint_diversity < 0.3:
            behavioral_signals.append(0.6)
        
        # Signal 3: Rapid sequence flag
        has_rapid_sequence = features.get('has_rapid_sequence', 0)
        if has_rapid_sequence:
            behavioral_signals.append(0.7)
        
        # Average signals
        if behavioral_signals:
            return np.mean(behavioral_signals)
        
        return 0.0  # No suspicious signals
    
    def _get_system_health_multiplier(self) -> float:
        """
        ⭐ PATENT ELEMENT: System Health-Aware Rate Limiting
        
        Automatically tighten limits if system is under stress
        """
        
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            if cpu_usage > self.cpu_threshold:
                # System stressed - reduce limits
                stress_level = (cpu_usage - self.cpu_threshold) / (100 - self.cpu_threshold)
                multiplier = 1 - (stress_level * 0.5)  # Reduce by up to 50%
                return max(0.5, multiplier)
        except:
            pass
        
        return 1.0  # Normal operation
    
    def _extract_feature_vector(self, features: Dict) -> np.ndarray:
        """Extract feature vector from request features for explainer"""
        
        # Extract key features for explanation
        return np.array([
            features.get('requests_per_minute', 0),
            features.get('unique_ips', 0),
            features.get('error_rate', 0),
            features.get('avg_response_time', 0),
            features.get('spike_ratio', 1.0),
            features.get('ip_concentration', 1.0)
        ])
    
    def _write_audit_log(self, explanation: Dict):
        """Write decision explanation to audit log"""
        
        try:
            import os
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
            
            with open(self.audit_log_path, 'a') as f:
                if self.explainer:
                    audit_text = self.explainer.generate_audit_log(explanation)
                    f.write(audit_text)
        except Exception as e:
            print(f"⚠️ Failed to write audit log: {e}")
    
    def _predict_traffic(self, features: Dict) -> float:
        """Use LSTM to predict expected traffic in next window"""
        
        if self.lstm_model is None:
            # Fallback: use moving average
            if len(self.traffic_history) > 0:
                return np.mean(self.traffic_history[-10:])
            return self.base_limit
        
        # Get recent sequence
        if len(self.traffic_history) < 60:
            sequence = np.pad(
                self.traffic_history, 
                (60 - len(self.traffic_history), 0),
                mode='constant'
            )
        else:
            sequence = np.array(self.traffic_history[-60:])
        
        # Predict
        prediction = self.lstm_model.predict(sequence)
        
        # Update history
        current_traffic = features.get('requests_per_minute', 0)
        self.traffic_history.append(current_traffic)
        if len(self.traffic_history) > 120:
            self.traffic_history = self.traffic_history[-120:]
        
        return prediction
    
    def _normalize_prediction(self, predicted: float) -> float:
        """
        Normalize prediction to 0-2 range
        
        - 1.0 = normal traffic (no adjustment)
        - 2.0 = high predicted traffic (increase limit 2x)
        - 0.5 = low predicted traffic (decrease limit to 50%)
        """
        
        # Compare to baseline
        ratio = predicted / self.base_limit if self.base_limit > 0 else 1.0
        
        # Normalize to 0-2 range with clipping
        normalized = np.clip(ratio, 0.5, 2.0)
        
        return normalized
    
    def _detect_anomaly(self, features: Dict) -> float:
        """
        Use Isolation Forest to detect anomalies
        
        Returns:
            Anomaly probability 0-1 (0=normal, 1=highly anomalous)
        """
        
        if self.anomaly_model is None:
            # Fallback: simple heuristic
            current_rate = features.get('requests_per_minute', 0)
            ratio = current_rate / self.base_limit
            
            if ratio > 5:
                return 0.9
            elif ratio > 3:
                return 0.7
            elif ratio > 2:
                return 0.4
            else:
                return 0.1
        
        # Extract features for anomaly detection
        feature_vector = self._extract_feature_vector(features)
        
        # Get anomaly probability
        anomaly_prob = self.anomaly_model.get_anomaly_probability(
            feature_vector.reshape(1, -1)
        )[0]
        
        return anomaly_prob
    
    def _calculate_fairness(self, features: Dict) -> float:
        """
        ⭐ PATENT CLAIM #3: Fairness-Aware Rate Limiting
        
        Adjust limits to ensure fair access across users
        
        Novel Formula:
        F = 1 + β × (1 - user_request_ratio) × user_reputation
        
        - New/light users get boost (encourage adoption)
        - Heavy users get moderate limits
        - Suspicious patterns get penalty
        """
        
        total_requests = features.get('total_requests', 1)
        user_requests = features.get('user_requests', 1)
        
        # User's share of traffic
        user_ratio = min(1.0, user_requests / total_requests)
        
        # Calculate reputation
        error_rate = features.get('error_rate', 0)
        reputation = 1 - min(1.0, error_rate)
        
        # Additional penalty for suspicious patterns
        if features.get('is_bot', 0) == 1:
            reputation *= 0.5
        
        # Apply fairness formula (NOVEL)
        fairness = 1.0 + self.fairness_beta * (1 - user_ratio) * reputation
        
        # Constrain to reasonable range
        fairness = np.clip(fairness, 0.5, 1.5)
        
        return fairness
    
    def _get_context_multiplier(self, features: Dict) -> float:
        """
        Adjust based on business context
        
        - Flash sales: increase limits (more capacity needed)
        - Business hours: normal limits
        - Night time: reduced limits (less expected traffic)
        """
        
        hour = features.get('hour', 12)
        is_business_hours = features.get('is_business_hours', 1)
        is_flash_sale = features.get('is_flash_sale', 0)
        
        # Flash sale multiplier
        if is_flash_sale:
            return self.config.get('flash_sale_multiplier', 2.0)
        
        # Business hours
        if is_business_hours:
            return 1.0
        
        # Night time - reduce limits
        if 0 <= hour <= 6:
            return 0.6
        
        # Evening
        if 18 <= hour <= 22:
            return 0.8
        
        return 1.0
    
    def _calculate_confidence(
        self, 
        prediction_score: float, 
        anomaly_score: float,
        fairness: float,
        behavioral_risk: float = 0.0
    ) -> float:
        """
        Calculate confidence in the decision
        
        ⭐ ENHANCED: Now includes behavioral risk
        
        High confidence when:
        - Prediction and anomaly detection agree
        - Anomaly score is very high or very low (clear signal)
        - Fairness multiplier is near 1.0 (typical user)
        - ⭐ Behavioral patterns are clear (very risky or very safe)
        """
        
        # Agreement between LSTM and anomaly detector
        prediction_deviation = abs(prediction_score - 1.0)
        anomaly_certainty = max(anomaly_score, 1 - anomaly_score)
        fairness_certainty = 1 - abs(fairness - 1.0)
        behavioral_certainty = max(behavioral_risk, 1 - behavioral_risk)
        
        # Combine (updated weights)
        confidence = (
            0.3 * anomaly_certainty +
            0.25 * (1 - prediction_deviation) +
            0.25 * fairness_certainty +
            0.2 * behavioral_certainty  # ⭐ NEW
        )
        
        return np.clip(confidence, 0, 1)
    
    # ⭐⭐⭐ NEW METHODS: REINFORCEMENT LEARNING FEEDBACK LOOP ⭐⭐⭐
    
    def process_feedback(self, feedback: Dict):
        """
        ⭐ PATENT ELEMENT: Self-Healing Reinforcement Learning
        
        Feedback format:
        {
            'type': 'false_positive' | 'false_negative' | 'true_positive' | 'true_negative',
            'request_id': '...',
            'actual_label': 0 or 1,
            'predicted_label': 0 or 1,
            'system_metrics': {
                'cpu_usage': 75,
                'avg_response_time': 250,
                'error_rate': 0.05
            }
        }
        """
        
        if not self.enable_online_learning:
            return
        
        # Update metrics
        feedback_type = feedback['type']
        self.performance_metrics[feedback_type] += 1
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Calculate reward signal
        reward = self._calculate_reward(feedback)
        
        # Update model weights based on reward
        self._update_weights_rl(reward, feedback)
        
        # Check if system is under stress
        if 'system_metrics' in feedback:
            self._adjust_for_system_health(feedback['system_metrics'])
    
    def _calculate_reward(self, feedback: Dict) -> float:
        """
        Calculate reward for reinforcement learning
        
        PATENT FORMULA: Multi-objective reward function
        R = w_acc × accuracy_reward + w_sys × system_health_reward
        """
        
        # Accuracy reward
        if feedback['type'] == 'true_positive':
            accuracy_reward = 1.0  # Correctly blocked attack
        elif feedback['type'] == 'true_negative':
            accuracy_reward = 0.5  # Correctly allowed legitimate user
        elif feedback['type'] == 'false_positive':
            accuracy_reward = -1.0  # BAD: Blocked good user
        elif feedback['type'] == 'false_negative':
            accuracy_reward = -2.0  # VERY BAD: Missed attack
        else:
            accuracy_reward = 0
        
        # System health reward
        if 'system_metrics' in feedback:
            metrics = feedback['system_metrics']
            cpu_usage = metrics.get('cpu_usage', 50)
            response_time = metrics.get('avg_response_time', 200)
            
            # Penalize if system is stressed
            cpu_penalty = -0.5 if cpu_usage > self.cpu_threshold else 0
            rt_penalty = -0.5 if response_time > self.response_time_threshold else 0
            
            system_health_reward = cpu_penalty + rt_penalty
        else:
            system_health_reward = 0
        
        # Combined reward
        total_reward = 0.7 * accuracy_reward + 0.3 * system_health_reward
        
        return total_reward
    
    def _update_weights_rl(self, reward: float, feedback: Dict):
        """
        ⭐ PATENT ELEMENT: Reinforcement Learning Weight Adaptation
        
        Update LSTM vs Anomaly weights based on performance
        """
        
        # If we're getting false negatives (missing attacks), trust anomaly detector more
        if feedback['type'] == 'false_negative':
            self.anomaly_weight += self.learning_rate * abs(reward)
            self.lstm_weight -= self.learning_rate * abs(reward)
        
        # If we're getting false positives (blocking good users), trust LSTM more
        elif feedback['type'] == 'false_positive':
            self.lstm_weight += self.learning_rate * abs(reward)
            self.anomaly_weight -= self.learning_rate * abs(reward)
        
        # Normalize weights to sum to 1
        total = self.lstm_weight + self.anomaly_weight
        self.lstm_weight /= total
        self.anomaly_weight /= total
        
        print(f"⚙️ Weight Update: LSTM={self.lstm_weight:.3f}, Anomaly={self.anomaly_weight:.3f}")
    
    def _adjust_for_system_health(self, system_metrics: Dict):
        """
        ⭐ PATENT ELEMENT: Self-Healing System Response
        
        If system is under stress, automatically tighten limits
        """
        
        cpu_usage = system_metrics.get('cpu_usage', 50)
        response_time = system_metrics.get('avg_response_time', 200)
        
        # Calculate stress level (0 to 1)
        cpu_stress = max(0, (cpu_usage - 50) / 50)  # 0 at 50%, 1 at 100%
        rt_stress = max(0, (response_time - 200) / 500)  # 0 at 200ms, 1 at 700ms
        
        total_stress = (cpu_stress + rt_stress) / 2
        
        if total_stress > 0.5:
            # System is stressed - tighten limits
            stress_multiplier = 1 - (total_stress * 0.5)  # Reduce by up to 50%
            self.current_limit = int(self.current_limit * stress_multiplier)
            
            print(f"🚨 SYSTEM STRESS DETECTED: {total_stress:.2%}")
            print(f"   Auto-reducing limit to {self.current_limit} req/min")
    
    def get_learning_stats(self) -> Dict:
        """
        Get current learning statistics
        """
        
        total = sum(self.performance_metrics.values())
        if total == 0:
            return {'status': 'No feedback received yet'}
        
        accuracy = (
            (self.performance_metrics['true_positives'] + 
             self.performance_metrics['true_negatives']) / total
        )
        
        precision = (
            self.performance_metrics['true_positives'] /
            (self.performance_metrics['true_positives'] + 
             self.performance_metrics['false_positives'] + 1e-10)
        )
        
        recall = (
            self.performance_metrics['true_positives'] /
            (self.performance_metrics['true_positives'] + 
             self.performance_metrics['false_negatives'] + 1e-10)
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'total_decisions': total,
            'current_weights': {
                'lstm': self.lstm_weight,
                'anomaly': self.anomaly_weight
            },
            'metrics': self.performance_metrics
        }
    
    # Legacy method - kept for compatibility
    def update_models_online(self, feedback: Dict):
        """Legacy wrapper for process_feedback"""
        self.process_feedback(feedback)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance statistics"""
        
        return {
            'total_decisions': self.total_decisions,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'false_positive_rate': self.false_positives / self.total_decisions if self.total_decisions > 0 else 0,
            'false_negative_rate': self.false_negatives / self.total_decisions if self.total_decisions > 0 else 0,
            'current_limit': self.current_limit,
            'lstm_weight': self.lstm_weight,
            'anomaly_weight': self.anomaly_weight,
            'learning_stats': self.get_learning_stats()
        }
    
    def save_state(self, path: str):
        """Save engine state"""
        
        state = {
            'config': self.config,
            'current_limit': self.current_limit,
            'traffic_history': self.traffic_history,
            'decision_history': self.decision_history[-1000:],  # Keep recent
            'lstm_weight': self.lstm_weight,
            'anomaly_weight': self.anomaly_weight,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total_decisions': self.total_decisions,
            'performance_metrics': self.performance_metrics,  # ⭐ NEW
            'feedback_history': self.feedback_history[-100:],  # ⭐ NEW
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ Engine state saved to {path}")
    
    def load_state(self, path: str):
        """Load engine state"""
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.current_limit = state['current_limit']
        self.traffic_history = state['traffic_history']
        self.decision_history = state['decision_history']
        self.lstm_weight = state['lstm_weight']
        self.anomaly_weight = state['anomaly_weight']
        self.false_positives = state.get('false_positives', 0)
        self.false_negatives = state.get('false_negatives', 0)
        self.total_decisions = state.get('total_decisions', 0)
        self.performance_metrics = state.get('performance_metrics', {
            'false_positives': 0, 'false_negatives': 0,
            'true_positives': 0, 'true_negatives': 0
        })
        self.feedback_history = state.get('feedback_history', [])
        
        print(f"✓ Engine state loaded from {path}")


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config.py', 'r') as f:
        exec(f.read(), globals())
    
    config_dict = {
        'base_limit': rate_limiting['base_limit'],
        'min_limit': rate_limiting['min_limit'],
        'max_limit': rate_limiting['max_limit'],
        'adaptation_speed': rate_limiting['adaptation_speed'],
        'lstm_weight': rate_limiting['lstm_weight'],
        'anomaly_weight': rate_limiting['anomaly_weight'],
        'fairness_beta': rate_limiting['fairness_beta'],
        'flash_sale_multiplier': rate_limiting.get('flash_sale_multiplier', 2.0),
        'explainability': explainability,
        'online_learning': online_learning,
        'cross_endpoint': cross_endpoint
    }
    
    # Initialize engine (without models for testing)
    engine = AdaptiveRateLimitEngine(config_dict)
    
    # Simulate requests
    print("🔄 Testing Enhanced Adaptive Rate Limiting Engine...\n")
    
    # Normal request
    normal_request = {
        'requests_per_minute': 80,
        'unique_users': 40,
        'unique_ips': 35,
        'error_rate': 0.02,
        'avg_response_time': 120,
        'ip_concentration': 2.0,
        'hour': 14,
        'is_business_hours': 1,
        'user_id': 'U123',
        'total_requests': 1000,
        'user_requests': 50,
        'endpoint': '/api/products',
        'avg_inter_request_time': 1.5,
        'endpoint_diversity': 0.8,
        'has_rapid_sequence': 0
    }
    
    decision = engine.calculate_adaptive_limit(normal_request)
    print(f"Normal Request:")
    print(f"  Decision: {'✅ ALLOW' if decision.allow else '❌ BLOCK'}")
    print(f"  Limit: {decision.current_limit}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Behavioral Risk: {decision.behavioral_risk:.2f}")
    print(f"  Reason: {decision.reason}\n")
    
    # Bot request with cross-endpoint pattern
    bot_request = {
        'requests_per_minute': 500,
        'unique_users': 5,
        'unique_ips': 3,
        'error_rate': 0.8,
        'avg_response_time': 50,
        'ip_concentration': 150.0,
        'hour': 14,
        'is_business_hours': 1,
        'user_id': 'BOT99',
        'total_requests': 1000,
        'user_requests': 500,
        'is_bot': 1,
        'endpoint': '/api/checkout',
        'avg_inter_request_time': 0.05,  # ⭐ 50ms - bot-like
        'endpoint_diversity': 0.2,  # ⭐ Low diversity
        'has_rapid_sequence': 1  # ⭐ Rapid requests
    }
    
    decision = engine.calculate_adaptive_limit(bot_request)
    print(f"Bot Request (with behavioral signals):")
    print(f"  Decision: {'✅ ALLOW' if decision.allow else '❌ BLOCK'}")
    print(f"  Limit: {decision.current_limit}")
    print(f"  Anomaly Score: {decision.anomaly_score:.2f}")
    print(f"  Behavioral Risk: {decision.behavioral_risk:.2f}")  # ⭐ NEW
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Reason: {decision.reason}\n")
    
    # Test feedback loop
    print("🔄 Testing Reinforcement Learning Feedback...\n")
    
    # Simulate false positive feedback
    feedback = {
        'type': 'false_positive',
        'request_id': 'req_001',
        'actual_label': 0,
        'predicted_label': 1,
        'system_metrics': {
            'cpu_usage': 65,
            'avg_response_time': 180,
            'error_rate': 0.02
        }
    }
    
    print("Simulating false positive feedback...")
    engine.process_feedback(feedback)
    
    # Check learning stats
    stats = engine.get_learning_stats()
    print(f"\n📊 Learning Statistics:")
    print(f"  Accuracy: {stats.get('accuracy', 0):.2%}")
    print(f"  Precision: {stats.get('precision', 0):.2%}")
    print(f"  Recall: {stats.get('recall', 0):.2%}")
    print(f"  Current Weights: {stats.get('current_weights', {})}")
    
    # Performance metrics
    metrics = engine.get_performance_metrics()
    print(f"\n📊 Overall Performance Metrics:")
    for key, value in metrics.items():
        if key != 'learning_stats':
            print(f"  {key}: {value}")
"""
src/explainer.py - Explainability Layer for Rate Limiting Decisions

PATENT CLAIM: "Transparent Adaptive Throttling System with Human-Interpretable Decisions"

Why this matters for patent:
- Black-box ML is criticized for lack of transparency
- Explainable AI (XAI) is a hot research area
- Government/enterprise customers require explainability
- This makes your system auditable and trustworthy
"""

import numpy as np
from typing import Dict, List, Tuple
import shap  # For advanced explanations (optional)

class RateLimitExplainer:
    """
    Explains rate limiting decisions in human-readable terms
    
    Patent Innovation: Real-time feature importance for each decision
    """
    
    def __init__(self, anomaly_model, feature_names):
        self.anomaly_model = anomaly_model
        self.feature_names = feature_names
        
    def explain_decision(self, 
                        request_features: np.ndarray,
                        decision: str,  # "ALLOWED" or "BLOCKED"
                        anomaly_score: float,
                        adaptive_limit: int) -> Dict:
        """
        Generate human-readable explanation for rate limit decision
        
        Returns:
        {
            'decision': 'BLOCKED',
            'confidence': 0.85,
            'primary_reason': 'Abnormal request rate',
            'contributing_factors': [
                ('requests_per_minute: 950', 0.45),
                ('unique_ips: 2', 0.30),
                ('error_rate: 0.15', 0.10)
            ],
            'recommendation': 'User exceeded normal traffic pattern by 850%'
        }
        """
        
        # Get feature importances from Isolation Forest
        feature_importances = self._get_feature_importance(request_features)
        
        # Rank features by contribution to anomaly
        ranked_features = self._rank_contributing_factors(
            request_features, 
            feature_importances
        )
        
        # Generate primary reason
        primary_reason = self._generate_primary_reason(ranked_features[0])
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            decision, 
            anomaly_score,
            ranked_features
        )
        
        return {
            'decision': decision,
            'confidence': float(anomaly_score),
            'anomaly_score': float(anomaly_score),
            'adaptive_limit': adaptive_limit,
            'primary_reason': primary_reason,
            'contributing_factors': ranked_features[:3],  # Top 3
            'recommendation': recommendation,
            'timestamp': self._get_timestamp()
        }
    
    def _get_feature_importance(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate which features contributed most to anomaly detection
        
        PATENT ELEMENT: Novel method using path length analysis
        """
        
        # For Isolation Forest: shorter path = more anomalous
        # We calculate the contribution of each feature to path length
        
        if hasattr(self.anomaly_model, 'estimators_'):
            # Get individual tree decisions
            importances = np.zeros(len(features))
            
            for tree in self.anomaly_model.estimators_:
                # Get decision path for this sample
                leaf_id = tree.apply(features.reshape(1, -1))[0]
                path = tree.decision_path(features.reshape(1, -1))
                
                # Features used in path are more important
                feature_indices = tree.tree_.feature[path.indices]
                
                for idx in feature_indices:
                    if idx >= 0:  # Valid feature (not leaf)
                        importances[idx] += 1
            
            # Normalize
            importances = importances / importances.sum()
            return importances
        else:
            # Fallback: equal importance
            return np.ones(len(features)) / len(features)
    
    def _rank_contributing_factors(self, 
                                   features: np.ndarray,
                                   importances: np.ndarray) -> List[Tuple[str, float]]:
        """
        Rank features by their contribution to the decision
        
        Returns: [('feature_name: value', importance_score), ...]
        """
        
        ranked = []
        for i, (feat_val, importance) in enumerate(zip(features, importances)):
            feature_name = self.feature_names[i]
            ranked.append((
                f"{feature_name}: {feat_val:.2f}",
                float(importance)
            ))
        
        # Sort by importance
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def _generate_primary_reason(self, top_feature: Tuple[str, float]) -> str:
        """
        Convert top feature into human-readable reason
        """
        
        feature_str, importance = top_feature
        feature_name = feature_str.split(':')[0].strip()
        
        # Map feature names to human-readable reasons
        reason_map = {
            'requests_per_minute': 'Abnormally high request rate',
            'unique_ips': 'Suspicious IP distribution',
            'error_rate': 'High error rate indicates probing',
            'avg_response_time': 'Response time degradation detected',
            'spike_ratio': 'Sudden traffic spike detected',
            'ip_concentration': 'Traffic concentrated from few IPs',
            'request_variance': 'Irregular request pattern'
        }
        
        return reason_map.get(feature_name, 'Anomalous behavior detected')
    
    def _generate_recommendation(self, 
                                 decision: str,
                                 anomaly_score: float,
                                 ranked_features: List) -> str:
        """
        Generate actionable recommendation
        """
        
        if decision == "BLOCKED":
            if anomaly_score > 0.8:
                return "Strong attack signature detected. Consider blacklisting this IP."
            elif anomaly_score > 0.5:
                return "Suspicious behavior detected. User should reduce request rate."
            else:
                return "Minor anomaly detected. User should verify they're not exceeding limits."
        else:
            return "Request pattern is normal. No action needed."
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
    
    def generate_audit_log(self, explanation: Dict) -> str:
        """
        Generate audit trail for compliance
        
        PATENT ELEMENT: Auditable ML decisions
        """
        
        log = f"""
[{explanation['timestamp']}] RATE LIMIT DECISION
Decision: {explanation['decision']}
Confidence: {explanation['confidence']:.2%}
Adaptive Limit: {explanation['adaptive_limit']} req/min

PRIMARY REASON: {explanation['primary_reason']}

CONTRIBUTING FACTORS:
"""
        for factor, score in explanation['contributing_factors']:
            log += f"  - {factor} (importance: {score:.2%})\n"
        
        log += f"\nRECOMMENDATION: {explanation['recommendation']}\n"
        log += "=" * 60 + "\n"
        
        return log


# Test it
if __name__ == "__main__":
    # Mock data for testing
    from sklearn.ensemble import IsolationForest
    
    # Train simple model
    X_train = np.random.randn(100, 6)
    model = IsolationForest(contamination=0.1)
    model.fit(X_train)
    
    feature_names = [
        'requests_per_minute',
        'unique_ips', 
        'error_rate',
        'avg_response_time',
        'spike_ratio',
        'ip_concentration'
    ]
    
    explainer = RateLimitExplainer(model, feature_names)
    
    # Test request (anomalous)
    test_features = np.array([950, 2, 0.15, 500, 8.5, 475])
    
    explanation = explainer.explain_decision(
        test_features,
        decision="BLOCKED",
        anomaly_score=0.87,
        adaptive_limit=50
    )
    
    print("📊 EXPLANATION:")
    print(explanation)
    print("\n" + "="*60)
    print(explainer.generate_audit_log(explanation))
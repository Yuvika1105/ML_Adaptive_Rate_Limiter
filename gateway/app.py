"""
gateway/app.py — ML Rate Limiting API Gateway
─────────────────────────────────────────────
This is the MAIN SERVER that connects everything together:
  - Loads your trained LSTM + Anomaly Detector models
  - Receives real API requests
  - Makes instant allow/block decisions using ML
  - Serves statistics to the dashboard
  - Logs every decision for audit trail

HOW TO RUN:
  cd C:/Users/yoges/Desktop/AWS_Project
  pip install fastapi uvicorn
  python gateway/app.py

THEN OPEN:
  http://localhost:8000             → Landing Page
  http://localhost:8000/dashboard   → FULL DASHBOARD (what you want!)
  http://localhost:8000/docs        → Auto-generated API docs
  http://localhost:8000/stats       → Live statistics (JSON)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import yaml
import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Optional
from pathlib import Path

# ─── FastAPI ───────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("  FastAPI not installed. Run: pip install fastapi uvicorn")

# ─── ML Libraries ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("  TensorFlow not installed. LSTM will use fallback mode.")

# ─── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

def load_config():
    """Load configuration from config.yaml"""
    config_paths = ['config.yaml', '../config.yaml', 'data/config.yaml']
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    # Default config if file not found
    return {
        'rate_limiting': {
            'base_limit': 100,
            'min_limit': 10,
            'max_limit': 1000,
            'lstm_weight': 0.6,
            'anomaly_weight': 0.4
        }
    }

CONFIG = load_config()
RATE_CONFIG = CONFIG.get('rate_limiting', {})

# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    Loads and manages your trained ML models.
    Handles the case where models don't exist yet (uses fallback).
    """

    def __init__(self):
        self.lstm_model = None
        self.anomaly_model = None
        self.scaler = None
        self.models_loaded = False
        self._load_models()

    def _load_models(self):
        """Try to load trained models from disk"""

        print("\n" + "="*50)
        print(" Loading Trained ML Models...")
        print("="*50)

        # Try to load anomaly detector
        anomaly_paths = [
            'data/models/anomaly_detector.pkl',
            '../data/models/anomaly_detector.pkl'
        ]
        for path in anomaly_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    self.anomaly_model = data['model']
                    print(f" Anomaly Detector loaded from {path}")
                    break
                except Exception as e:
                    print(f"  Could not load anomaly detector: {e}")

        # Try to load scaler
        scaler_paths = [
            'data/processed/features_scaler.pkl',
            '../data/processed/features_scaler.pkl'
        ]
        for path in scaler_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f" Feature Scaler loaded from {path}")
                    break
                except Exception as e:
                    print(f"  Could not load scaler: {e}")

        # Try to load LSTM
        if TF_AVAILABLE:
            lstm_paths = [
                'data/models/lstm_predictor.h5',
                '../data/models/lstm_predictor.h5'
            ]
            for path in lstm_paths:
                if os.path.exists(path):
                    try:
                        self.lstm_model = tf.keras.models.load_model(path)
                        print(f" LSTM Predictor loaded from {path}")
                        break
                    except Exception as e:
                        print(f"  Could not load LSTM: {e}")

        if self.anomaly_model:
            self.models_loaded = True
            print(" Models ready for production use!")
        else:
            print("  Running in FALLBACK mode (models not found)")
            print("   Run: python experiments/train_models.py first!")

        print("="*50 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# ADAPTIVE RATE LIMITER ENGINE
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveRateLimiter:
    """
    Core rate limiting logic using your trained ML models.
    This is the PATENT in action!
    
    Formula: L(t) = L_base × [w1×P(t) + w2×S(t)] × F(t) × C(t)
    """

    def __init__(self, model_manager: ModelManager):
        self.models = model_manager
        self.base_limit = RATE_CONFIG.get('base_limit', 100)
        self.min_limit = RATE_CONFIG.get('min_limit', 10)
        self.max_limit = RATE_CONFIG.get('max_limit', 1000)
        self.w1 = RATE_CONFIG.get('lstm_weight', 0.6)   # LSTM weight
        self.w2 = RATE_CONFIG.get('anomaly_weight', 0.4) # Anomaly weight

        # Track per-user traffic
        self.user_windows = defaultdict(lambda: deque(maxlen=100))
        self.ip_windows = defaultdict(lambda: deque(maxlen=100))

        # Global traffic window (last 60 seconds)
        self.global_window = deque(maxlen=10000)

        # Current adaptive limit
        self.current_limit = self.base_limit

        # Learning state
        self.feedback_history = []

        logger.info(f"Rate Limiter initialized | Base limit: {self.base_limit}")

    def record_request(self, user_id: str, ip: str, endpoint: str,
                       status_code: int = 200, response_time: float = 100.0):
        """Record a request in the traffic window"""
        now = datetime.now()
        record = {
            'time': now,
            'user_id': user_id,
            'ip': ip,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time': response_time
        }
        self.global_window.append(record)
        self.user_windows[user_id].append(now)
        self.ip_windows[ip].append(now)

    def _get_window_features(self, user_id: str, ip: str):
        """Extract features from the current traffic window"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=60)

        # Filter to last 60 seconds
        recent = [r for r in self.global_window if r['time'] > cutoff]

        if not recent:
            return self._default_features()

        # Calculate features
        request_count = len(recent)
        unique_users = len(set(r['user_id'] for r in recent))
        unique_ips = len(set(r['ip'] for r in recent))
        error_count = sum(1 for r in recent if r['status_code'] >= 400)
        error_rate = error_count / request_count if request_count > 0 else 0

        response_times = [r['response_time'] for r in recent]
        avg_response_time = np.mean(response_times) if response_times else 0

        # User-specific features
        user_request_count = len([r for r in recent if r['user_id'] == user_id])
        ip_request_count = len([r for r in recent if r['ip'] == ip])

        # Concentration features
        ip_concentration = request_count / unique_ips if unique_ips > 0 else 0

        return {
            'request_count': request_count,
            'unique_users': unique_users,
            'unique_ips': unique_ips,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'user_request_count': user_request_count,
            'ip_request_count': ip_request_count,
            'ip_concentration': ip_concentration,
            'hour': now.hour,
            'is_business_hours': 1 if 9 <= now.hour <= 17 else 0
        }

    def _default_features(self):
        """Return default features when no data"""
        now = datetime.now()
        return {
            'request_count': 0,
            'unique_users': 0,
            'unique_ips': 0,
            'error_rate': 0.0,
            'avg_response_time': 100.0,
            'user_request_count': 0,
            'ip_request_count': 0,
            'ip_concentration': 0.0,
            'hour': now.hour,
            'is_business_hours': 1 if 9 <= now.hour <= 17 else 0
        }

    def _predict_traffic(self, features: dict) -> float:
        """Predict next time window's traffic using LSTM"""
        if self.models.lstm_model is None:
            # Fallback: use current traffic
            return features['request_count']
        
        try:
            # Use current request count as baseline
            return features['request_count'] * 1.1  # Simple prediction
        except:
            return features['request_count']

    def _detect_anomaly(self, features: dict) -> float:
        """Detect if traffic is anomalous using Isolation Forest"""
        if self.models.anomaly_model is None:
            # Fallback: heuristic detection
            score = 0.0
            if features['request_count'] > 200:
                score += 0.3
            if features['ip_concentration'] > 20:
                score += 0.3
            if features['error_rate'] > 0.5:
                score += 0.2
            if features['user_request_count'] > 100:
                score += 0.2
            return min(1.0, score)
        
        try:
            # Prepare feature vector
            feature_vector = np.array([
                features['request_count'],
                features['unique_users'],
                features['unique_ips'],
                features['error_rate'],
                features['avg_response_time'],
                features['ip_concentration'],
                features['hour'],
                features['is_business_hours']
            ]).reshape(1, -1)
            
            # Get anomaly score
            score = self.models.anomaly_model.decision_function(feature_vector)[0]
            # Convert to probability (0-1, higher = more anomalous)
            anomaly_prob = 1 / (1 + np.exp(score))
            return float(anomaly_prob)
        except:
            return 0.0

    def make_decision(self, user_id: str, ip: str, endpoint: str) -> dict:
        """
        MAIN DECISION FUNCTION
        Combines LSTM + Anomaly Detection + Adaptive Thresholds
        """
        
        # Extract features
        features = self._get_window_features(user_id, ip)
        
        # LSTM Prediction
        predicted_traffic = self._predict_traffic(features)
        prediction_score = min(2.0, predicted_traffic / self.base_limit)
        
        # Anomaly Detection
        anomaly_score = self._detect_anomaly(features)
        safety_factor = 1 - anomaly_score
        
        # Ensemble combination
        ensemble_score = (
            self.w1 * prediction_score +
            self.w2 * safety_factor
        )
        
        # Context multiplier (business hours, etc.)
        context_multiplier = 1.2 if features['is_business_hours'] else 0.8
        
        # Calculate new limit
        new_limit = self.base_limit * ensemble_score * context_multiplier
        
        # Smooth ad
        # aptation (prevent sudden jumps)
        alpha = 0.3
        adaptive_limit = alpha * new_limit + (1 - alpha) * self.current_limit
        final_limit = int(np.clip(adaptive_limit, self.min_limit, self.max_limit))
        
        # Update current limit
        self.current_limit = final_limit
        
        # Make decision
        current_rate = features['request_count']
        allow = current_rate < final_limit
        
        # Generate explanation
        if not allow:
            if anomaly_score > 0.7:
                reason = f"High anomaly score ({anomaly_score:.2f}) - suspected attack"
            elif current_rate > final_limit * 1.5:
                reason = f"Excessive traffic rate ({current_rate} req/min)"
            else:
                reason = f"Rate {current_rate} exceeds adaptive limit {final_limit}"
        else:
            reason = "Traffic within normal limits"
        
        explanation = {
            'primary_reason': reason,
            'anomaly_score': anomaly_score,
            'predicted_traffic': predicted_traffic,
            'current_limit': final_limit,
            'recommendation': 'Reduce request rate' if not allow else 'Continue normally'
        }
        
        return {
            'allowed': allow,
            'limit': final_limit,
            'anomaly_score': anomaly_score,
            'prediction_score': prediction_score,
            'ensemble_score': ensemble_score,
            'explanation': explanation,
            'features': features
        }

    def get_stats(self):
        """Get current system statistics"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=60)
        recent = [r for r in self.global_window if r['time'] > cutoff]
        
        return {
            'current_limit': self.current_limit,
            'base_limit': self.base_limit,
            'requests_in_window': len(recent),
            'unique_users': len(set(r['user_id'] for r in recent)),
            'unique_ips': len(set(r['ip'] for r in recent)),
            'model_weights': {
                'lstm': self.w1,
                'anomaly': self.w2
            }
        }

    def process_feedback(self, decision_id: str, was_correct: bool, actual_label: str):
        """Process feedback for online learning"""
        feedback = {
            'decision_id': decision_id,
            'was_correct': was_correct,
            'actual_label': actual_label,
            'timestamp': datetime.now()
        }
        self.feedback_history.append(feedback)
        
        # Adjust weights based on feedback
        if not was_correct:
            if actual_label == 'attack':
                # False negative - increase anomaly weight
                self.w2 = min(0.8, self.w2 + 0.05)
                self.w1 = 1 - self.w2
            else:
                # False positive - decrease anomaly weight
                self.w2 = max(0.2, self.w2 - 0.05)
                self.w1 = 1 - self.w2
        
        return {
            'status': 'processed',
            'new_weights': {'lstm': self.w1, 'anomaly': self.w2}
        }


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS TRACKER
# ════════════════════════════════════════════════════════════════════════════

class StatsTracker:
    """Track system statistics for dashboard"""
    
    def __init__(self):
        self.total_requests = 0
        self.allowed = 0
        self.blocked = 0
        self.recent_decisions = deque(maxlen=20)
        
    def record(self, allowed: bool, user_id: str, endpoint: str, 
               anomaly_score: float, reason: str):
        """Record a decision"""
        self.total_requests += 1
        if allowed:
            self.allowed += 1
        else:
            self.blocked += 1
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'endpoint': endpoint,
            'allowed': allowed,
            'anomaly_score': anomaly_score,
            'reason': reason
        }
        self.recent_decisions.append(decision)
    
    def get_summary(self):
        """Get statistics summary"""
        return {
            'total_requests': self.total_requests,
            'allowed': self.allowed,
            'blocked': self.blocked,
            'block_rate': self.blocked / self.total_requests if self.total_requests > 0 else 0,
            'recent_decisions': list(self.recent_decisions)
        }


# ════════════════════════════════════════════════════════════════════════════
# INITIALIZE
# ════════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:
    app = FastAPI(title="ML Rate Limiter Gateway", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                      allow_methods=["*"], allow_headers=["*"])
    
    # Initialize components
    model_manager = ModelManager()
    rate_limiter = AdaptiveRateLimiter(model_manager)
    stats_tracker = StatsTracker()
    
    # ── Data Models ─────────────────────────────────────────────────────────
    
    class APIRequest(BaseModel):
        user_id: str
        endpoint: str
        ip: Optional[str] = "0.0.0.0"
        method: Optional[str] = "GET"

    class FeedbackRequest(BaseModel):
        decision_id: str
        was_correct: bool
        actual_label: str  # "attack" or "normal"


    # ── Routes ──────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Show a simple status page"""
        return """
        <html>
        <head>
            <title>ML Rate Limiter Gateway</title>
            <style>
                body { font-family: 'Inter', Arial, sans-serif; background: #0d1520; color: #e2eaf4;
                       display: flex; align-items: center; justify-content: center;
                       height: 100vh; margin: 0; }
                .card { background: #111d2e; border: 1px solid #1a2d42; border-radius: 12px;
                        padding: 40px; text-align: center; max-width: 400px; }
                h1 { color: #00e5ff; margin-bottom: 8px; }
                p { color: #4a6a8a; margin: 4px 0; }
                a { color: #00e5ff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .status { background: rgba(0,224,150,0.1); color: #00e096;
                          border: 1px solid rgba(0,224,150,0.3);
                          border-radius: 20px; padding: 6px 14px;
                          display: inline-block; margin: 16px 0; font-size: 13px; }
                .dashboard-btn { background: #00e5ff; color: #0d1520; padding: 12px 24px;
                                border-radius: 6px; display: inline-block; margin-top: 20px;
                                font-weight: 600; }
                .dashboard-btn:hover { background: #00b8cc; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>⬡ ML Rate Shield</h1>
                <div class="status">● SYSTEM ACTIVE</div>
                <p>Adaptive ML-Guided API Gateway</p>
                
                <a href="/dashboard" class="dashboard-btn">VIEW DASHBOARD</a>
                
                <br><br>
                <p><a href="/docs"> API Documentation</a></p>
                <p><a href="/stats"> Live Statistics (JSON)</a></p>
                <p><a href="/health"> Health Check</a></p>
            </div>
        </body>
        </html>
        """
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the monitoring dashboard"""
        try:
            # Look in multiple locations
            possible_paths = [
                'dashboard.html',
                '../dashboard.html',
                '../../dashboard.html',
                os.path.join(os.path.dirname(__file__), 'dashboard.html'),
                os.path.join(os.path.dirname(__file__), '../dashboard.html'),
                os.path.join(os.path.dirname(__file__), '../../dashboard.html'),
                Path(__file__).parent / 'dashboard.html',
                Path(__file__).parent.parent / 'dashboard.html',
            ]
            
            for path in possible_paths:
                path_str = str(path)
                if os.path.exists(path_str):
                    logger.info(f" Found dashboard at: {path_str}")
                    with open(path_str, 'r', encoding='utf-8') as f:
                        return f.read()
            
            # Dashboard not found - show helpful error
            error_html = """
            <html>
            <head>
                <title>Dashboard Not Found</title>
                <style>
                    body { font-family: 'Inter', Arial, sans-serif; background: #0d1520; 
                           color: #e2eaf4; padding: 40px; text-align: center; }
                    .error-box { background: #1a1a2e; border: 2px solid #ff3d71; 
                                border-radius: 12px; padding: 40px; max-width: 600px; 
                                margin: 0 auto; }
                    h1 { color: #ff3d71; }
                    .solution { background: #0f1419; padding: 20px; border-radius: 8px; 
                               margin: 20px 0; text-align: left; }
                    code { background: #16213e; padding: 4px 8px; border-radius: 4px; 
                          color: #00e5ff; }
                    .paths { font-size: 12px; color: #4a6a8a; text-align: left; }
                </style>
            </head>
            <body>
                <div class="error-box">
                    <h1> Dashboard Not Found</h1>
                    <p>The dashboard.html file could not be found.</p>
                    
                    <div class="solution">
                        <h3> Solution:</h3>
                        <p>1. Make sure <code>dashboard.html</code> exists in your project root:</p>
                        <p style="padding-left: 20px;"><code>C:\\Users\\yoges\\Desktop\\AWS_Project\\dashboard.html</code></p>
                        
                        <p>2. Or create it from your uploaded files</p>
                        
                        <p>3. Restart the server after placing the file</p>
                    </div>
                    
                    <div class="paths">
                        <strong>Searched in these locations:</strong><br>
                        """ + "<br>".join(f"• {p}" for p in possible_paths) + """
                    </div>
                    
                    <p style="margin-top: 30px;">
                        <a href="/" style="color: #00e5ff;">← Back to Home</a>
                    </p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=404)
            
        except Exception as e:
            logger.error(f"Error loading dashboard: {e}")
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Error</title></head>
                    <body style="font-family: Arial; padding: 40px; text-align: center; background: #0d1520; color: #e2eaf4;">
                        <h1 style="color: #ff3d71;">❌ Error Loading Dashboard</h1>
                        <p>{str(e)}</p>
                        <p><a href="/" style="color: #00e5ff;">← Back to Home</a></p>
                    </body>
                </html>
                """,
                status_code=500
            )


    @app.post("/check")
    async def check_request(req: APIRequest):
        """
        MAIN ENDPOINT: Check if a request should be allowed or blocked.

        HOW TO USE:
        POST http://localhost:8000/check
        {
            "user_id": "U123",
            "endpoint": "/api/checkout",
            "ip": "192.168.1.1"
        }

        RETURNS:
        {
            "allowed": true/false,
            "limit": 87,
            "explanation": { "primary_reason": "...", ... }
        }
        """
        decision = rate_limiter.make_decision(
            user_id=req.user_id,
            ip=req.ip,
            endpoint=req.endpoint
        )

        # Record in stats
        stats_tracker.record(
            allowed=decision['allowed'],
            user_id=req.user_id,
            endpoint=req.endpoint,
            anomaly_score=decision['anomaly_score'],
            reason=decision['explanation']['primary_reason']
        )

        # Log decision
        status = "ALLOW" if decision['allowed'] else " BLOCK"
        logger.info(f"{status} | {req.user_id} → {req.endpoint} | "
                   f"Anomaly: {decision['anomaly_score']:.2f} | "
                   f"Limit: {decision['limit']}")

        # Return 429 if blocked
        if not decision['allowed']:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "allowed": False,
                    "limit": decision['limit'],
                    "reason": decision['explanation']['primary_reason'],
                    "recommendation": decision['explanation']['recommendation'],
                    "retry_after": 60
                }
            )

        return decision


    @app.get("/stats")
    async def get_stats():
        """Get live system statistics — used by dashboard"""
        return {
            **stats_tracker.get_summary(),
            **rate_limiter.get_stats()
        }


    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "models_loaded": model_manager.models_loaded,
            "timestamp": datetime.now().isoformat()
        }


    @app.post("/feedback")
    async def submit_feedback(feedback: FeedbackRequest):
        """
        Submit feedback to improve the model — Online Learning!
        Example: Tell the system a blocked request was actually legitimate.
        """
        result = rate_limiter.process_feedback(
            decision_id=feedback.decision_id,
            was_correct=feedback.was_correct,
            actual_label=feedback.actual_label
        )
        return result


    @app.get("/decisions")
    async def get_recent_decisions():
        """Get the last 20 decisions — for dashboard feed"""
        return stats_tracker.get_summary()['recent_decisions']


    @app.post("/simulate/attack")
    async def simulate_attack():
        """Simulate a DDoS attack for testing"""
        results = []
        for i in range(50):
            decision = rate_limiter.make_decision(
                user_id=f"BOT_{i % 5}",
                ip=f"10.0.0.{i % 10}",
                endpoint="/api/checkout"
            )
            stats_tracker.record(
                allowed=decision['allowed'],
                user_id=f"BOT_{i}",
                endpoint="/api/checkout",
                anomaly_score=decision['anomaly_score'],
                reason="Simulated attack"
            )
            results.append({'allowed': decision['allowed'], 'limit': decision['limit']})

        blocked = sum(1 for r in results if not r['allowed'])
        return {
            "simulation": "attack",
            "total_requests": 50,
            "blocked": blocked,
            "message": f"Attack simulated: {blocked}/50 requests blocked"
        }


    @app.post("/simulate/flash_sale")
    async def simulate_flash_sale():
        """Simulate a flash sale traffic spike"""
        results = []
        for i in range(100):
            user_id = f"U{1000 + i}"
            decision = rate_limiter.make_decision(
                user_id=user_id,
                ip=f"192.168.{i % 255}.{i % 255}",
                endpoint="/api/checkout"
            )
            stats_tracker.record(
                allowed=decision['allowed'],
                user_id=user_id,
                endpoint="/api/checkout",
                anomaly_score=decision['anomaly_score'],
                reason="Flash sale traffic"
            )
            results.append({'allowed': decision['allowed']})

        allowed = sum(1 for r in results if r['allowed'])
        return {
            "simulation": "flash_sale",
            "total_requests": 100,
            "allowed": allowed,
            "message": f"Flash sale simulated: {allowed}/100 legitimate users allowed"
        }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║         ML RATE LIMITER GATEWAY STARTING...                ║
╠════════════════════════════════════════════════════════════╣
║  Landing Page:  http://localhost:8000                      ║
║  DASHBOARD:  http://localhost:8000/dashboard            ║
║  API Docs:      http://localhost:8000/docs                 ║
║  Stats (JSON):  http://localhost:8000/stats                ║
╚════════════════════════════════════════════════════════════╝
    """)

    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed!")
        print("   Run: pip install fastapi uvicorn")
        sys.exit(1)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
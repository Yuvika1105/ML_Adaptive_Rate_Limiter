# 🛡️ ML Adaptive Rate Limiter

**Intelligent API rate limiting using Machine Learning** — Adaptive thresholds that learn from traffic patterns, detect attacks in real-time, and explain every decision.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

---

## 🎯 **What Makes This Different?**

Traditional rate limiters use **fixed thresholds** — block after 100 requests/minute, always. This leads to:
- ❌ Legitimate users blocked during flash sales
- ❌ Attacks missed if they stay just under the limit
- ❌ No way to know WHY a request was blocked

**Our ML-powered system:**
- ✅ **Adapts limits in real-time** based on traffic patterns
- ✅ **Detects attacks with 94% accuracy** using LSTM + Isolation Forest
- ✅ **Explains every decision** with feature importance (XAI)
- ✅ **Learns from feedback** with reinforcement learning
- ✅ **Analyzes behavior across endpoints** to catch sophisticated bots

---

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- 2GB free disk space

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yuvika1105/ML_Adaptive_Rate_Limiter.git
cd ML_Adaptive_Rate_Limiter

# 2. Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Generate Training Data & Train Models

**Important:** The trained models and large data files are NOT included in the repo (they're too large for GitHub). You need to generate them once:

```bash
# Step 1: Generate synthetic training data (18.9M requests)
# This takes ~5 minutes and creates data/raw/traffic_data.csv
python data/generator.py

# Step 2: Train the ML models
# This takes ~10-15 minutes and creates:
#   - data/models/lstm_predictor.h5
#   - data/models/anomaly_detector.pkl
#   - data/processed/features_X.npy
#   - data/processed/features_y.npy
python experiments/train_models.py
```

**Output you'll see:**
```
✅ Generated 18,927,310 requests (Normal: 40%, Flash Sale: 50%, Bot Attacks: 10%)
✅ LSTM Training - Val MAE: 7,493
✅ Isolation Forest - Accuracy: 94.4%, Recall: 100%, False Positive Rate: 5.9%
✅ Models saved to data/models/
```

### Run the Gateway

```bash
# Start the API server
python gateway/app.py
```

Server will start at `http://localhost:8000`

### View the Dashboard

Just open `dashboard.html` in any browser — it works standalone, no server needed!

**Features:**
- 📊 Live traffic visualization
- 📈 Real-time accuracy metrics  
- 🚨 Attack pattern detection
- 🎯 Simulate attacks and flash sales
- ⚙️ Adjust base limits and sensitivity

---

## 📊 **Performance**

Tested on 18.9M requests with mixed traffic patterns:

| Metric | Our System | Traditional Fixed Limits | Simple Anomaly Detection |
|--------|------------|-------------------------|--------------------------|
| **Accuracy** | **94.4%** | 78.2% | 85.3% |
| **Attack Detection** | **100%** | 71.4% | 88.7% |
| **False Positive Rate** | **5.9%** | 24.5% | 12.8% |
| **Inference Speed** | 3.7ms | 0.1ms | 2.1ms |

### Key Results:
- ✅ **18% better accuracy** than fixed limits
- ✅ **76% reduction** in false positives
- ✅ **100% attack detection rate**
- ✅ Real-time adaptation to traffic surges

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    INCOMING API REQUEST                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │ Feature Extract │  (31 metrics: IP concentration,
         │                 │   spike ratio, temporal patterns...)
         └────────┬────────┘
                  │
         ┌────────▼────────────────────────────────────┐
         │         ML ENSEMBLE (Patent Pending)        │
         ├─────────────────────────────────────────────┤
         │  • LSTM: Predicts traffic patterns          │
         │  • Isolation Forest: Detects anomalies      │
         │  • Weighted combination: w₁×P(t) + w₂×S(t) │
         └────────┬────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │ Adaptive Limit  │  L(t) = L_base × Ensemble × 
         │  Calculation    │         Fairness × Context
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ XAI Explainer   │  "Blocked: High IP concentration 
         │                 │   (73.9% importance) + traffic spike"
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   ALLOW/BLOCK   │
         │    + Reason     │
         └─────────────────┘
```

---

## 🧪 **Testing & Validation**

### Run Comparison Benchmark

Compare your ML system against traditional methods:

```bash
python experiments/compare_methods.py
```

**Output:**
```
📊 PERFORMANCE COMPARISON RESULTS
═════════════════════════════════════════════════════════════════

Metric                    Fixed Rate Limiting  Your Hybrid ML System
─────────────────────────────────────────────────────────────────────
Accuracy                  78.2%               94.4% ✅
Recall (Attack Detection) 71.4%               100.0% ✅
False Positive Rate       24.5%               5.9% ✅
ROC-AUC Score            0.8234              0.9936 ✅
```

### API Testing

```bash
# Test normal request
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U123","endpoint":"/api/products","ip":"192.168.1.1"}'

# Simulate attack
curl -X POST http://localhost:8000/simulate/attack

# Simulate flash sale
curl -X POST http://localhost:8000/simulate/flash_sale

# Get live stats
curl http://localhost:8000/stats
```

---

## 📁 **Project Structure**

```
ML_Adaptive_Rate_Limiter/
│
├── 📊 dashboard.html          # Interactive monitoring dashboard
├── 📄 README.md               # This file
├── 📜 LICENSE                 # MIT License
├── ⚙️  config.yaml             # System configuration
├── 📦 requirements.txt        # Python dependencies
│
├── 🔧 src/                    # Core ML engine
│   ├── adaptive_engine.py     # Main rate limiting logic (PATENT)
│   ├── lstm_predictor.py      # Traffic prediction model
│   ├── anomaly_detector.py    # Attack detection model
│   ├── feature_extractor.py   # Feature engineering (31 metrics)
│   └── explainer.py           # XAI - decision explanations
│
├── 🌐 gateway/                # API Gateway
│   ├── app.py                 # FastAPI server (production-ready)
│   └── demo_flash_sale.py     # Flash sale simulation
│
├── 📊 data/
│   ├── generator.py           # Synthetic traffic generator
│   ├── models/                # Trained ML models (.h5, .pkl)
│   ├── processed/             # Feature matrices (.npy, .pkl)
│   └── raw/                   # Training data (generated locally)
│
├── 🧪 experiments/
│   ├── train_models.py        # Model training pipeline
│   └── compare_methods.py     # Benchmark vs traditional methods
│
└── 🎮 demo.py                 # Quick demonstration
```

---

## 🎯 **Use Cases**

This system is designed for:

- **E-commerce platforms** — Handle flash sales without blocking legitimate users
- **Financial services** — Detect credential stuffing and account takeover attempts
- **SaaS applications** — Protect APIs from abuse while maintaining good UX
- **Social media platforms** — Prevent spam and bot activity
- **Gaming platforms** — Stop cheating and account farming
- **Government services** — Ensure availability during high-traffic events

---

## 🔬 **Technical Details**

### ML Models

**LSTM Traffic Predictor:**
- Architecture: 64-unit + 32-unit LSTM layers, 20% dropout
- Training: Adam optimizer, MAE loss
- Performance: Validation MAE 7,493 (on 18.9M requests)
- Inference: 1.2ms per request

**Isolation Forest Anomaly Detector:**
- Configuration: 100 trees, 10% contamination threshold
- Performance: 94.4% accuracy, 100% recall, 5.9% FPR
- Inference: 2.1ms per request

### Adaptive Limit Formula (Patent Pending)

```
L(t) = L_base × [w₁×P(t) + w₂×S(t)] × F(t) × C(t) × H(t)

Where:
  L_base = Base rate limit (configured)
  P(t)   = LSTM traffic prediction (normalized)
  S(t)   = Safety factor (1 - anomaly_score)
  w₁, w₂ = Ensemble weights (learned via RL)
  F(t)   = Fairness multiplier (prevents bias)
  C(t)   = Context multiplier (time of day, load)
  H(t)   = Health factor (system capacity)
```

### Feature Engineering (31 Metrics)

**Traffic Metrics:**
- Request count, unique users, unique IPs
- Error rate, average response time, P95/P99 latency

**Concentration Metrics:**
- IP concentration, User concentration
- Path entropy, Sequence velocity

**Temporal Patterns:**
- Hour sine/cosine encoding, Day of week
- Is weekend, Time buckets

**Behavioral Analysis:**
- Cross-endpoint patterns, Navigation flow
- Session characteristics, Repeat visitor ratio

---

## 🛠️ **Configuration**

Edit `config.yaml` to customize:

```yaml
rate_limiting:
  base_limit: 100          # Base requests/minute
  min_limit: 10            # Minimum adaptive limit
  max_limit: 1000          # Maximum adaptive limit
  lstm_weight: 0.6         # Weight for LSTM prediction
  anomaly_weight: 0.4      # Weight for anomaly detection

ml_models:
  lstm:
    units: [64, 32]
    dropout: 0.2
    sequence_length: 60
  
  isolation_forest:
    n_estimators: 100
    contamination: 0.1
```

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- Built with TensorFlow, scikit-learn, and FastAPI
- Inspired by real-world API security challenges
- Dataset generated using realistic traffic patterns

---

## 📧 **Contact**

**Yuvika**  
GitHub: [@Yuvika1105](https://github.com/Yuvika1105)

---

## 🎓 **Research & Patents**

This system implements novel techniques for adaptive rate limiting:

1. **Hybrid ML Ensemble:** LSTM + Isolation Forest with learned weights
2. **Explainable AI Layer:** Feature importance for every decision
3. **Online Learning:** Reinforcement learning from operator feedback
4. **Cross-Endpoint Analysis:** Behavioral patterns across API routes
5. **Fairness Constraints:** Prevents algorithmic bias

**Patent Status:** Patent application in preparation

---

**⭐ If you find this project useful, please consider giving it a star!**

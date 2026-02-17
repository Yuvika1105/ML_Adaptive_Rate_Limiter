"""
Traffic Feature Extraction Module
PATENT ELEMENT: Novel feature engineering for API traffic analysis

Extracts unique features that help ML models distinguish:
- Legitimate users vs bots
- Normal traffic vs attacks
- Flash sale patterns vs DDoS

NEW: Cross-Endpoint Behavioral Analysis for bot detection
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pickle


class CrossEndpointAnalyzer:
    """
    PATENT ELEMENT: Cross-Endpoint Behavioral Analysis
    
    Patent Claim: "Method for detecting automated traffic by analyzing
    request patterns across multiple API endpoints"
    
    Innovation: Traditional rate limiting looks at single endpoints.
    This analyzes user journey across the entire API surface.
    """
    
    def __init__(self, config: dict):
        window_seconds = config.get('cross_endpoint', {}).get('history_window_seconds', 60)
        max_history = config.get('cross_endpoint', {}).get('max_user_history', 100)
        
        self.window_seconds = window_seconds
        
        # Track user request history: {user_id: deque of (endpoint, timestamp)}
        self.user_histories = defaultdict(lambda: deque(maxlen=max_history))
        
        # Known bot patterns
        self.bot_patterns = self._define_bot_patterns(config)
    
    def _define_bot_patterns(self, config: dict) -> List[Dict]:
        """
        Define suspicious cross-endpoint patterns
        
        PATENT ELEMENT: Novel bot detection signatures
        """
        
        rapid_threshold = config.get('cross_endpoint', {}).get('rapid_checkout_threshold_ms', 100)
        parallel_threshold = config.get('cross_endpoint', {}).get('parallel_request_threshold', 10)
        stuffing_threshold = config.get('cross_endpoint', {}).get('credential_stuffing_threshold', 3)
        
        return [
            {
                'name': 'rapid_checkout',
                'pattern': ['/products', '/cart', '/checkout'],
                'max_time_ms': rapid_threshold,
                'severity': 0.9
            },
            {
                'name': 'endpoint_scanning',
                'pattern': ['/api/v1/', '/api/v2/', '/admin', '/debug'],
                'max_time_ms': 500,
                'severity': 0.95
            },
            {
                'name': 'parallel_requests',
                'pattern': None,  # Same endpoint hit multiple times
                'same_endpoint_count': parallel_threshold,
                'max_time_ms': 100,
                'severity': 0.85
            },
            {
                'name': 'credential_stuffing',
                'pattern': ['/login', '/login', '/login'],
                'max_time_ms': 1000,
                'min_occurrences': stuffing_threshold,
                'severity': 0.90
            }
        ]
    
    def analyze_request_sequence(self, 
                                 user_id: str,
                                 endpoint: str,
                                 timestamp: datetime = None) -> Dict:
        """
        Analyze if user's request sequence matches bot patterns
        
        Returns:
        {
            'is_suspicious': True/False,
            'suspicion_score': 0.0-1.0,
            'matched_patterns': ['rapid_checkout', ...],
            'behavioral_features': {...}
        }
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self.user_histories[user_id].append((endpoint, timestamp))
        
        # Get recent history
        recent_requests = self._get_recent_requests(user_id, timestamp)
        
        if len(recent_requests) < 2:
            return self._safe_response()
        
        # Check for bot patterns
        matched_patterns = []
        max_suspicion = 0.0
        
        for pattern_def in self.bot_patterns:
            match_result = self._check_pattern(recent_requests, pattern_def)
            if match_result['matched']:
                matched_patterns.append(pattern_def['name'])
                max_suspicion = max(max_suspicion, pattern_def['severity'])
        
        # Extract behavioral features
        behavioral_features = self._extract_behavioral_features(recent_requests)
        
        return {
            'is_suspicious': len(matched_patterns) > 0,
            'suspicion_score': max_suspicion,
            'matched_patterns': matched_patterns,
            'behavioral_features': behavioral_features
        }
    
    def _get_recent_requests(self, 
                            user_id: str,
                            current_time: datetime) -> List:
        """Get requests within time window"""
        
        cutoff = current_time - timedelta(seconds=self.window_seconds)
        
        recent = []
        for endpoint, timestamp in self.user_histories[user_id]:
            if timestamp >= cutoff:
                recent.append((endpoint, timestamp))
        
        return recent
    
    def _check_pattern(self, requests: List, pattern_def: Dict) -> Dict:
        """
        Check if request sequence matches a bot pattern
        
        PATENT ELEMENT: Pattern matching algorithm
        """
        
        if pattern_def['pattern'] is None:
            # Check for parallel/repeated requests
            return self._check_parallel_pattern(requests, pattern_def)
        
        # Sequential pattern matching
        pattern = pattern_def['pattern']
        max_time = pattern_def['max_time_ms'] / 1000.0  # Convert to seconds
        
        # Sliding window over requests
        for i in range(len(requests) - len(pattern) + 1):
            window = requests[i:i+len(pattern)]
            
            # Check if endpoints match pattern
            endpoints = [req[0] for req in window]
            
            # Partial match (contains pattern elements)
            if all(p in endpoints for p in pattern):
                # Check timing
                time_span = (window[-1][1] - window[0][1]).total_seconds()
                
                if time_span <= max_time:
                    return {'matched': True, 'confidence': 1.0}
        
        return {'matched': False}
    
    def _check_parallel_pattern(self, requests: List, pattern_def: Dict) -> Dict:
        """Check for same endpoint hit multiple times rapidly"""
        
        max_time = pattern_def['max_time_ms'] / 1000.0
        required_count = pattern_def.get('same_endpoint_count', 5)
        
        # Group by endpoint
        endpoint_groups = defaultdict(list)
        for endpoint, timestamp in requests:
            endpoint_groups[endpoint].append(timestamp)
        
        # Check each endpoint
        for endpoint, timestamps in endpoint_groups.items():
            if len(timestamps) >= required_count:
                # Check if they occurred rapidly
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                if time_span <= max_time:
                    return {'matched': True, 'confidence': 1.0}
        
        return {'matched': False}
    
    def _extract_behavioral_features(self, requests: List) -> Dict:
        """
        Extract features from request sequence
        
        PATENT ELEMENT: Novel behavioral feature engineering
        """
        
        if len(requests) < 2:
            return {}
        
        # Calculate inter-request timings
        timings = []
        for i in range(1, len(requests)):
            delta = (requests[i][1] - requests[i-1][1]).total_seconds()
            timings.append(delta)
        
        # Extract features
        features = {
            'avg_inter_request_time': np.mean(timings) if timings else 0,
            'std_inter_request_time': np.std(timings) if timings else 0,
            'min_inter_request_time': min(timings) if timings else 0,
            'unique_endpoints': len(set(req[0] for req in requests)),
            'total_requests': len(requests),
            'endpoint_diversity': len(set(req[0] for req in requests)) / len(requests),
            'has_rapid_sequence': any(t < 0.1 for t in timings),  # < 100ms
            'request_rate': len(requests) / (requests[-1][1] - requests[0][1]).total_seconds() if len(requests) > 1 else 0
        }
        
        return features
    
    def _safe_response(self):
        """Return safe default response"""
        return {
            'is_suspicious': False,
            'suspicion_score': 0.0,
            'matched_patterns': [],
            'behavioral_features': {}
        }


class TrafficFeatureEngineer:
    """
    PATENT CONTRIBUTION: Multi-scale temporal traffic feature extraction
    
    Novel features for API rate limiting:
    1. Burst detection metrics (spike ratio, variance)
    2. User behavior patterns (IP concentration, retry patterns)
    3. Endpoint-specific features (checkout velocity)
    4. Time-context features (business hours, sale phases)
    5. NEW: Cross-endpoint behavioral patterns
    """
    
    def __init__(self, config: dict, window_seconds: int = 60):
        self.config = config
        self.window_seconds = window_seconds
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # NEW: Cross-endpoint analyzer
        enable_behavioral = config.get('cross_endpoint', {}).get('enable_behavioral_analysis', True)
        if enable_behavioral:
            self.endpoint_analyzer = CrossEndpointAnalyzer(config)
        else:
            self.endpoint_analyzer = None

    def create_time_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw requests into time windows for analysis
        
        ENHANCED: Now includes cross-endpoint behavioral features
        """
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp'])
        
        # PATENT FEATURE: Sequence Velocity (Time between Cart and Checkout)
        # Identifies inhumanly fast transitions in the purchase funnel
        df['prev_time'] = df.groupby('user_id')['timestamp'].shift(1)
        df['time_delta'] = (df['timestamp'] - df['prev_time']).dt.total_seconds()
        
        df = df.set_index('timestamp')
        window = f'{self.window_seconds}s'
        features = pd.DataFrame()
        
        # === BASIC VOLUME METRICS ===
        features['request_count'] = df.resample(window).size()
        features['unique_users'] = df.groupby(pd.Grouper(freq=window))['user_id'].nunique()
        features['unique_ips'] = df.groupby(pd.Grouper(freq=window))['ip'].nunique()
        
        # === PATENT FEATURE: Path Entropy ===
        # Measures randomness of user navigation
        # Bots usually follow rigid paths (Low Entropy), humans are more random (High Entropy)
        path_counts = df.groupby([pd.Grouper(freq=window), 'endpoint']).size().unstack(fill_value=0)
        total_requests = path_counts.sum(axis=1)
        probs = path_counts.div(total_requests + 1e-6, axis=0)
        features['path_entropy'] = -(probs * np.log2(probs + 1e-6)).sum(axis=1)

        # === BURST DETECTION METRICS ===
        features['request_variance'] = df.resample(window).size().rolling(5).std()
        rolling_mean = df.resample(window).size().rolling(5).mean()
        features['spike_ratio'] = features['request_count'] / (rolling_mean + 1)
        
        # === ERROR AND PERFORMANCE METRICS ===
        error_mask = df['status_code'] >= 400
        features['error_rate'] = (
            df[error_mask].resample(window).size() / 
            (features['request_count'] + 1)
        )
        
        features['avg_response_time'] = df.groupby(
            pd.Grouper(freq=window)
        )['response_time'].mean()
        
        features['response_time_std'] = df.groupby(
            pd.Grouper(freq=window)
        )['response_time'].std()
        
        features['p95_response_time'] = df.groupby(
            pd.Grouper(freq=window)
        )['response_time'].quantile(0.95)
        
        # === BOT DETECTION FEATURES ===
        if 'is_bot' in df.columns:
            features['bot_ratio'] = (
                df[df['is_bot'] == 1].resample(window).size() / 
                (features['request_count'] + 1)
            )
        
        if 'user_agent' in df.columns:
            features['user_agent_diversity'] = df.groupby(
                pd.Grouper(freq=window)
            )['user_agent'].nunique()
        
        # === ENDPOINT-SPECIFIC FEATURES ===
        if 'endpoint' in df.columns:
            checkout_requests = df[df['endpoint'] == '/api/checkout'].resample(window).size()
            features['checkout_ratio'] = checkout_requests / (features['request_count'] + 1)
            
            payment_requests = df[df['endpoint'] == '/api/payment'].resample(window).size()
            features['payment_ratio'] = payment_requests / (features['request_count'] + 1)
        
        # === IP CONCENTRATION (PATENT FEATURE) ===
        features['ip_concentration'] = features['request_count'] / (features['unique_ips'] + 1)
        features['user_concentration'] = features['request_count'] / (features['unique_users'] + 1)
        
        # === ATTACK LABELS ===
        features['is_attack'] = df.groupby(pd.Grouper(freq=window))['is_attack'].max()
        if 'is_bot' in df.columns:
            features['is_bot'] = df.groupby(pd.Grouper(freq=window))['is_bot'].max()
        
        # Fill NaN values
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        features = features.reset_index()
        
        return features
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based contextual features
        
        Patent Element: Sale-phase-aware temporal features
        """
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # === TIME OF DAY FEATURES ===
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Business hours indicator
        df['is_business_hours'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17)
        ).astype(int)
        
        # Night time indicator
        df['is_night'] = (
            (df['hour'] >= 0) & (df['hour'] <= 6)
        ).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # === CYCLICAL ENCODING (NOVEL) ===
        # Encode hour as sine/cosine for ML models
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame, windows: List[int] = [5, 10, 30]) -> pd.DataFrame:
        """
        Add multi-scale rolling window features
        
        Patent Element: Multi-resolution temporal analysis
        """
        
        for window in windows:
            # Rolling mean
            df[f'request_count_ma{window}'] = df['request_count'].rolling(
                window, min_periods=1
            ).mean()
            
            # Rolling std
            df[f'request_count_std{window}'] = df['request_count'].rolling(
                window, min_periods=1
            ).std()
            
            # Rolling max (peak detection)
            df[f'request_count_max{window}'] = df['request_count'].rolling(
                window, min_periods=1
            ).max()
            
            # Deviation from rolling mean (anomaly indicator)
            df[f'deviation_ma{window}'] = (
                df['request_count'] - df[f'request_count_ma{window}']
            ) / (df[f'request_count_std{window}'] + 1)
        
        return df
    
    def add_rate_of_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rate of change metrics
        
        Patent Element: Velocity and acceleration of traffic
        """
        
        # First derivative (velocity)
        df['request_velocity'] = df['request_count'].diff()
        
        # Second derivative (acceleration)
        df['request_acceleration'] = df['request_velocity'].diff()
        
        # Percentage change
        df['request_pct_change'] = df['request_count'].pct_change()
        
        # Fill initial NaN values
        df['request_velocity'] = df['request_velocity'].fillna(0)
        df['request_acceleration'] = df['request_acceleration'].fillna(0)
        df['request_pct_change'] = df['request_pct_change'].fillna(0)
        
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare final feature matrix for ML models
        """
        
        # Define feature columns
        self.feature_names = [
            'request_count', 'unique_users', 'unique_ips',
            'path_entropy', 
            'request_variance', 'spike_ratio',
            'error_rate', 'avg_response_time', 'response_time_std', 'p95_response_time',
            'checkout_ratio', 'payment_ratio',
            'ip_concentration', 'user_concentration',
            'hour', 'is_business_hours', 'is_night', 'is_weekend',
            'hour_sin', 'hour_cos',
            'request_count_ma5', 'request_count_std5', 'deviation_ma5',
            'request_count_ma10', 'request_count_std10', 'deviation_ma10',
            'request_velocity', 'request_acceleration', 'request_pct_change'
        ]
        
        # Add optional features if present
        if 'bot_ratio' in df.columns:
            self.feature_names.append('bot_ratio')
        if 'user_agent_diversity' in df.columns:
            self.feature_names.append('user_agent_diversity')
        
        # Extract features
        X = df[[col for col in self.feature_names if col in df.columns]].fillna(0).values
        y = df['is_attack'].values
        
        # Normalize features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def process_pipeline(self, raw_data_path: str, save_path: str = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Complete feature engineering pipeline
        """
        
        print(" Starting feature engineering pipeline...")
        
        # Step 1: Load raw data
        print("  Loading raw data...")
        df = pd.read_csv(raw_data_path)
        print(f" Loaded {len(df):,} raw requests")
        
        # Step 2: Create time windows
        print("    Creating time windows...")
        df_windowed = self.create_time_windows(df)
        print(f" Created {len(df_windowed)} time windows")
        
        # Step 3: Add temporal features
        print("  Adding temporal features...")
        df_features = self.add_temporal_features(df_windowed)
        
        # Step 4: Add rolling statistics
        print("  Computing rolling statistics...")
        df_features = self.add_rolling_statistics(df_features)
        
        # Step 5: Add rate of change features
        print("  Computing rate of change features...")
        df_features = self.add_rate_of_change_features(df_features)
        
        # Step 6: Prepare ML features
        print("  Preparing ML feature matrix...")
        X, y = self.prepare_ml_features(df_features, fit_scaler=True)
        
        print(f"\n Feature Engineering Complete!")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Attack windows: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")
        print(f"  Total features: {len(self.feature_names)}")
        
        # Save processed data
        if save_path:
            df_features.to_csv(save_path.replace('.npy', '.csv'), index=False)
            np.save(save_path.replace('.csv', '_X.npy'), X)
            np.save(save_path.replace('.csv', '_y.npy'), y)
            
            # Save scaler and analyzer
            with open(save_path.replace('.csv', '_scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            if self.endpoint_analyzer:
                print("Skipping endpoint_analyzer (will recreate on load)")
            
            print(f"  Saved to {save_path}")
        
        return X, y, df_features
    
    def extract_realtime_features(self, request_data: Dict) -> Dict:
        """
        ENHANCED: Extract features from a single incoming request
        Now includes behavioral analysis
        """
        
        # Basic features
        basic_features = {
            'requests_per_minute': request_data.get('requests_per_minute', 0),
            'unique_users': request_data.get('unique_users', 0),
            'unique_ips': request_data.get('unique_ips', 0),
            'error_rate': request_data.get('error_rate', 0),
            'avg_response_time': request_data.get('avg_response_time', 0),
            'checkout_ratio': request_data.get('checkout_ratio', 0),
            'ip_concentration': request_data.get('ip_concentration', 0),
            'hour': request_data.get('hour', 0),
            'is_business_hours': request_data.get('is_business_hours', 0),
        }
        
        # NEW: Behavioral analysis
        if self.endpoint_analyzer:
            user_id = request_data.get('user_id', 'anonymous')
            endpoint = request_data.get('endpoint', '/api/default')
            
            behavior_analysis = self.endpoint_analyzer.analyze_request_sequence(
                user_id, endpoint
            )
            
            # Add behavioral features
            basic_features.update({
                'behavioral_suspicion_score': behavior_analysis.get('suspicion_score', 0.0),
                'avg_inter_request_time': behavior_analysis['behavioral_features'].get('avg_inter_request_time', 0),
                'endpoint_diversity': behavior_analysis['behavioral_features'].get('endpoint_diversity', 0),
                'has_rapid_sequence': behavior_analysis['behavioral_features'].get('has_rapid_sequence', 0),
            })
        
        return basic_features


if __name__ == "__main__":
    # Test the enhanced feature extractor
    
    # Load config
    config = {
        'cross_endpoint': {
            'enable_behavioral_analysis': True,
            'history_window_seconds': 60,
            'max_user_history': 100,
            'rapid_checkout_threshold_ms': 100,
            'parallel_request_threshold': 10,
            'credential_stuffing_threshold': 3
        }
    }
    
    engineer = TrafficFeatureEngineer(config, window_seconds=60)
    
    # Test behavioral analysis
    print(" Testing Cross-Endpoint Behavioral Analysis...\n")
    
    # Simulate bot behavior
    bot_user = "bot_123"
    
    # Rapid checkout sequence
    from datetime import datetime, timedelta
    base_time = datetime.now()
    
    for i, endpoint in enumerate(['/api/products', '/api/cart', '/api/checkout']):
        timestamp = base_time + timedelta(milliseconds=50 * i)  # 50ms between requests
        result = engineer.endpoint_analyzer.analyze_request_sequence(bot_user, endpoint, timestamp)
        
        print(f"Request {i+1} to {endpoint}:")
        print(f"  Is Suspicious: {result['is_suspicious']}")
        print(f"  Suspicion Score: {result['suspicion_score']:.2f}")
        print(f"  Matched Patterns: {result['matched_patterns']}")
        print()
    
    # Normal user behavior
    normal_user = "user_456"
    base_time = datetime.now()
    
    for i, endpoint in enumerate(['/api/products', '/api/cart', '/api/checkout']):
        timestamp = base_time + timedelta(seconds=5 * i)  # 5 seconds between requests
        result = engineer.endpoint_analyzer.analyze_request_sequence(normal_user, endpoint, timestamp)
        
        print(f"Normal User Request {i+1} to {endpoint}:")
        print(f"  Is Suspicious: {result['is_suspicious']}")
        print(f"  Suspicion Score: {result['suspicion_score']:.2f}")
        print()
    
    # Process full pipeline if data exists
    try:
        X, y, df_features = engineer.process_pipeline(
            raw_data_path='data/raw/traffic_data.csv',
            save_path='data/processed/features.csv'
        )
        
        print("\n Feature Statistics:")
        print(df_features[engineer.feature_names].describe())
    except FileNotFoundError:
        print("\n Raw data not found. Generate traffic data first using data/generator.py")
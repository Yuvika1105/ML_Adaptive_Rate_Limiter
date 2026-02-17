"""
Synthetic Traffic Data Generator for Flash Sale Scenarios
Generates realistic API traffic patterns with:
- Normal user behavior (daily patterns)
- Flash sale traffic spikes
- Bot/malicious traffic
- Various API endpoints
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
import os


class FlashSaleTrafficGenerator:
    """
    Generates synthetic API traffic data for training ML models
    Simulates: normal users, flash sale events, bot attacks
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.normal_rate = config['normal_traffic_rate']
        self.flash_sale_rate = config['flash_sale_traffic_rate']
        self.bot_rate = config['bot_traffic_rate']
        
        # API endpoints with different characteristics
        self.endpoints = [
            {'path': '/api/products', 'weight': 0.4, 'avg_response': 100},
            {'path': '/api/cart', 'weight': 0.3, 'avg_response': 150},
            {'path': '/api/checkout', 'weight': 0.2, 'avg_response': 300},
            {'path': '/api/payment', 'weight': 0.1, 'avg_response': 500},
        ]
        
    def _generate_normal_traffic(self, hours: int) -> pd.DataFrame:
        """Generate legitimate user traffic with realistic daily patterns"""
        
        start_time = datetime.now()
        timestamps = pd.date_range(
            start=start_time,
            periods=hours * 3600,
            freq='1s'
        )
        
        data = []
        
        for ts in timestamps:
            hour = ts.hour
            
            # Daily pattern: business hours are busier
            if 9 <= hour <= 17:
                rate_multiplier = 1.8  # Peak hours
            elif 18 <= hour <= 22:
                rate_multiplier = 1.2  # Evening
            elif 0 <= hour <= 6:
                rate_multiplier = 0.3  # Night
            else:
                rate_multiplier = 1.0  # Normal
            
            # Weekend effect
            if ts.dayofweek >= 5:  # Saturday, Sunday
                rate_multiplier *= 0.7
            
            # Add some randomness
            num_requests = int(np.random.poisson(self.normal_rate * rate_multiplier))
            
            for _ in range(num_requests):
                # Select endpoint based on weights
                endpoint = np.random.choice(
                    [e['path'] for e in self.endpoints],
                    p=[e['weight'] for e in self.endpoints]
                )
                
                endpoint_data = next(e for e in self.endpoints if e['path'] == endpoint)
                
                # Generate user behavior
                user_id = f"U{np.random.randint(1, 1000)}"
                
                data.append({
                    'timestamp': ts,
                    'user_id': user_id,
                    'ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                    'endpoint': endpoint,
                    'method': 'GET' if endpoint == '/api/products' else 'POST',
                    'status_code': np.random.choice([200, 201, 400, 404], p=[0.85, 0.10, 0.03, 0.02]),
                    'response_time': max(10, np.random.gamma(2, endpoint_data['avg_response'] / 2)),
                    'user_agent': np.random.choice([
                        'Mozilla/5.0 (Windows)', 'Mozilla/5.0 (Mac)', 
                        'Mozilla/5.0 (Android)', 'Mozilla/5.0 (iPhone)'
                    ]),
                    'traffic_type': 'normal',
                    'is_bot': 0,
                    'is_attack': 0,
                })
        
        return pd.DataFrame(data)
    
    def _inject_flash_sale_traffic(
        self, 
        df: pd.DataFrame, 
        sale_start_hour: int = 12,
        sale_duration_min: int = 30
    ) -> pd.DataFrame:
        """Inject flash sale traffic surge with both legitimate users and bots"""
        
        flash_data = []
        sale_start = df['timestamp'].min() + timedelta(hours=sale_start_hour)
        sale_end = sale_start + timedelta(minutes=sale_duration_min)
        
        # Pre-sale rush (5 minutes before)
        pre_sale_start = sale_start - timedelta(minutes=5)
        
        # Generate flash sale traffic
        sale_timestamps = pd.date_range(
            start=pre_sale_start,
            end=sale_end,
            freq='100ms'  # Very high frequency
        )
        
        for ts in sale_timestamps:
            # Determine phase
            if ts < sale_start:
                phase = 'pre_sale'
                intensity = 0.5
            elif ts < sale_start + timedelta(minutes=5):
                phase = 'sale_start'
                intensity = 2.0  # Massive spike at start
            elif ts < sale_start + timedelta(minutes=15):
                phase = 'sale_peak'
                intensity = 1.5
            else:
                phase = 'sale_end'
                intensity = 0.8  # Tapering off
            
            # Legitimate buyers
            num_legit = int(np.random.poisson(self.flash_sale_rate * intensity / 10))
            
            for _ in range(num_legit):
                user_id = f"U{np.random.randint(1, 5000)}"
                
                flash_data.append({
                    'timestamp': ts,
                    'user_id': user_id,
                    'ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                    'endpoint': np.random.choice(['/api/products', '/api/cart', '/api/checkout'], 
                                                p=[0.3, 0.4, 0.3]),
                    'method': 'POST',
                    'status_code': np.random.choice([200, 201, 503, 429], p=[0.6, 0.2, 0.1, 0.1]),
                    'response_time': np.random.gamma(5, 100),  # Slower due to load
                    'user_agent': np.random.choice([
                        'Mozilla/5.0 (Windows)', 'Mozilla/5.0 (Mac)', 
                        'Mozilla/5.0 (Android)', 'Mozilla/5.0 (iPhone)'
                    ]),
                    'traffic_type': 'flash_sale',
                    'is_bot': 0,
                    'is_attack': 0,
                })
        
        flash_df = pd.DataFrame(flash_data)
        combined = pd.concat([df, flash_df]).sort_values('timestamp').reset_index(drop=True)
        
        return combined
    
    def _inject_bot_traffic(
        self, 
        df: pd.DataFrame,
        attack_start_hour: int = 12,
        attack_duration_min: int = 20
    ) -> pd.DataFrame:
        """Inject bot/malicious traffic patterns"""
        
        bot_data = []
        attack_start = df['timestamp'].min() + timedelta(hours=attack_start_hour)
        attack_end = attack_start + timedelta(minutes=attack_duration_min)
        
        # Bot IP pools
        bot_ips = [f"10.0.{i}.{j}" for i in range(1, 10) for j in range(1, 20)]
        
        attack_timestamps = pd.date_range(
            start=attack_start,
            end=attack_end,
            freq='50ms'  # Very aggressive
        )
        
        for ts in attack_timestamps:
            num_bots = int(np.random.poisson(self.bot_rate / 20))
            
            for _ in range(num_bots):
                bot_data.append({
                    'timestamp': ts,
                    'user_id': f"BOT{np.random.randint(1, 100)}",
                    'ip': np.random.choice(bot_ips),
                    'endpoint': '/api/checkout',  # Target checkout
                    'method': 'POST',
                    'status_code': np.random.choice([429, 503, 403], p=[0.5, 0.3, 0.2]),
                    'response_time': np.random.gamma(2, 50),  # Fast, scripted
                    'user_agent': np.random.choice([
                        'python-requests/2.28', 'curl/7.68', 'Bot/1.0', 'Scrapy/2.7'
                    ]),
                    'traffic_type': 'bot_attack',
                    'is_bot': 1,
                    'is_attack': 1,
                })
        
        bot_df = pd.DataFrame(bot_data)
        combined = pd.concat([df, bot_df]).sort_values('timestamp').reset_index(drop=True)
        
        return combined
    
    def generate_dataset(
        self, 
        hours: int = 24,
        num_flash_sales: int = 2,
        num_bot_attacks: int = 3,
        save_path: str = None
    ) -> pd.DataFrame:
        """Generate complete dataset with all traffic patterns"""
        
        print(f" Generating {hours} hours of traffic data...")
        
        # Step 1: Generate normal traffic baseline
        df = self._generate_normal_traffic(hours)
        print(f" Generated {len(df):,} normal requests")
        
        # Step 2: Inject flash sale events
        sale_times = [8, 14, 20][:num_flash_sales]
        for sale_hour in sale_times:
            df = self._inject_flash_sale_traffic(
                df, 
                sale_start_hour=sale_hour,
                sale_duration_min=self.config['flash_sale_duration_min']
            )
        print(f"✓ Injected {num_flash_sales} flash sale events")
        
        # Step 3: Inject bot attacks
        attack_times = [8, 12, 18][:num_bot_attacks]
        for attack_hour in attack_times:
            df = self._inject_bot_traffic(
                df,
                attack_start_hour=attack_hour,
                attack_duration_min=20
            )
        print(f" Injected {num_bot_attacks} bot attacks")
        
        # Add derived features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        print(f"\n Final Dataset Statistics:")
        print(f"  Total requests: {len(df):,}")
        print(f"  Normal traffic: {(df['traffic_type'] == 'normal').sum():,}")
        print(f"  Flash sale traffic: {(df['traffic_type'] == 'flash_sale').sum():,}")
        print(f"  Bot attacks: {df['is_attack'].sum():,} ({df['is_attack'].mean()*100:.2f}%)")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f" Saved to {save_path}")
        
        return df


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate data
    generator = FlashSaleTrafficGenerator(config['data_generation'])
    df = generator.generate_dataset(
        hours=24,
        num_flash_sales=2,
        num_bot_attacks=3,
        save_path='data/raw/traffic_data.csv'
    )
    
    # Display sample
    print("\n Sample Data:")
    print(df.head(10))
    
    print("\n Traffic Type Distribution:")
    print(df['traffic_type'].value_counts())
    
    print("\n Endpoint Distribution:")
    print(df['endpoint'].value_counts())


    
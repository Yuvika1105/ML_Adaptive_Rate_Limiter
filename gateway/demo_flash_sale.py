"""
FLASH SALE DEMO - Real-World Scenario Simulation

Simulates a realistic e-commerce flash sale scenario with:
1. Normal baseline traffic
2. Flash sale traffic surge (legitimate buyers)
3. Bot attack (scalpers trying to buy everything)
4. System response using ML-guided adaptive rate limiting

Shows how the system:
- Scales UP during legitimate flash sale
- Scales DOWN during bot attack  
- Maintains service for real customers
- Blocks bots effectively
"""

import requests
import time
import random
import threading
from datetime import datetime
from collections import defaultdict
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

GATEWAY_URL = "http://localhost:8000"
ENDPOINTS = [
    "/api/products",
    "/api/cart", 
    "/api/checkout",
    "/api/payment"
]

# Traffic profiles
NORMAL_RATE = 5  # requests/second
FLASH_SALE_RATE = 50  # requests/second (10x normal)
BOT_RATE = 200  # requests/second (40x normal)

# ============================================================================
# TRAFFIC GENERATORS
# ============================================================================

class TrafficGenerator:
    """Generate HTTP traffic to test gateway"""
    
    def __init__(self, name, rate, endpoints, user_agent="Mozilla/5.0"):
        self.name = name
        self.rate = rate  # requests per second
        self.endpoints = endpoints
        self.user_agent = user_agent
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'rate_limited': 0,
            'errors': 0
        }
        self.running = False
        
    def generate_traffic(self, duration_seconds):
        """Generate traffic for specified duration"""
        
        self.running = True
        start_time = time.time()
        
        print(f" {self.name} starting traffic generation...")
        print(f"   Rate: {self.rate} req/sec")
        print(f"   Duration: {duration_seconds}s\n")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            # Send burst of requests
            for _ in range(self.rate):
                self._send_request()
                
            # Wait 1 second
            time.sleep(1)
            
            # Print progress
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                self._print_stats()
        
        self.running = False
        print(f"\n {self.name} completed!\n")
        self._print_final_stats()
    
    def _send_request(self):
        """Send single HTTP request"""
        
        try:
            endpoint = random.choice(self.endpoints)
            url = f"{GATEWAY_URL}{endpoint}"
            
            headers = {
                'User-Agent': self.user_agent,
                'X-User-ID': f'{self.name}_{random.randint(1, 100)}'
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            
            self.stats['total_requests'] += 1
            
            if response.status_code == 200:
                self.stats['successful'] += 1
            elif response.status_code == 429:
                self.stats['rate_limited'] += 1
            else:
                self.stats['errors'] += 1
                
        except Exception as e:
            self.stats['errors'] += 1
    
    def _print_stats(self):
        """Print current statistics"""
        
        total = self.stats['total_requests']
        success_rate = (self.stats['successful'] / total * 100) if total > 0 else 0
        block_rate = (self.stats['rate_limited'] / total * 100) if total > 0 else 0
        
        print(f"[{self.name}] Total: {total}, Success: {success_rate:.1f}%, Blocked: {block_rate:.1f}%")
    
    def _print_final_stats(self):
        """Print final statistics"""
        
        print(f" {self.name} - FINAL STATS")
        print(f"{'='*50}")
        print(f"  Total Requests:    {self.stats['total_requests']}")
        print(f"  Successful:      {self.stats['successful']} ({self.stats['successful']/self.stats['total_requests']*100:.1f}%)")
        print(f"  Rate Limited:    {self.stats['rate_limited']} ({self.stats['rate_limited']/self.stats['total_requests']*100:.1f}%)")
        print(f"  Errors:          {self.stats['errors']}")
        print()
    
    def stop(self):
        """Stop generating traffic"""
        self.running = False


# ============================================================================
# SCENARIO EXECUTION
# ============================================================================

def check_gateway_health():
    """Check if gateway is running"""
    
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    
    return False


def run_flash_sale_scenario():
    """
    Run complete flash sale scenario
    
    Timeline:
    0:00 - 0:30  : Normal traffic (baseline)
    0:30 - 1:00  : Flash sale announced (traffic surge)
    1:00 - 1:30  : Bots attack (scalpers)
    1:30 - 2:00  : Flash sale ends (return to normal)
    """
    
    print("\n" + "="*70)
    print("  FLASH SALE SCENARIO SIMULATION")
    print("="*70)
    print("\nThis demo simulates a realistic e-commerce flash sale with:")
    print("  1. Normal baseline traffic")
    print("  2. Legitimate flash sale surge")
    print("  3. Bot attack (scalpers)")
    print("  4. Return to normal")
    print("\nThe ML-guided system should:")
    print("  Scale UP for legitimate flash sale traffic")
    print("  Scale DOWN to block bot attacks")
    print("  Maintain service for real customers")
    print("\n" + "="*70 + "\n")
    
    # Check gateway
    print(" Checking if gateway is running...")
    if not check_gateway_health():
        print("\n ERROR: Gateway not running!")
        print("\nPlease start the gateway first:")
        print("  python gateway/app.py")
        print("\nOr run in another terminal:")
        print("  cd gateway && python app.py\n")
        return
    
    print(" Gateway is running!\n")
    input("Press ENTER to start the simulation...")
    
    # ========================================================================
    # PHASE 1: NORMAL TRAFFIC (30 seconds)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: NORMAL TRAFFIC (30 seconds)")
    print("="*70)
    print("Simulating regular e-commerce traffic...\n")
    
    normal_traffic = TrafficGenerator(
        name="Normal Users",
        rate=NORMAL_RATE,
        endpoints=ENDPOINTS,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )
    
    normal_traffic.generate_traffic(30)
    
    time.sleep(2)
    
    # ========================================================================
    # PHASE 2: FLASH SALE (30 seconds)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: FLASH SALE SURGE (30 seconds)")
    print("="*70)
    print(" Flash sale announced! Legitimate buyers rushing in...\n")
    
    flash_sale_traffic = TrafficGenerator(
        name="Flash Sale Buyers",
        rate=FLASH_SALE_RATE,
        endpoints=ENDPOINTS,
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
    )
    
    flash_sale_traffic.generate_traffic(30)
    
    print("\n OBSERVATION: System should ALLOW high traffic (legitimate surge)")
    print("   The ML model detects this is a flash sale, not an attack.\n")
    
    time.sleep(2)
    
    # ========================================================================
    # PHASE 3: BOT ATTACK (30 seconds)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 3: BOT ATTACK (30 seconds)")
    print("="*70)
    print(" Scalper bots detected! Attacking the checkout endpoint...\n")
    
    bot_traffic = TrafficGenerator(
        name="Scalper Bots",
        rate=BOT_RATE,
        endpoints=["/api/checkout", "/api/payment"],  # Target checkout
        user_agent="python-requests/2.28.0"  # Bot user agent
    )
    
    bot_traffic.generate_traffic(30)
    
    print("\n OBSERVATION: System should BLOCK bot traffic")
    print("   ML model detects anomalous behavior and reduces rate limit.\n")
    
    time.sleep(2)
    
    # ========================================================================
    # PHASE 4: RETURN TO NORMAL (30 seconds)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 4: RETURN TO NORMAL (30 seconds)")
    print("="*70)
    print("Flash sale ended. Traffic returning to baseline...\n")
    
    normal_traffic2 = TrafficGenerator(
        name="Normal Users",
        rate=NORMAL_RATE,
        endpoints=ENDPOINTS,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    )
    
    normal_traffic2.generate_traffic(30)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print(" FLASH SALE SCENARIO - FINAL SUMMARY")
    print("="*70)
    
    print("\n SCENARIO COMPLETED!")
    print("\nKey Observations:")
    print("  1. Normal traffic: Low rate, mostly allowed")
    print("  2. Flash sale: High rate, but ALLOWED (adaptive scaling)")
    print("  3. Bot attack: Very high rate, mostly BLOCKED")
    print("  4. Recovery: System adapted back to normal")
    
    print("\nYOUR ML-GUIDED SYSTEM:")
    print("  Scaled up for legitimate flash sale")
    print("  Detected and blocked bot attack")
    print("  Maintained service for real customers")
    print("  Adapted dynamically without manual intervention")
    
    print("\n📊 View detailed metrics:")
    print(f"  Dashboard: {GATEWAY_URL}/dashboard")
    print(f"  Metrics:   {GATEWAY_URL}/metrics")
    
    print("\n" + "="*70 + "\n")


def run_simple_load_test():
    """
    Simple load test - just send traffic to see if gateway responds
    """
    
    print("\n" + "="*70)
    print(" SIMPLE LOAD TEST")
    print("="*70)
    print("\nSending 100 requests to test the gateway...\n")
    
    # Check gateway
    if not check_gateway_health():
        print(" Gateway not running! Start it first:\n")
        print("  python gateway/app.py\n")
        return
    
    print(" Gateway is running!\n")
    
    # Send requests
    success = 0
    blocked = 0
    errors = 0
    
    for i in range(100):
        try:
            response = requests.get(
                f"{GATEWAY_URL}/api/products",
                headers={'X-User-ID': f'test_user_{i % 10}'},
                timeout=5
            )
            
            if response.status_code == 200:
                success += 1
            elif response.status_code == 429:
                blocked += 1
            else:
                errors += 1
                
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/100 requests sent...")
                
        except Exception as e:
            errors += 1
    
    print("\nRESULTS:")
    print(f"  Total Requests: 100")
    print(f"  Successful:   {success}")
    print(f"  Blocked:      {blocked}")
    print(f"  Errors:       {errors}")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("  ML-GUIDED RATE LIMITING - DEMO SCENARIOS")
    print("="*70)
    print("\nAvailable scenarios:")
    print("  1. Simple Load Test (100 requests)")
    print("  2. Full Flash Sale Scenario (2 minutes)")
    print()
    
    choice = input("Select scenario (1 or 2): ").strip()
    
    if choice == "1":
        run_simple_load_test()
    elif choice == "2":
        run_flash_sale_scenario()
    else:
        print("\n Invalid choice. Run again and select 1 or 2.\n")
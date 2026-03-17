import requests
import numpy as np
import sys, os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rns_bridge import RNSContext
from batch_encoder import BatchEncoder

def print_dashboard(data, display_count=15):
    print("\n" + "="*55)
    print(" 🛡️  ZERO-TRUST EDGE: LIVE THREAT FEATURES  🛡️ ")
    print("="*55 + "\n")
    
    max_val = max(data[:display_count]) if max(data[:display_count]) > 0 else 1
    
    for i, val in enumerate(data[:display_count]):
        bar_len = int((val / max_val) * 35)
        # ANSI color codes: Cyan for the bar
        bar = "\033[96m" + "█" * bar_len + "\033[0m"
        print(f" Sensor_CH_{i:03d} | Output: {val:05d} | {bar}")
        time.sleep(0.08)  # Cascading effect for the presentation
        
    print(f"\n[✓] {len(data)}-Slot SIMD Array Successfully Decrypted.\n")

print("\n--- ZERO-TRUST EDGE: CLIENT ---")
rns = RNSContext()
encoder = BatchEncoder(N=1024, T=65537)

sensor_data = np.arange(1024, dtype=np.uint32)
pt_poly = encoder.encode(sensor_data)
ct0, ct1 = rns.encrypt(pt_poly)

print("Transmitting encrypted SIMD array to Inference API...")
t0 = time.perf_counter()
response = requests.post("http://127.0.0.1:8000/predict/features", json={
    "ct0": ct0.get().tolist(),
    "ct1": ct1.get().tolist()
})
network_time = (time.perf_counter() - t0) * 1000

if response.status_code == 200:
    res_data = response.json()
    res_c0 = np.array(res_data["ct0"], dtype=np.uint32)
    res_c1 = np.array(res_data["ct1"], dtype=np.uint32)
    
    dec_poly = rns.decrypt(res_c0, res_c1)
    final_features = encoder.decode(dec_poly)
    
    print(f"\n[Network Round-Trip + GPU Inference]: {network_time:.2f}ms")
    print_dashboard(final_features)
else:
    print(f"Error: {response.text}")

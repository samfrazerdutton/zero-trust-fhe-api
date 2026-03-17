import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rns_bridge import RNSContext
from batch_encoder import BatchEncoder
from nn.layers import EncryptedFeatureExtractor

print("\n--- ZERO-TRUST EDGE: THREAT DETECTION ---")
rns = RNSContext()
encoder = BatchEncoder(N=1024, T=65537)
layer = EncryptedFeatureExtractor(rns, encoder)

# 1. Create Drone Sensor Data (e.g., altitude, velocity, payload mass)
sensor_data = np.arange(1024, dtype=np.uint32)
print(f"\n1. Drone Telemetry (first 5 sensors): {sensor_data[:5]}")
print(f"   ML Weights (first 5): {layer.weights[:5]}")

# 2. Encrypt the SIMD Array
pt_poly = encoder.encode(sensor_data)
ct = rns.encrypt(pt_poly)

# 3. Secure AI Inference (GPU)
res_ct = layer.forward(ct)

# 4. Decrypt & Decode
dec_poly = rns.decrypt(res_ct[0], res_ct[1])
final_data = encoder.decode(dec_poly)

print(f"\n2. Extracted Threat Features (first 5 slots): {final_data[:5]}")
print("   (Expected: Telemetry * Weights)")

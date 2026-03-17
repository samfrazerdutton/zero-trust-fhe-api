import requests
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from fhe_bridge import cuFHE
from batch_encoder import BatchEncoder

print("\n--- SIMD ZERO-TRUST CLIENT ---")
fhe = cuFHE()
encoder = BatchEncoder(N=1024, T=12289)

# Generate an array of 1024 different sensor readings (0 to 1023)
sensor_data = np.arange(1024, dtype=np.uint32)
print(f"1. Original Sensor Data (first 5): {sensor_data[:5]}")

# Pack into a single polynomial and encrypt
msg_poly = encoder.encode(sensor_data)
ct0, ct1 = fhe.encrypt(msg_poly)

print("2. Transmitting packed ciphertext to API...")
response = requests.post("http://127.0.0.1:8000/predict/simd", json={
    "ct0": ct0.get().tolist(),
    "ct1": ct1.get().tolist()
})

if response.status_code == 200:
    res_data = response.json()
    
    # Decrypt and unpack
    dec_poly = fhe.decrypt(res_data["ct0"], res_data["ct1"])
    unpacked_data = encoder.decode(dec_poly)
    
    print(f"3. Decrypted & Unpacked Data (first 5): {unpacked_data[:5]}")
    print("   (Expected: Original * 5)")
else:
    print(f"Error: {response.text}")

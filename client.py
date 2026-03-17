import requests
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from fhe_bridge import cuFHE, N

print("\n--- ZERO-TRUST EDGE CLIENT ---")
fhe = cuFHE()

# Generate a simple payload: [1, 1, 1, 1...]
# 1 * 1 should equal 1
msg = np.zeros(N, dtype=np.uint32)
msg[0:8] = 1

print("Encrypting telemetry under sparse client keys...")
ct0, ct1 = fhe.encrypt(msg)

pk0, pk1 = fhe.export_public_key()
rlk0 = fhe.d_rlk0.get().tolist()
rlk1 = fhe.d_rlk1.get().tolist()

payload = {
    "ct0": ct0.get().tolist(),
    "ct1": ct1.get().tolist(),
    "pk0": pk0.tolist(),
    "pk1": pk1.tolist(),
    "rlk0": rlk0,
    "rlk1": rlk1
}

print("Transmitting encrypted payload + RLK to API...")
url = "http://127.0.0.1:8000/predict/encrypted"
response = requests.post(url, json=payload)

if response.status_code == 200:
    res_data = response.json()
    res_ct0 = np.array(res_data["ct0"], dtype=np.uint32)
    res_ct1 = np.array(res_data["ct1"], dtype=np.uint32)
    
    decrypted = fhe.decrypt(res_ct0, res_ct1)
    print(f"\nDecrypted first 8 outputs: {decrypted[0:8].tolist()}")
else:
    print(f"Error: {response.text}")

import os
import sys
import requests
import time
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from src.fhe_bridge import cuFHE, N, T

API_URL = "http://127.0.0.1:8000/predict/encrypted"

print("\n--- ZERO-TRUST EDGE CLIENT ---")
print("Generating client keypair...")
client_fhe = cuFHE()
pk0, pk1 = client_fhe.export_public_key()

msg = np.random.randint(0, T, N, dtype=np.uint32)
print("Encrypting client data under client public key...")
ct0, ct1 = client_fhe.encrypt(msg, pk=(pk0, pk1))

payload = {
    "client_id": "Edge-Node-01",
    "task_id": "Secure-Inference-Job",
    "public_key_0": pk0.tolist(),
    "public_key_1": pk1.tolist(),
    "ciphertext_0": cp.asnumpy(ct0).astype(np.uint32).tolist(),
    "ciphertext_1": cp.asnumpy(ct1).astype(np.uint32).tolist(),
}

print("Transmitting client public key + ciphertext to API...\n")
t0 = time.perf_counter()
response = requests.post(API_URL, json=payload)
ms = (time.perf_counter() - t0) * 1000
response.raise_for_status()
body = response.json()

print(f"Encrypted inference result received in {ms:.2f}ms")
out_ct0 = cp.asarray(np.array(body["ciphertext_0"], dtype=np.uint32))
out_ct1 = cp.asarray(np.array(body["ciphertext_1"], dtype=np.uint32))

out_msg = client_fhe.decrypt(out_ct0, out_ct1)
print("Decrypted first 8 outputs:", out_msg[:8].tolist())

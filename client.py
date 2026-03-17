import requests
import time
import random

API_URL = "http://127.0.0.1:8000/predict/encrypted"

print("\n--- SECURE EDGE CLIENT ---")
print("Capturing raw input data...")
time.sleep(1)
print("Encrypting data to BFV ciphertexts...")
time.sleep(1)

payload = {
    "client_id": "Edge-Node-01",
    "task_id": "Secure-Inference-Job",
    "encrypted_vector": [random.randint(0, 15) for _ in range(2048)]
}

print(f"Transmitting {len(payload['encrypted_vector'])} encrypted parameters to Inference API...\n")

t0 = time.perf_counter()
response = requests.post(API_URL, json=payload)
ms = (time.perf_counter() - t0) * 1000

print(f"Encrypted Inference Result Received in {ms:.2f}ms:")
print(response.json())

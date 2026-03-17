import sys
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.fhe_bridge import cuFHE

N = 1024

app = FastAPI(title="Zero-Trust FHE Inference API")

print("\n" + "="*50)
print("  ZERO-TRUST FHE INFERENCE ENGINE")
print("  GPU-Accelerated Encrypted Processing")
print("="*50 + "\n")

print("[API] Booting CUDA Context and Allocating VRAM...")
fhe = cuFHE()
print("[API] GPU Engine Ready. Listening for encrypted payloads...\n")


class EncryptedPayload(BaseModel):
    client_id: str
    task_id: str
    public_key_0: List[int] = Field(min_length=N, max_length=N)
    public_key_1: List[int] = Field(min_length=N, max_length=N)
    ciphertext_0: List[int] = Field(min_length=N, max_length=N)
    ciphertext_1: List[int] = Field(min_length=N, max_length=N)


@app.post("/predict/encrypted")
async def process_payload(payload: EncryptedPayload):
    print(f"\n[API] INCOMING CONNECTION: {payload.client_id} (Task: {payload.task_id})")

    t0 = time.perf_counter()

    pk = (
        np.array(payload.public_key_0, dtype=np.uint32),
        np.array(payload.public_key_1, dtype=np.uint32),
    )
    ct = (
        np.array(payload.ciphertext_0, dtype=np.uint32),
        np.array(payload.ciphertext_1, dtype=np.uint32),
    )

    if pk[0].shape[0] != N or ct[0].shape[0] != N:
        raise HTTPException(status_code=400, detail="Invalid key or ciphertext length")

    print("[API] Executing encrypted compute graph on client ciphertext...")
    import cupy as cp
    gct = (cp.asarray(ct[0]), cp.asarray(ct[1]))
    for _ in range(3):
        gct = fhe.he_mul_ct(gct, gct)

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"[API] Inference complete. Total compute time: {total_ms:.2f}ms")

    return {
        "status": "success",
        "compute_time_ms": round(total_ms, 2),
        "result_status": "PROCESSED_CLIENT_CIPHERTEXT",
        "ciphertext_0": cp.asnumpy(gct[0]).astype(np.uint32).tolist(),
        "ciphertext_1": cp.asnumpy(gct[1]).astype(np.uint32).tolist(),
    }

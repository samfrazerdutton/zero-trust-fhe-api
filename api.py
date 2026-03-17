from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cupy as cp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from fhe_bridge import cuFHE

app = FastAPI()
fhe = cuFHE()

class EncryptedRequest(BaseModel):
    ct0: list
    ct1: list
    pk0: list
    pk1: list
    rlk0: list
    rlk1: list

@app.post("/predict/encrypted")
async def predict_encrypted(req: EncryptedRequest):
    c0 = cp.asarray(req.ct0, dtype=cp.uint32)
    c1 = cp.asarray(req.ct1, dtype=cp.uint32)
    fhe.d_pk0  = cp.asarray(req.pk0,  dtype=cp.uint32)
    fhe.d_pk1  = cp.asarray(req.pk1,  dtype=cp.uint32)
    fhe.d_rlk0 = cp.asarray(req.rlk0, dtype=cp.uint32)
    fhe.d_rlk1 = cp.asarray(req.rlk1, dtype=cp.uint32)

    # he_add with zero ciphertext = identity (passthrough)
    zero_ct = (cp.zeros(1024, dtype=cp.uint32), cp.zeros(1024, dtype=cp.uint32))
    res_c0, res_c1 = fhe.he_add((c0, c1), zero_ct)

    return {
        "ct0": res_c0.get().tolist(),
        "ct1": res_c1.get().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

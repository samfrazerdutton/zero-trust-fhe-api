from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cupy as cp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rns_bridge import RNSContext
from batch_encoder import BatchEncoder
from nn.layers import EncryptedFeatureExtractor

app = FastAPI()
rns = RNSContext()
encoder = BatchEncoder(N=1024, T=65537)
layer = EncryptedFeatureExtractor(rns, encoder)

class EncryptedRequest(BaseModel):
    ct0: list
    ct1: list

@app.post("/predict/features")
async def predict_features(req: EncryptedRequest):
    c0 = cp.asarray(req.ct0, dtype=cp.uint32)
    c1 = cp.asarray(req.ct1, dtype=cp.uint32)
    
    # Pass the 5-limb RNS ciphertext through the ML layer
    res_c0, res_c1 = layer.forward((c0, c1))

    return {
        "ct0": res_c0.get().tolist(),
        "ct1": res_c1.get().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

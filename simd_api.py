from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from fhe_bridge import cuFHE
from batch_encoder import BatchEncoder

app = FastAPI()
fhe = cuFHE()
encoder = BatchEncoder(N=1024, T=12289)

class EncryptedRequest(BaseModel):
    ct0: list
    ct1: list

@app.post("/predict/simd")
async def predict_simd(req: EncryptedRequest):
    c0 = np.array(req.ct0, dtype=np.uint32)
    c1 = np.array(req.ct1, dtype=np.uint32)
    
    # Encode 1024 weights (e.g., multiply all incoming sensor readings by 5)
    weights = np.ones(1024, dtype=np.uint32) * 5
    pt_poly = encoder.encode(weights)
    
    # Homomorphic SIMD Multiplication
    res_c0 = fhe._polymul(c0, pt_poly)
    res_c1 = fhe._polymul(c1, pt_poly)

    return {
        "ct0": res_c0.get().tolist(),
        "ct1": res_c1.get().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

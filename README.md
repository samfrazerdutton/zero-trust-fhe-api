# zero-trust-fhe-api

A GPU-accelerated Fully Homomorphic Encryption (FHE) inference API. An edge client encrypts telemetry locally, transmits the ciphertext to the server, and the server computes on it without ever seeing the plaintext. The client decrypts the result.

Built on a custom BFV implementation with bare-metal CUDA kernels for the NTT polynomial arithmetic.

---

## How it works

BFV (Brakerski/Fan-Vercauteren) is a scheme that allows arithmetic on ciphertexts directly. The core operation is polynomial multiplication in the negacyclic ring `Z_Q[x]/(x^N+1)`, accelerated here via a custom Number Theoretic Transform (NTT) on the GPU.

The pipeline:
```
Edge Client                          API Server
-----------                          ----------
Generate keypair
Encrypt telemetry (pk)
Send ct + public key  ──────────►   Load ciphertext onto GPU
                                     Compute he_mul_plain / he_add
                      ◄──────────   Return result ciphertext
Decrypt result (sk)
```

The server never holds the secret key. All computation happens on encrypted data.

---

## Crypto parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Scheme | BFV | |
| N | 1024 | Polynomial degree |
| Q | 12289 | NTT-friendly prime: 3·2¹² + 1 |
| T | 16 | Plaintext modulus |
| Δ | 768 | Scaling factor: ⌊Q/T⌋ |
| PSI | 1945 | Primitive 2N-th root of unity mod Q |

---

## Requirements

- NVIDIA GPU (any SM version from 6.0 upward)
- CUDA toolkit with `nvcc` on PATH
- Python 3.10+
```
pip install -r requirements.txt
```

`gpu_utils.py` auto-detects the GPU's SM version at runtime and compiles the PTX if a pre-built binary for that architecture isn't present. No manual recompilation needed when moving between machines.

---

## Running

Start the API server:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second terminal, run the edge client:
```bash
python3 client.py
```

Expected output:
```
--- ZERO-TRUST EDGE CLIENT ---
[GPU] NVIDIA GeForce RTX 2060 with Max-Q Design | SM_75 | 6.0GB VRAM
[cuFHE] Ready — N=1024, Q=12289, T=16, Δ=768
Encrypting telemetry under sparse client keys...
[cuFHE] Encrypt (pk) 2.070ms
Transmitting encrypted payload + RLK to API...
[cuFHE] Decrypt 3.037ms
Decrypted first 8 outputs: [1, 1, 1, 1, 1, 1, 1, 1]
```

---

## Repository structure
```
.
├── api.py                  # FastAPI server — receives ciphertext, computes, returns result
├── client.py               # Edge node — encrypts, transmits, decrypts
├── src/
│   ├── fhe_bridge.py       # cuFHE class: keygen, encrypt, decrypt, HE operations
│   └── gpu_utils.py        # Auto-detects SM version, compiles PTX if needed
├── kernels/
│   ├── fhe_kernel.cu       # CUDA source: NTT, BFV encrypt/decrypt, poly arithmetic
│   └── fhe_kernel_sm_*.ptx # Pre-compiled PTX binaries for various architectures
└── requirements.txt
```

---

## Performance

Measured on RTX 2060 Max-Q (SM_75, 6GB VRAM):

| Operation | Latency |
|-----------|---------|
| Encrypt (pk) | ~2ms |
| Decrypt | ~2ms |
| Full client round-trip | ~60ms (includes HTTP) |

---

## NTT implementation note

The NTT uses Algorithm 1 from Longa & Naehrig (2016) for the negacyclic convolution in `Z_Q[x]/(x^N+1)`. Each butterfly stage uses twiddle factors `PSI^bitrev(m+i, log2N)` directly, avoiding a separate pre/post-multiply pass. The roots table is laid out as `roots[m+i] = PSI^bitrev(m+i, log2N)` so the CUDA kernel can index it with a single `roots[m + threadIdx]` lookup per butterfly.

---

## Limitations

- Secret key lives on the client only — the server has no way to decrypt
- No noise budget tracking — deep computation chains will corrupt results without bootstrapping
- Single-ciphertext inference only — batching via SIMD slot packing not yet implemented
- Requires NVIDIA GPU — no CPU fallback

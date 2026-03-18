# Zero-Trust Edge FHE Engine

A hardware-accelerated, Fully Homomorphic Encryption (FHE) inference pipeline designed for edge computing. This engine allows a client to transmit encrypted sensor telemetry to an API, where a GPU performs machine learning feature extraction on the ciphertexts *without ever decrypting the payload*. 

This project was developed as demonstrating a lightweight, high-performance alternative to massive standard FHE libraries.

## 🚀 Key Capabilities
* **True SIMD Processing:** Uses Chinese Remainder Theorem (CRT) batching to pack 1,024 discrete sensor readings into a single polynomial.
* **Arbitrary-Precision Cryptography:** Handles massive $10^{24}$ Residue Number System (RNS) integers locally to bypass 64-bit overflow limits.
* **Native GPU Acceleration:** Custom CUDA kernels map the RNS limbs directly into VRAM, executing $O(N^2)$ negacyclic convolutions in under 5 milliseconds.
* **Zero-Noise Decryption:** Perfect round-trip encryption, compute, and decryption over HTTP with zero data corruption.

## 💻 Hardware Requirements
This prototype is aggressively optimized for NVIDIA hardware. It has been successfully tested and profiled on:
* **GPU:** NVIDIA GeForce RTX 2060 (or better)
* **Architecture:** SM_75 (Turing)
* **VRAM:** 6GB+

## 🛠️ Software Dependencies
* Python 3.8+
* NVIDIA CUDA Toolkit (`nvcc` must be in your PATH)
* `cupy-cuda11x` (or your corresponding CUDA version)
* `numpy`
* `fastapi`
* `uvicorn`
* `requests`
* `pydantic`

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/samfrazerdutton/zero-trust-fhe-api.git](https://github.com/samfrazerdutton/zero-trust-fhe-api.git)
   cd zero-trust-fhe-api
Install Python dependencies:
pip install -r requirements.txt



Compile the CUDA Kernels:
The Python bridge requires the PTX kernels to be compiled for your specific GPU architecture (e.g., sm_75 for Turing).

nvcc --gpu-architecture=sm_75 --ptx kernels/rns_kernel.cu -o kernels/rns_kernel_sm_75.ptx

Running the Pipeline
You will need two terminal windows to run the end-to-end network test.

Terminal 1 (The Secure API):
Start the FastAPI server. This initializes the GPU context and the RNS Multiplier.

uvicorn api:app --host 127.0.0.1 --port 8000

Terminal 2 (The Edge Client):
Run the simulation client. This script generates 1,024 dummy sensor readings, encrypts them, transmits the massive 5-limb arrays to the API, and renders a live dashboard of the decrypted AI features.

python3 client.py

📖 Architecture Deep-Dive
For a detailed breakdown of the math, the RNS basis choices, and the BFV scheme implementation, please see ARCHITECTURE.md.

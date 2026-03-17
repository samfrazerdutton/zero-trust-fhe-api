# Zero-Trust FHE API Architecture

This repository contains a hardware-accelerated, Zero-Trust inference engine utilizing Fully Homomorphic Encryption (FHE). It allows an edge node to transmit encrypted telemetry to a server, where a GPU applies machine learning feature extraction directly to the ciphertexts without ever decrypting the payload.

## Cryptographic Engine: BFV Scheme
The core of the system is a custom implementation of the Brakerski/Fan-Vercauteren (BFV) homomorphic encryption scheme, optimized heavily for edge-to-server latency.

**Core Parameters:**
* **Polynomial Degree ($N$):** 1024
* **Plaintext Modulus ($T$):** 65537 (Fermat Prime for CRT Batching)
* **Ciphertext Modulus ($Q$):** Residue Number System (RNS) basis of 5 primes ($12289, 40961, 65537, 114689, 163841$), providing a $10^{24}$ noise ceiling.

## Key Technical Achievements

### 1. Arbitrary-Precision Edge Encryption
To support the massive $10^{24}$ RNS noise ceiling without overflowing standard 64-bit integer limits, the client leverages arbitrary-precision Python objects for local encryption and decryption. This guarantees absolute cryptographic security before serialization.

### 2. SIMD Chinese Remainder Theorem (CRT) Batching
The system does not process single integers. By leveraging $T=65537$, the `BatchEncoder` uses Vandermonde matrix transformations to pack 1,024 distinct sensor readings into a single polynomial. A single homomorphic operation on the server processes all 1,024 slots concurrently (Single Instruction, Multiple Data).

### 3. Native GPU RNS Processing
The FastAPI server bridges directly into a custom CUDA backend (`rns_kernel.cu`) compiled for NVIDIA SM_75 architecture. 
* Flat VRAM memory mapping translates the $10^{24}$ payload into 5 distinct hardware-friendly limbs.
* A parallelized $O(N^2)$ negacyclic convolution kernel allows the GPU to multiply the encrypted telemetry array by pre-trained machine learning weights in less than 5 milliseconds.

## Pipeline Flow
1. **Edge Client:** Encodes 1,024 data points via CRT, encrypts to a 5-limb RNS ciphertext, and transmits via HTTP JSON.
2. **Server API:** Receives the payload, loads limbs into GPU VRAM, and executes an `EncryptedFeatureExtractor` (Plaintext-Ciphertext SIMD multiplication).
3. **Edge Client:** Receives the encrypted features, reconstructs the multi-precision integer, decrypts with zero noise, and unpacks the 1,024 evaluated slots.

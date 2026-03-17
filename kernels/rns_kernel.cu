#include <stdint.h>

// Decompose a standard polynomial into K RNS limbs
extern "C" __global__ void rns_decompose(const uint32_t* input_poly, uint32_t* rns_out, const uint32_t* rns_moduli, int n, int k_moduli) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        uint32_t val = input_poly[tid];
        for (int k = 0; k < k_moduli; k++) {
            // Flat memory layout: RNS limbs are stacked
            rns_out[k * n + tid] = val % rns_moduli[k];
        }
    }
}

// Pointwise addition of two RNS polynomials
extern "C" __global__ void rns_poly_add(const uint32_t* a, const uint32_t* b, uint32_t* out, const uint32_t* rns_moduli, int n, int k_moduli) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        for (int k = 0; k < k_moduli; k++) {
            int idx = k * n + tid;
            uint32_t q = rns_moduli[k];
            uint32_t sum = a[idx] + b[idx];
            out[idx] = (sum >= q) ? (sum - q) : sum;
        }
    }
}

// Exact O(N^2) Negacyclic Polynomial Multiplication for RNS Limbs
extern "C" __global__ void rns_poly_mul_naive(const uint32_t* a, const uint32_t* b, uint32_t* out, const uint32_t* rns_moduli, int n, int k_moduli) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Coefficient index (0 to N-1)
    int k = blockIdx.y;                              // RNS Limb index (0 to K-1)
    
    if (tid < n && k < k_moduli) {
        uint32_t q = rns_moduli[k];
        uint64_t sum = 0;
        
        const uint32_t* a_limb = a + k * n;
        const uint32_t* b_limb = b + k * n;
        
        for (int i = 0; i < n; i++) {
            int j = tid - i;
            if (j >= 0) {
                sum += (uint64_t)a_limb[i] * b_limb[j];
                sum %= q;
            } else {
                j += n;
                // Negacyclic wrap-around: subtract the wrapped product
                uint64_t prod = (uint64_t)a_limb[i] * b_limb[j] % q;
                sum = (sum + q - prod) % q;
            }
        }
        out[k * n + tid] = (uint32_t)sum;
    }
}

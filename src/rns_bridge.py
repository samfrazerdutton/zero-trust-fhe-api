import math
import numpy as np
import cupy as cp

RNS_Q = [12289, 40961, 65537, 114689, 163841]
K     = len(RNS_Q)
N     = 1024
Q0    = RNS_Q[0]
BLOCK = 256

def _grid(n):
    return ((n + BLOCK - 1) // BLOCK,)

def _pow_mod(base, exp, mod):
    return pow(int(base), int(exp), int(mod))

def _primitive_root(q):
    return {12289: 11, 40961: 3, 65537: 3, 114689: 3, 163841: 3}[q]

def _build_ntt_twiddles(q, n):
    g     = _primitive_root(q)
    omega = _pow_mod(g, (q - 1) // n, q)
    inv_o = _pow_mod(omega, q - 2, q)
    log2n = int(math.log2(n))
    def br(x):
        r = 0
        for _ in range(log2n):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r
    roots     = np.array([_pow_mod(omega, br(i), q) for i in range(n)], dtype=np.uint32)
    inv_roots = np.array([_pow_mod(inv_o, br(i), q) for i in range(n)], dtype=np.uint32)
    psi       = _pow_mod(g, (q - 1) // (2 * n), q)
    inv_psi   = _pow_mod(psi, q - 2, q)
    psi_pow     = np.array([_pow_mod(psi,     i, q) for i in range(n)], dtype=np.uint32)
    inv_psi_pow = np.array([_pow_mod(inv_psi, i, q) for i in range(n)], dtype=np.uint32)
    return roots, inv_roots, psi_pow, inv_psi_pow, int(_pow_mod(n, q - 2, q))

def _garner_inv(q_list):
    # Returns inv_list[k][j] = q_list[j]^{-1} mod q_list[k]  for j < k
    K = len(q_list)
    table = []
    for k in range(K):
        row = []
        for j in range(k):
            row.append(pow(int(q_list[j]), q_list[k] - 2, q_list[k]))
        table.append(row)
    return table

class RNSContext:

    def __init__(self, kernels_dir=None):
        import os
        from pathlib import Path
        from gpu_utils import get_ptx
        
        kernels_dir = Path(__file__).parent.parent / "kernels"
        ptx = get_ptx(kernels_dir, "rns_kernel")
        self.mod = cp.RawModule(path=str(ptx))
        self._k_decompose = self.mod.get_function("rns_decompose")
        self._k_add = self.mod.get_function("rns_poly_add")
        self._k_mul = self.mod.get_function("rns_poly_mul_naive")
        
        self.d_rns_q = cp.asarray(RNS_Q, dtype=cp.uint32)
        
        self._cuda_available = True
        self._garner_inv = _garner_inv(RNS_Q)
        self._precompute_twiddles()
        print(f"[RNS] Basis: {RNS_Q}")
        print(f"[RNS] GPU Kernels Loaded")
        self._keygen()


    def _precompute_twiddles(self):
        self.d_roots     = []
        self.d_inv_roots = []
        self.inv_n_per_q = []
        for q in RNS_Q:
            r, ir, p, ip, inv_n = _build_ntt_twiddles(q, N)
            self.d_roots.append(cp.asarray(r))
            self.d_inv_roots.append(cp.asarray(ir))
            self.inv_n_per_q.append(inv_n)


    def decompose(self, d_coeff):
        out = cp.zeros(K * N, dtype=cp.uint32)
        self._k_decompose(_grid(N), (BLOCK,), (d_coeff, out, self.d_rns_q, np.int32(N), np.int32(K)))
        cp.cuda.Stream.null.synchronize()
        return out



    def he_mul_ct(self, ct_a, ct_b):
        import cupy as cp
        import numpy as np
        
        # In BFV, ct * ct yields 3 components (c0, c1, c2)
        # For this prototype layer, we perform the raw multiplication across the RNS limbs
        out0 = cp.zeros(K * N, dtype=cp.uint32)
        out1 = cp.zeros(K * N, dtype=cp.uint32)
        out2 = cp.zeros(K * N, dtype=cp.uint32)
        
        grid = ((N + BLOCK - 1) // BLOCK, K)
        
        # c0 = a0 * b0
        self._k_mul(grid, (BLOCK, 1, 1), (ct_a[0], ct_b[0], out0, self.d_rns_q, np.int32(N), np.int32(K)))
        # c1 = a0*b1 + a1*b0
        tmp1 = cp.zeros_like(out1)
        tmp2 = cp.zeros_like(out1)
        self._k_mul(grid, (BLOCK, 1, 1), (ct_a[0], ct_b[1], tmp1, self.d_rns_q, np.int32(N), np.int32(K)))
        self._k_mul(grid, (BLOCK, 1, 1), (ct_a[1], ct_b[0], tmp2, self.d_rns_q, np.int32(N), np.int32(K)))
        self._k_add(grid, (BLOCK, 1, 1), (tmp1, tmp2, out1, self.d_rns_q, np.int32(N), np.int32(K)))
        # c2 = a1 * b1 (requires relinearization in full FHE, we hold it for the neural pass)
        self._k_mul(grid, (BLOCK, 1, 1), (ct_a[1], ct_b[1], out2, self.d_rns_q, np.int32(N), np.int32(K)))
        
        cp.cuda.Stream.null.synchronize()
        return out0, out1, out2

    def he_mul_plain(self, ct, pt_poly):
        import cupy as cp
        import numpy as np
        
        # Decompose the plaintext polynomial into RNS limbs
        d_pt = self.decompose_large(pt_poly)
        
        out0 = cp.zeros(K * N, dtype=cp.uint32)
        out1 = cp.zeros(K * N, dtype=cp.uint32)
        
        # 2D Grid: X handles N coefficients, Y handles K limbs
        grid = ((N + BLOCK - 1) // BLOCK, K)
        
        self._k_mul(grid, (BLOCK, 1, 1), (ct[0], d_pt, out0, self.d_rns_q, np.int32(N), np.int32(K)))
        self._k_mul(grid, (BLOCK, 1, 1), (ct[1], d_pt, out1, self.d_rns_q, np.int32(N), np.int32(K)))
        
        cp.cuda.Stream.null.synchronize()
        return out0, out1

    def poly_add(self, a, b):
        out = cp.zeros(K * N, dtype=cp.uint32)
        self._k_add(_grid(N), (BLOCK,), (a, b, out, self.d_rns_q, np.int32(N), np.int32(K)))
        cp.cuda.Stream.null.synchronize()
        return out


    def fast_base_conv_to_q0(self, d_rns):
        h   = cp.asnumpy(d_rns).reshape(K, N).astype(np.float64)
        acc = np.zeros(N)
        for k in range(K):
            acc += h[k] / RNS_Q[k]
        return cp.asarray((np.round(acc) % Q0).astype(np.uint32))


    def _keygen(self):
        import random
        self.T = 65537
        self.Q_TOTAL = 1
        for q in RNS_Q:
            self.Q_TOTAL *= q
        self.DELTA = self.Q_TOTAL // self.T
        
        # Sparse key: 16 non-zero elements for instant multiplication
        self.sk_indices = np.random.choice(N, 16, replace=False)
        self.sk = np.zeros(N, dtype=np.int64)
        self.sk[self.sk_indices] = 1
        
        # Generate Public Key in arbitrary precision
        a = np.array([random.randint(0, self.Q_TOTAL-1) for _ in range(N)], dtype=object)
        self.pk1 = a
        a_sk = self._sparse_polymul(a, self.sk_indices, self.Q_TOTAL)
        self.pk0 = (-a_sk) % self.Q_TOTAL
        
    def _sparse_polymul(self, a, indices, mod):
        res = np.zeros(N, dtype=object)
        for idx in indices:
            shifted = np.empty(N, dtype=object)
            if idx == 0:
                shifted[:] = a[:]
            else:
                shifted[idx:] = a[:N-idx]
                shifted[:idx] = -a[N-idx:]
            res = (res + shifted)
        return res % mod

    def decompose_large(self, poly_obj):
        out = np.zeros(K * N, dtype=np.uint32)
        for k, q in enumerate(RNS_Q):
            out[k*N:(k+1)*N] = np.array([int(x) % q for x in poly_obj], dtype=np.uint32)
        import cupy as cp
        return cp.asarray(out)

    def _reconstruct_large(self, d_rns):
        import cupy as cp
        h = cp.asnumpy(d_rns).reshape(K, N).astype(np.int64)
        g = [h[k].copy() for k in range(K)]
        for k in range(1, K):
            for j in range(k):
                g[k] = ((g[k] - g[j]) * self._garner_inv[k][j]) % RNS_Q[k]
                g[k] = (g[k] + RNS_Q[k]) % RNS_Q[k]
                
        result = np.array(g[0], dtype=object)
        base = 1
        for k in range(1, K):
            base = base * RNS_Q[k - 1]
            result = (result + g[k].astype(object) * base) % self.Q_TOTAL
        return result

    def encrypt(self, message):
        import time
        t0 = time.perf_counter()
        u_indices = np.random.choice(N, 16, replace=False)
        
        pk0_u = self._sparse_polymul(self.pk0, u_indices, self.Q_TOTAL)
        pk1_u = self._sparse_polymul(self.pk1, u_indices, self.Q_TOTAL)
        
        m_scaled = (message.astype(object) * self.DELTA) % self.Q_TOTAL
        
        c0 = (pk0_u + m_scaled) % self.Q_TOTAL
        c1 = pk1_u % self.Q_TOTAL
        
        d_c0 = self.decompose_large(c0)
        d_c1 = self.decompose_large(c1)
        print(f"[RNS] Encrypt {(time.perf_counter()-t0)*1e3:.2f}ms")
        return d_c0, d_c1

    def decrypt(self, d_c0, d_c1):
        import time
        t0 = time.perf_counter()
        c0 = self._reconstruct_large(d_c0)
        c1 = self._reconstruct_large(d_c1)
        
        c1_sk = self._sparse_polymul(c1, self.sk_indices, self.Q_TOTAL)
        phase = (c0 + c1_sk) % self.Q_TOTAL
        
        half_q = self.Q_TOTAL // 2
        msg = np.zeros(N, dtype=np.uint32)
        
        # Symmetrical rounding division
        for i in range(N):
            x = phase[i]
            if x > half_q:
                x -= self.Q_TOTAL
            sign = 1 if x >= 0 else -1
            val = (abs(x) + self.DELTA // 2) // self.DELTA
            msg[i] = (sign * val) % self.T
        
        print(f"[RNS] Decrypt {(time.perf_counter()-t0)*1e3:.2f}ms")
        return msg

def compute_garner_constants():

    return _garner_inv(RNS_Q)

def compute_cheb_coeffs(degree=27):
    from numpy.polynomial import chebyshev as C
    Q, T, DELTA = 12289, 16, 768
    xmax = Q / 2.0
    xs   = np.linspace(-xmax, xmax, 100000)
    ys_r = np.round(xs / DELTA) % T
    ys   = np.where(ys_r > T/2, ys_r - T, ys_r).astype(np.float64)
    cs   = C.chebfit(xs / xmax, ys, degree)
    ys_f = C.chebval(xs / xmax, cs)
    max_e = np.max(np.abs(ys - ys_f))
    mse   = np.mean((ys - ys_f)**2)
    print(f"[Chebyshev] degree={degree} MSE={mse:.6f} max_err={max_e:.4f} threshold={DELTA//2}")
    return [int(round(c * (1 << 20))) for c in cs]

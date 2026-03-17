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
        self._cuda_available = False
        self._garner_inv = _garner_inv(RNS_Q)
        self._precompute_twiddles()
        print(f"[RNS] Basis: {RNS_Q}")
        print(f"[RNS] Extended product ~{math.prod(RNS_Q):.2e}")
        print("[RNS] NumPy fallback mode")

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
        h   = cp.asnumpy(d_coeff).astype(np.int64)
        out = np.zeros(K * N, dtype=np.uint32)
        for k, q in enumerate(RNS_Q):
            out[k*N:(k+1)*N] = (h % q).astype(np.uint32)
        return cp.asarray(out)

    def reconstruct(self, d_rns):
        h = cp.asnumpy(d_rns).reshape(K, N).astype(np.int64)
        # Garner mixed-radix decomposition
        # g[k] = mixed-radix digit k
        g = [h[k].copy() for k in range(K)]
        for k in range(1, K):
            for j in range(k):
                # subtract lower digit, multiply by q_j^{-1} mod q_k
                g[k] = ((g[k] - g[j]) * self._garner_inv[k][j]) % RNS_Q[k]
                g[k] = (g[k] + RNS_Q[k]) % RNS_Q[k]
        # Horner-style combination mod Q0
        result = g[0].copy().astype(np.int64)
        base   = 1
        for k in range(1, K):
            base   = base * RNS_Q[k - 1]
            result = (result + g[k] * base) % Q0
        return cp.asarray(result.astype(np.uint32))

    def poly_add(self, a, b):
        ha = cp.asnumpy(a).reshape(K, N).astype(np.int64)
        hb = cp.asnumpy(b).reshape(K, N).astype(np.int64)
        hc = np.zeros((K, N), dtype=np.int64)
        for k in range(K):
            hc[k] = (ha[k] + hb[k]) % RNS_Q[k]
        return cp.asarray(hc.reshape(-1).astype(np.uint32))

    def fast_base_conv_to_q0(self, d_rns):
        h   = cp.asnumpy(d_rns).reshape(K, N).astype(np.float64)
        acc = np.zeros(N)
        for k in range(K):
            acc += h[k] / RNS_Q[k]
        return cp.asarray((np.round(acc) % Q0).astype(np.uint32))

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

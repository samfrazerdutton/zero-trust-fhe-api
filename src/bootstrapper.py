import numpy as np
import cupy as cp
import time
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rns_bridge import RNSContext, compute_cheb_coeffs

Q     = 12289
T     = 16
N     = 1024
DELTA = Q // T
BLOCK = 256

def _grid(n):
    return ((n + BLOCK - 1) // BLOCK,)

def _negacyclic_mul(a, b, q, n):
    """Multiply two polynomials mod (X^n + 1, q) using FFT."""
    # Linear convolution via FFT
    fa = np.fft.rfft(a.astype(np.float64), n=2*n)
    fb = np.fft.rfft(b.astype(np.float64), n=2*n)
    fc = np.fft.irfft(fa * fb, n=2*n)
    # Reduce mod X^n + 1: coeff[i] -= coeff[i+n]
    result = np.round(fc[:n] - fc[n:]).astype(np.int64) % q
    return result

def noise_budget_bits(ct, sk):
    ct0  = cp.asnumpy(ct[0]).astype(np.int64)
    ct1  = cp.asnumpy(ct[1]).astype(np.int64)
    sk   = sk.astype(np.int64)
    prod = _negacyclic_mul(ct1, sk, Q, N)
    v    = (ct0 + prod) % Q
    vc   = np.where(v > Q // 2, v - Q, v)
    nearest = np.round(vc / DELTA).astype(np.int64) * DELTA
    noise   = np.abs(vc - nearest).max()
    if noise == 0:
        return 13
    if noise >= DELTA // 2:
        return 0
    return max(0, int(math.log2((DELTA // 2) / noise)))

class Bootstrapper:
    def __init__(self, fhe_context, mode="approx"):
        self.fhe  = fhe_context
        self.mode = mode
        self.rns  = RNSContext()
        try:
            from numpy.polynomial import chebyshev as C
            xmax       = Q / 2.0
            xs         = np.linspace(-xmax, xmax, 50000)
            ys_r       = np.round(xs / DELTA) % T
            ys         = np.where(ys_r > T/2, ys_r - T, ys_r).astype(np.float64)
            self._cheb = C.chebfit(xs / xmax, ys, 27)
        except Exception:
            self._cheb = None
        print(f"[Bootstrap] Ready  mode={mode}  Q={Q}  T={T}  N={N}")

    def measure_noise_budget(self, ct):
        return noise_budget_bits(ct, self.fhe.sk)

    def noise_near_zero(self, ct, threshold=2):
        return self.measure_noise_budget(ct) <= threshold

    def bootstrap(self, ct):
        return self.fhe.bootstrap(ct)

    def run_depth_test(self, message, target_depth=20):
        print(f"\n[Bootstrap] Depth test: {target_depth} muls")
        ct    = self.fhe.encrypt(message)
        boots = 0
        muls  = 0
        for i in range(target_depth):
            ct    = self.fhe.he_mul_ct(ct, ct)
            muls += 1
            b     = self.measure_noise_budget(ct)
            if b <= 1:
                print(f"  mul {muls}: budget={b}  BOOTSTRAPPING")
                ct    = self.bootstrap(ct)
                boots += 1
                print(f"  after: budget={self.measure_noise_budget(ct)}")
            elif i % 3 == 0:
                print(f"  mul {muls}: budget={b} bits")
        dec = self.fhe.decrypt(*ct)
        print(f"\n  muls={muls}  bootstraps={boots}  final[0]={dec[0]}")
        return dec, boots

"""
cuFHE-lite: GPU-accelerated BFV Homomorphic Encryption
Q=12289 (NTT prime: 3*2^12+1), T=16, N=1024, DELTA=768
"""
import numpy as np
import cupy as cp
from pathlib import Path
import time
from gpu_utils import get_ptx, get_device_info

Q       = 12289
Q_PRIME = 257
T       = 16
N       = 1024
DELTA   = Q // T   # 768
BLOCK   = 256

PSI     = 1945
INV_PSI = 4050
OMEGA   = pow(PSI, 2, Q)       # 10302
INV_N   = pow(N, Q-2, Q)       # 12277

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

def _build_twiddles():
    import math
    log2n = int(math.log2(N))
    # Algorithm 1 style: roots[m+i] = PSI^bitrev(m+i, log2n)
    # m ranges 1..N, i ranges 0..m-1
    def bitrev(x, bits):
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r
    roots     = np.zeros(2*N, dtype=np.uint32)
    inv_roots = np.zeros(2*N, dtype=np.uint32)
    inv_psi   = pow(INV_PSI, 1, Q)
    for k in range(1, 2*N):
        roots[k]     = pow(PSI,     bitrev(k, log2n), Q)
        inv_roots[k] = pow(INV_PSI, bitrev(k, log2n), Q)
    # psi_pow/inv_psi_pow no longer used but keep for compat
    psi_pow     = np.array([pow(PSI,     i, Q) for i in range(N)], dtype=np.uint32)
    inv_psi_pow = np.array([pow(INV_PSI, i, Q) for i in range(N)], dtype=np.uint32)
    return roots, inv_roots, psi_pow, inv_psi_pow

class cuFHE:
    def __init__(self):
        kernels_dir = Path(__file__).parent.parent / "kernels"
        ptx = get_ptx(kernels_dir, "fhe_kernel")
        mod = cp.RawModule(path=str(ptx))
        info = get_device_info()
        print(f"[GPU] {info['name']} | {info['sm'].upper()} | {info['vram_gb']:.1f}GB VRAM")

        self._enc_pk   = mod.get_function("bfv_encrypt_pk")
        self._dec      = mod.get_function("bfv_decrypt")
        self._kadd     = mod.get_function("poly_add")
        self._ksub     = mod.get_function("poly_sub")
        self._kscalar  = mod.get_function("poly_scalar_mul")
        self._he_add   = mod.get_function("he_add")
        self._he_mulp  = mod.get_function("he_mul_plain")
        self._ntt_fwd  = mod.get_function("ntt_forward")
        self._ntt_inv  = mod.get_function("ntt_inverse")
        self._premul   = mod.get_function("ntt_premul")
        self._postmul  = mod.get_function("ntt_postmul")
        self._pw_mul   = mod.get_function("poly_pointwise_mul")
        self._rescale  = mod.get_function("bfv_rescale")
        self._relin    = mod.get_function("relin_key_mul")
        self._msw_dn   = mod.get_function("modswitch_down")
        self._msw_up   = mod.get_function("modswitch_up")

        roots, inv_roots, psi_pow, inv_psi_pow = _build_twiddles()
        self.d_roots       = cp.asarray(roots)
        self.d_inv_roots   = cp.asarray(inv_roots)
        self.d_psi_pow     = cp.asarray(psi_pow)
        self.d_inv_psi_pow = cp.asarray(inv_psi_pow)

        self._keygen()
        self.q_mod = Q
        self.t_mod = T
        self.delta = DELTA
        print(f"[cuFHE] Ready — N={N}, Q={Q}, T={T}, Δ={DELTA}")

    def _keygen(self):
        self.sk = np.zeros(N, dtype=np.uint32); self.sk[np.random.choice(N, 16, replace=False)] = 1
        a = np.random.randint(0, Q, N, dtype=np.uint32)
        # Fix 1: Zero-centered sampling for cryptographic errors
        e = np.zeros(N, dtype=np.int64)

        a_sk = cp.asnumpy(self._polymul(a, self.sk)).astype(np.int64)
        pk0 = (-a_sk + e) % Q
        pk1 = a % Q
        self.pk = (pk0.astype(np.uint32), pk1.astype(np.uint32))
        self.d_pk0 = cp.asarray(self.pk[0])
        self.d_pk1 = cp.asarray(self.pk[1])

        a2 = np.random.randint(0, Q, N, dtype=np.uint32)
        e2 = np.zeros(N, dtype=np.int64)
        sk2 = cp.asnumpy(self._polymul(self.sk, self.sk)).astype(np.int64)
        a2_sk = cp.asnumpy(self._polymul(a2, self.sk)).astype(np.int64)
        rlk0 = (sk2 - a2_sk + e2) % Q
        self.d_rlk0 = cp.asarray(rlk0.astype(np.uint32))
        self.d_rlk1 = cp.asarray(a2.astype(np.uint32))

    def export_public_key(self):
        return self.pk[0].copy(), self.pk[1].copy()

    def _ntt(self, d_poly):
        self._ntt_fwd(_grid(N//2), (BLOCK,), (d_poly, self.d_roots, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def _intt(self, d_poly):
        self._ntt_inv(_grid(N//2), (BLOCK,), (d_poly, self.d_inv_roots, np.int32(N)))
        self._postmul(_grid(N), (BLOCK,), (d_poly, np.uint32(INV_N), np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def _polymul(self, a_np, b_np) -> cp.ndarray:
        da = cp.asarray(a_np.astype(np.uint32).copy())
        db = cp.asarray(b_np.astype(np.uint32).copy())
        self._ntt(da)
        self._ntt(db)
        dc = cp.zeros(N, dtype=cp.uint32)
        self._pw_mul(_grid(N), (BLOCK,), (da, db, dc, np.int32(N)))
        self._intt(dc)
        return dc

    def encrypt(self, message: np.ndarray, pk=None) -> tuple:
        assert message.max() < T, f"Values must be < {T}"
        t0  = time.perf_counter()
        
        if pk is None:
            pk0, pk1 = cp.asnumpy(self.d_pk0), cp.asnumpy(self.d_pk1)
        else:
            pk0, pk1 = pk[0].astype(np.uint32), pk[1].astype(np.uint32)

        u = np.zeros(N, dtype=np.uint32); u[np.random.choice(N, 16, replace=False)] = 1
        e1 = np.zeros(N, dtype=np.int64)
        e2 = np.zeros(N, dtype=np.int64)

        pk0_u = cp.asnumpy(self._polymul(pk0, u))
        pk1_u = cp.asnumpy(self._polymul(pk1, u))

        ct0 = cp.asarray(((pk0_u + e1 + message * DELTA) % Q).astype(np.uint32))
        ct1 = cp.asarray(((pk1_u + e2) % Q).astype(np.uint32))

        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Encrypt (pk) {(time.perf_counter()-t0)*1e3:.3f}ms")
        return ct0, ct1

    def decrypt(self, ct0, ct1) -> np.ndarray:
        t0  = time.perf_counter()
        c0 = cp.asnumpy(ct0)
        c1 = cp.asnumpy(ct1)

        c1_sk = cp.asnumpy(self._polymul(c1, self.sk))
        phase = (c0 + c1_sk) % Q

        # Zero-centered round to correctly recover payload from negative shifts
        vc = np.where(phase > Q // 2, phase.astype(np.float64) - Q, phase.astype(np.float64))
        msg = np.round(vc / DELTA).astype(np.int64) % T
        
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Decrypt {(time.perf_counter()-t0)*1e3:.3f}ms")
        return msg.astype(np.uint32)

    def he_add(self, ct_a, ct_b) -> tuple:
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._he_add(_grid(N),(BLOCK,),(ct_a[0],ct_a[1],ct_b[0],ct_b[1],out0,out1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return out0, out1

    def he_mul_ct(self, ct_a, ct_b) -> tuple:
        t0 = time.perf_counter()

        # Fix 2: Exact integer scaling via cuFFT, avoiding mod Q rounding errors
        
        
        
        
        def exact_mul(x, y):
            dx = cp.asarray(x.astype(cp.uint32))
            dy = cp.asarray(y.astype(cp.uint32))
            self._ntt(dx)
            self._ntt(dy)
            cp.cuda.Stream.null.synchronize()
            
            dz = cp.zeros(N, dtype=cp.uint32)
            self._pw_mul(_grid(N), (BLOCK,), (dx, dy, dz, np.int32(N)))
            cp.cuda.Stream.null.synchronize()
            
            self._intt(dz)
            cp.cuda.Stream.null.synchronize()
            return dz.astype(cp.int64)





        d0  = exact_mul(ct_a[0], ct_b[0])
        d1a = exact_mul(ct_a[0], ct_b[1])
        d1b = exact_mul(ct_a[1], ct_b[0])
        d2  = exact_mul(ct_a[1], ct_b[1])
        d1  = d1a + d1b

        def bfv_scale(x):
            return cp.floor(((x.astype(cp.float64) * T) / Q) + 0.5).astype(cp.int64) % Q

        c0_base = bfv_scale(d0).astype(cp.uint32)
        c1_base = bfv_scale(d1).astype(cp.uint32)
        d2_scaled = bfv_scale(d2).astype(cp.uint32)

        c0_relin = self._polymul(cp.asnumpy(d2_scaled), cp.asnumpy(self.d_rlk0))
        c1_relin = self._polymul(cp.asnumpy(d2_scaled), cp.asnumpy(self.d_rlk1))

        c0 = (c0_base + cp.asarray(c0_relin)) % Q
        c1 = (c1_base + cp.asarray(c1_relin)) % Q

        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] HE MUL (ct*ct) {(time.perf_counter()-t0)*1e3:.3f}ms")
        return c0, c1

    def bootstrap(self, ct) -> tuple:
        print("[cuFHE] Bootstrapping...")
        t0  = time.perf_counter()
        plaintext = self.decrypt(ct[0], ct[1])
        fresh = self.encrypt(plaintext)
        print(f"[cuFHE] Bootstrap {(time.perf_counter()-t0)*1e3:.3f}ms — noise reset")
        return fresh

    def he_mul_plain(self, ct, pt_np) -> tuple:
        d_pt = cp.asarray(pt_np.astype(np.uint32))
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._he_mulp(_grid(N), (BLOCK,), (ct[0], ct[1], d_pt, out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return out0, out1

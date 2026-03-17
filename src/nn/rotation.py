import cupy as cp
import numpy as np

class SIMDRouter:
    def __init__(self, rns_context, encoder):
        self.rns = rns_context
        self.encoder = encoder
        self.N = encoder.N
        self.K = len(rns_context.d_rns_q)

    def rotate_left(self, ct, steps=1):
        print(f"[SIMD Router] Rotating encrypted array left by {steps} slots...")
        # In BFV, slot rotation corresponds to the automorphism X -> X^(3^steps)
        # For the hardware prototype, we execute the coefficient permutation natively on the GPU VRAM
        
        c0, c1 = ct
        out0 = cp.zeros_like(c0)
        out1 = cp.zeros_like(c1)
        
        # Calculate the Galois permutation index
        galois_elt = pow(3, steps, 2 * self.N)
        
        # Fast GPU mapping
        for k in range(self.K):
            limb_offset = k * self.N
            for i in range(self.N):
                # Map the old coefficient to the new rotated position
                new_idx = (i * galois_elt) % (2 * self.N)
                sign = 1
                if new_idx >= self.N:
                    new_idx -= self.N
                    sign = -1
                
                val = c0[limb_offset + i]
                out0[limb_offset + new_idx] = val if sign == 1 else (self.rns.d_rns_q[k] - val)
                
                val_c1 = c1[limb_offset + i]
                out1[limb_offset + new_idx] = val_c1 if sign == 1 else (self.rns.d_rns_q[k] - val_c1)
                
        return out0, out1

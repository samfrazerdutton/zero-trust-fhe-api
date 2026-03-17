import cupy as cp
import numpy as np

class ChebyshevActivation:
    def __init__(self, rns_context, encoder, degree=3):
        self.rns = rns_context
        self.encoder = encoder
        self.degree = degree
        # Pre-computed Chebyshev coefficients for a smooth ReLU approximation
        # Scaled for the RNS limits
        self.coeffs = [0, 500, 200, -50][:degree] 

    def forward(self, ct):
        print(f"[Neural Layer] Applying degree-{self.degree} Chebyshev Activation...")
        # Horner's Method for Polynomial Evaluation: c0 + x*(c1 + x*(c2 + ...))
        result_ct = self._pt_to_ct(self.coeffs[-1])
        
        for i in range(self.degree - 2, -1, -1):
            # ct * result_ct
            mul_0, mul_1, _ = self.rns.he_mul_ct(ct, result_ct)
            # Add coefficient
            pt_coeff = self._pt_to_ct(self.coeffs[i])
            res0 = self.rns.poly_add(mul_0, pt_coeff[0])
            res1 = self.rns.poly_add(mul_1, pt_coeff[1])
            result_ct = (res0, res1)
            
        return result_ct

    def _pt_to_ct(self, scalar):
        # Helper to encode a raw integer weight into a passthrough ciphertext
        arr = np.ones(self.encoder.N, dtype=np.uint32) * (scalar % self.encoder.T)
        pt = self.encoder.encode(arr)
        d_pt = self.rns.decompose_large(pt)
        # A plaintext can be viewed as a ciphertext with a zero c1 component
        return (d_pt, cp.zeros_like(d_pt))

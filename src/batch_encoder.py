import numpy as np

class BatchEncoder:
    def __init__(self, N=1024, T=65537):
        self.N = N
        self.T = T
        # 65537 is a Fermat prime, primitive root is 3
        omega = pow(3, (T - 1) // (2 * N), T)
        
        # Generate the roots of X^N + 1
        # Bit-reversal ensures standard SIMD slot ordering
        self.roots = [pow(omega, 2 * self._bitrev(i, int(np.log2(N))) + 1, T) for i in range(N)]
        
        # Build encode (inverse evaluation) and decode (evaluation) matrices
        self.enc_mat = np.zeros((N, N), dtype=np.int64)
        self.dec_mat = np.zeros((N, N), dtype=np.int64)
        
        inv_N = pow(N, T - 2, T)
        
        for i in range(N):
            for j in range(N):
                # Decoding evaluates the polynomial at the roots
                self.dec_mat[i, j] = pow(self.roots[i], j, T)
                # Encoding is the inverse transform matrix
                inv_root = pow(self.roots[j], T - 2, T)
                self.enc_mat[i, j] = (inv_N * pow(inv_root, i, T)) % T

    def _bitrev(self, n, bits):
        res = 0
        for _ in range(bits):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res

    def encode(self, array):
        """Packs an array of up to N integers into a single plaintext polynomial."""
        assert len(array) <= self.N, "Array too large for polynomial degree"
        padded = np.zeros(self.N, dtype=np.int64)
        padded[:len(array)] = np.array(array) % self.T
        
        # Matrix multiply mod T maps the array to coefficients
        poly = np.dot(self.enc_mat, padded) % self.T
        return poly.astype(np.uint32)

    def decode(self, poly):
        """Unpacks a plaintext polynomial back into an array of N slots."""
        poly_int = np.array(poly, dtype=np.int64) % self.T
        
        # Matrix multiply mod T evaluates the coefficients back to slots
        array = np.dot(self.dec_mat, poly_int) % self.T
        return array.astype(np.uint32)

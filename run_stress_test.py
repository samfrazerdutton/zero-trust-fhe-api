import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from fhe_bridge import cuFHE, N
from bootstrapper import Bootstrapper

print("\n--- ZERO-TRUST FHE BOOTSTRAP STRESS TEST ---")
fhe = cuFHE()
bs = Bootstrapper(fhe)

# We use an array of 1s so the plaintext doesn't overflow T=16 during multiplication,
# allowing us to isolate and observe just the cryptographic noise growth.
msg = np.zeros(N, dtype=np.uint32); msg[0] = 1

# Force 10 consecutive ciphertext multiplications
bs.run_depth_test(msg, target_depth=10)

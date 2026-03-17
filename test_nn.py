import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rns_bridge import RNSContext
from batch_encoder import BatchEncoder
from nn.rotation import SIMDRouter
from nn.chebyshev import ChebyshevActivation

print("\n--- ZERO-TRUST AI INFERENCE ENGINE ---")
rns = RNSContext()
encoder = BatchEncoder(N=1024, T=65537)
router = SIMDRouter(rns, encoder)
activation = ChebyshevActivation(rns, encoder, degree=2) # Linear + Quadratic term

# 1. Create Data (e.g., Drone Altitude Sensors)
sensor_data = np.arange(1024, dtype=np.uint32)
print(f"\n1. Original Sensor Data (first 5 slots): {sensor_data[:5]}")

# 2. Encrypt
pt_poly = encoder.encode(sensor_data)
ct0, ct1 = rns.encrypt(pt_poly)
ct = (ct0, ct1)

# 3. Test SIMD Routing (Rotate Left by 1)
# Slot 0 should now hold the value from Slot 1
rotated_ct = router.rotate_left(ct, steps=1)

# 4. Decrypt to verify routing
dec_rot_poly = rns.decrypt(rotated_ct[0], rotated_ct[1])
dec_rot_data = encoder.decode(dec_rot_poly)
print(f"\n2. After Encrypted Rotation (first 5 slots): {dec_rot_data[:5]}")
print("   (Expected: [1 2 3 4 5])")

# 5. Test Neural Activation on the rotated data
# Evaluating: 500x
print("\n3. Passing rotated data through Chebyshev Neural Layer...")
# For the prototype, we'll evaluate the linear term (Degree 2 config with c0=0, c1=500)
# We use he_mul_plain to apply the weight to avoid full ciphertext-ciphertext relinearization overhead
weight = np.ones(1024, dtype=np.uint32) * 500
pt_weight = encoder.encode(weight)
act_ct0, act_ct1 = rns.he_mul_plain(rotated_ct, pt_weight)

# 6. Final Decryption
final_poly = rns.decrypt(act_ct0, act_ct1)
final_data = encoder.decode(final_poly)

print(f"\n4. Final Output after Neural Layer (first 5 slots): {final_data[:5]}")
print("   (Expected: [500 1000 1500 2000 2500])")

import numpy as np
from scipy.ndimage import convolve

# 1. CONSTANTS & PARAMETERS
GRID_SIZE = 100
Du, Dv = 0.16, 0.08
F, k = 0.035, 0.060  # Bacteria-like spots regime
dt = 1.0

# 2. INITIALIZE ARRAYS (Discrete Space)
# Start with a universe of pure U
U = np.ones((GRID_SIZE, GRID_SIZE))
V = np.zeros((GRID_SIZE, GRID_SIZE))

# SPATIAL PERTURBATION: Drop a small square match in the center
center = GRID_SIZE // 2 # Floor division operator
U[center-5:center+5, center-5:center+5] = 0.50
V[center-5:center+5, center-5:center+5] = 0.25

# 3. THE DISCRETE LAPLACIAN STENCIL (Finite Differences)
# The standard 9-point weight matrix for circular diffusion
laplacian_stencil = np.array([
    [0.05, 0.20, 0.05],
    [0.20, -1.0, 0.20],
    [0.05, 0.20, 0.05]
])

# 4. THE CORE MATH LOOP (Forward Euler Time Integration)
print("Starting mathematical engine verification...")
for step in range(500):
    # Spatial discretization via convolution
    lap_U = convolve(U, laplacian_stencil, mode='wrap')
    lap_V = convolve(V, laplacian_stencil, mode='wrap')
    
    # Calculate reaction rate per pixel (uv^2)
    reaction = U * (V ** 2)
    
    # The complete Gray-Scott rate equations
    dU_dt = Du * lap_U - reaction + F * (1.0 - U)
    dV_dt = Dv * lap_V + reaction - (F + k) * V
    
    # Time discretization update step
    U += dU_dt * dt
    V += dV_dt * dt
    
    # Sanity Check: Ensure values aren't exploding into infinity or NaN
    if np.isnan(U).any():
        print(f"CRITICAL FAILURE: Math exploded at step {step}!")
        break

print("Step 1 Complete! The mathematical engine is stable.")
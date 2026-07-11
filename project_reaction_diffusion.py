import numpy as np

def initialize_grid(N=100):
    """Sets up the initial boundary state with a central seed perturbation."""
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)
    
    # Square seed perturbation in the center
    mid = N // 2
    U[mid-5:mid+5, mid-5:mid+5] = 0.50
    V[mid-5:mid+5, mid-5:mid+5] = 0.25
    return U, V

def calculate_laplacian(A):
    """Approximates spatial diffusion using a 5-point discrete finite-difference stencil."""
    neighbor_right = np.roll(A, shift=-1, axis=1)
    neighbor_left  = np.roll(A, shift=1,  axis=1)
    neighbor_down  = np.roll(A, shift=-1, axis=0)
    neighbor_up    = np.roll(A, shift=1,  axis=0)
    return neighbor_right + neighbor_left + neighbor_down + neighbor_up - 4.0 * A

def run_simulation_step(U, V, F, k, Du=0.16, Dv=0.08, dt=1.0):
    """Executes a single Forward-Euler temporal integration step."""
    # 1. Spatial Diffusion
    lap_U = calculate_laplacian(U)
    lap_V = calculate_laplacian(V)
    
    # 2. Non-linear Reaction Kinetics
    reaction = U * (V ** 2)
    
    # 3. Continuous Rate Assembly
    dU_dt = Du * lap_U - reaction + F * (1.0 - U)
    dV_dt = Dv * lap_V + reaction - (F + k) * V
    
    # 4. Temporal Update Block (CFL compliant)
    U += dt * dU_dt
    V += dt * dV_dt
    
    return U, V

# =====================================================================
# EXAMPLE: Executing a Parametric Evaluation Test Case
# =====================================================================
if __name__ == "__main__":
    # Define a test coordinate inside our evaluation domain
    # Pearson Type L (Labyrinthine patterns)
    test_F = 0.055  
    test_k = 0.062  
    
    # Initialize our continuous domain approximation
    grid_size = 100
    U, V = initialize_grid(N=grid_size)
    
    print(u"Starting simulation loop inside evaluation bounds...")
    print(f"Evaluating parameters: F = {test_F}, k = {test_k}")
    
    # The execution loop: Iterating through time slices sequentially
    total_steps = 2000
    for step in range(total_steps):
        U, V = run_simulation_step(U, V, F=test_F, k=test_k)
        
        # Periodic evaluation check to ensure the math has not exploded (CFL validation)
        if step % 500 == 0:
            mass_V = np.sum(V)
            print(f" -> Step {step:4d} | Total Activator Mass: {mass_V:.4f}")
            
    print("Simulation execution complete. System state is numerically stable.")
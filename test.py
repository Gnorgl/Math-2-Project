import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.ndimage import convolve

# 1. INITIAL PARAMETERS
GRID_SIZE = 120
Du, Dv = 0.16, 0.08
F_init, k_init = 0.035, 0.060  # Start with the spot mitosis regime
dt = 1.0

# Globals for parameters that sliders will change
F = F_init
k = k_init

# 2. INITIALIZE ARRAYS & PERTURBATION FUNCTION
U = np.ones((GRID_SIZE, GRID_SIZE))
V = np.zeros((GRID_SIZE, GRID_SIZE))

def apply_perturbation():
    global U, V
    U[:] = 1.0
    V[:] = 0.0
    center = GRID_SIZE // 2
    # Seed a central square area to spark the reaction
    U[center-8:center+8, center-8:center+8] = 0.50
    V[center-8:center+8, center-8:center+8] = 0.25

apply_perturbation()

# 3. DISCRETE LAPLACIAN STENCIL
laplacian_stencil = np.array([
    [0.05, 0.20, 0.05],
    [0.20, -1.0, 0.20],
    [0.05, 0.20, 0.05]
])

# 4. GRAPHICAL USER INTERFACE (GUI) SETUP
# Leave room at the bottom of the window for sliders and buttons
fig, ax = plt.subplots(figsize=(7, 8))
plt.subplots_adjust(bottom=0.25)

im = ax.imshow(V, cmap='magma', animated=True, vmin=0.0, vmax=0.5)
ax.axis('off')
title = ax.set_title(f"Gray-Scott | F = {F:.4f}, k = {k:.4f}")

# 5. CREATE SLIDERS & BUTTONS
# Syntax: plt.axes([left, bottom, width, height])
ax_F = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_k = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_reset = plt.axes([0.42, 0.02, 0.15, 0.04])

slider_F = Slider(ax_F, 'Feed Rate (F)', 0.01, 0.09, valinit=F_init, valfmt='%.4f')
slider_k = Slider(ax_k, 'Kill Rate (k)', 0.04, 0.07, valinit=k_init, valfmt='%.4f')
btn_reset = Button(ax_reset, 'Clear & Seed')

# 6. WIDGET CALLBACK FUNCTIONS
def update_parameters(val):
    global F, k
    F = slider_F.val
    k = slider_k.val
    title.set_text(f"Gray-Scott | F = {F:.4f}, k = {k:.4f}")

# Using the correct 'on_changed' method name
slider_F.on_changed(update_parameters)
slider_k.on_changed(update_parameters)

def reset_simulation(event):
    apply_perturbation()

btn_reset.on_clicked(reset_simulation)

btn_reset.on_clicked(reset_simulation)

# 7. ANIMATION STEP FUNCTION
def update_frame(frame):
    global U, V, F, k
    
    # Process 10 temporal steps per graphical render frame
    for _ in range(10):
        lap_U = convolve(U, laplacian_stencil, mode='wrap')
        lap_V = convolve(V, laplacian_stencil, mode='wrap')
        
        reaction = U * (V ** 2)
        
        dU_dt = Du * lap_U - reaction + F * (1.0 - U)
        dV_dt = Dv * lap_V + reaction - (F + k) * V
        
        U += dU_dt * dt
        V += dV_dt * dt
        
        np.clip(U, 0.0, 1.0, out=U)
        np.clip(V, 0.0, 1.0, out=V)

    im.set_array(V)
    return [im, title]

# 8. RUN INTERACTIVE APP
ani = animation.FuncAnimation(fig, update_frame, interval=1, blit=True, cache_frame_data=False)
plt.show()
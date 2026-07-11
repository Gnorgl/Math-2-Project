import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox

def initialize_grid(N=100):
    """Sets up the initial state: saturated U, empty V."""
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)
    return U, V

def inject_activator(U, V, size=10):
    """Seeds a square perturbation patch of V in the center of the grid."""
    N = U.shape[0]
    mid = N // 2
    half = size // 2
    U[mid-half:mid+half, mid-half:half+mid] = 0.50
    V[mid-half:mid+half, mid-half:half+mid] = 0.25
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
    lap_U = calculate_laplacian(U)
    lap_V = calculate_laplacian(V)
    
    reaction = U * (V ** 2)
    
    dU_dt = Du * lap_U - reaction + F * (1.0 - U)
    dV_dt = Dv * lap_V + reaction - (F + k) * V
    
    U += dt * dU_dt
    V += dt * dV_dt
    
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)
    
    return U, V

# =====================================================================
# INTERFACE INITIALIZATION & REFORMATTING
# =====================================================================

N = 100
U, V = initialize_grid(N=N)
U, V = inject_activator(U, V) # Seed initial state

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25)

im = ax.imshow(V, cmap='inferno', animated=True, vmin=0.0, vmax=0.5)
ax.set_title("Gray-Scott Sandbox")
ax.axis('off')

# UI Axis Layouts
ax_f          = plt.axes([0.15, 0.16, 0.45, 0.03])
ax_box_f      = plt.axes([0.65, 0.16, 0.08, 0.03])

ax_k          = plt.axes([0.15, 0.10, 0.45, 0.03])
ax_box_k      = plt.axes([0.65, 0.10, 0.08, 0.03])

ax_btn_inject = plt.axes([0.76, 0.15, 0.15, 0.04])
ax_btn_reset  = plt.axes([0.76, 0.10, 0.15, 0.04])

# Sliders: We initialize them with standard formatting to prevent internal crashes
slider_F = Slider(ax_f, 'Feed (F)', 0.01, 0.09, valinit=0.035, valfmt="%.4f")
slider_k = Slider(ax_k, 'Kill (k)', 0.04, 0.07, valinit=0.060, valfmt="%.4f")

# Explicitly turn off the visibility of the native built-in text labels
slider_F.valtext.set_visible(False)
slider_k.valtext.set_visible(False)

# TextBoxes act as the exclusive real-time readout indicator
text_F = TextBox(ax_box_f, '', initial=f"{slider_F.val:.4f}")
text_k = TextBox(ax_box_k, '', initial=f"{slider_k.val:.4f}")

btn_inject = Button(ax_btn_inject, 'Inject V')
btn_reset  = Button(ax_btn_reset, 'Reset System')

# Callback Actions
def inject_callback(event):
    global U, V
    U, V = inject_activator(U, V)
btn_inject.on_clicked(inject_callback)

def reset_callback(event):
    global U, V
    U, V = initialize_grid(N=N)
    U, V = inject_activator(U, V)
    slider_F.set_val(0.035)
    slider_k.set_val(0.060)
btn_reset.on_clicked(reset_callback)

def submit_F(text_value):
    try:
        val = float(text_value)
        val = np.clip(val, 0.01, 0.09)
        slider_F.set_val(val)
    except ValueError:
        text_F.set_val(f"{slider_F.val:.4f}")

def submit_k(text_value):
    try:
        val = float(text_value)
        val = np.clip(val, 0.04, 0.07)
        slider_k.set_val(val)
    except ValueError:
        text_k.set_val(f"{slider_k.val:.4f}")

text_F.on_submit(submit_F)
text_k.on_submit(submit_k)

# Synchronization wrapper: updates text boxes when sliders are dragged
def update_readouts(val):
    text_F.set_val(f"{slider_F.val:.4f}")
    text_k.set_val(f"{slider_k.val:.4f}")
slider_F.on_changed(update_readouts)
slider_k.on_changed(update_readouts)

# Runtime Execution Loop
def update_frame(frame_number):
    global U, V
    current_F = slider_F.val
    current_k = slider_k.val
    
    steps_per_frame = 30
    for _ in range(steps_per_frame):
        U, V = run_simulation_step(U, V, F=current_F, k=current_k)
        
    im.set_array(V)
    ax.set_title(f"Gray-Scott Sandbox | F = {current_F:.4f} | k = {current_k:.4f}")
    return [im]

ani = animation.FuncAnimation(
    fig, 
    update_frame, 
    interval=1, 
    blit=True, 
    cache_frame_data=False
)

plt.show()
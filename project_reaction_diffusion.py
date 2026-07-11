import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons

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
# SYSTEM PRESETS DATA STORE
# =====================================================================
PRESETS = {
    "Default Baseline":           (0.035, 0.060),
    "Spot Mitosis":               (0.026, 0.053),
    "Chaos / Waves":              (0.014, 0.047),
    "Stable Labyrinths":          (0.055, 0.062),
    "Solitons / Moving Spots":    (0.030, 0.062),
    "U-Skate World":              (0.062, 0.061)
}

# =====================================================================
# USER INTERFACE SETUP
# =====================================================================

N = 100
U, V = initialize_grid(N=N)
U, V = inject_activator(U, V)

fig, ax = plt.subplots(figsize=(10, 8)) # Widened to provide side panel room
plt.subplots_adjust(bottom=0.25, right=0.72) # Leaves clean margins right and bottom

im = ax.imshow(V, cmap='inferno', animated=True, vmin=0.0, vmax=0.5)
ax.axis('off')

# UI Axes Layout Mapping
ax_radio      = plt.axes([0.76, 0.35, 0.20, 0.30]) # Dedicated right-hand panel

ax_f          = plt.axes([0.15, 0.16, 0.45, 0.03])
ax_box_f      = plt.axes([0.65, 0.16, 0.08, 0.03])

ax_k          = plt.axes([0.15, 0.10, 0.45, 0.03])
ax_box_k      = plt.axes([0.65, 0.10, 0.08, 0.03])

ax_btn_inject = plt.axes([0.76, 0.15, 0.18, 0.04])
ax_btn_reset  = plt.axes([0.76, 0.10, 0.18, 0.04])

# Widget Configurations
radio_menu = RadioButtons(ax_radio, list(PRESETS.keys()), active=0)

# Configure sliders with standard formatting to avoid internal crashes
slider_F = Slider(ax_f, 'Feed (F)', 0.01, 0.09, valinit=0.035, valfmt="%.4f")
slider_k = Slider(ax_k, 'Kill (k)', 0.04, 0.07, valinit=0.060, valfmt="%.4f")

# Explicitly hide the native built-in text labels to leave room for the text box readouts
slider_F.valtext.set_visible(False)
slider_k.valtext.set_visible(False)

text_F = TextBox(ax_box_f, '', initial=f"{slider_F.val:.4f}")
text_k = TextBox(ax_box_k, '', initial=f"{slider_k.val:.4f}")

btn_inject = Button(ax_btn_inject, 'Inject V')
btn_reset  = Button(ax_btn_reset, 'Reset System')

# Interactive Callbacks
def preset_callback(selected_label):
    global U, V
    new_F, new_k = PRESETS[selected_label]
    slider_F.set_val(new_F)
    slider_k.set_val(new_k)
    text_F.set_val(f"{new_F:.4f}")
    text_k.set_val(f"{new_k:.4f}")
    U, V = initialize_grid(N=N)
    U, V = inject_activator(U, V)

radio_menu.on_clicked(preset_callback)

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
    # Safely reset the radio panel display element
    radio_menu.set_active(0)
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

def update_readouts(val):
    text_F.set_val(f"{slider_F.val:.4f}")
    text_k.set_val(f"{slider_k.val:.4f}")
slider_F.on_changed(update_readouts)
slider_k.on_changed(update_readouts)

# Runtime Evaluation Step Link
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
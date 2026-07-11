import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons

"""
Gruppe: Modeling Biological Pattern Formation via Gray-Scott Reaction-Diffusion Systems

Richard Schroeder: rschroeder@stud.hs-bremen.de
Khang Pham: kpham@stud.hs-bremen.de
Finn Rusche: frusche@stud.hs-bremen.de
Gian Isikli: gisikli@stud.hs-bremen.de
"""

def verify_stability_criterion(dt, dx, Du, Dv):
    max_diffusion = max(Du, Dv)
    stability_limit = (dx**2) / (4 * max_diffusion)

    print("--- SOLVER STABILITY ANALYSIS ---")
    print(f"Max Diffusion Coefficient: {max_diffusion}")
    print(f"Mathematical Stability Limit for dt: <= {stability_limit:.4f}")
    print(f"Your Selected dt: {dt}")

    if dt <= stability_limit:
        print("RESULT: [PASSED] The system parameters are numerically stable.\n")
        return True
    else:
        print("RESULT: [CRITICAL FAILURE] The current grid parameters risk numerical explosion!")
        print(f"Please lower your dt or adjust spatial dimensions.\n")
        return False

def print_solver_metrics(frame, U, V, V_prev):
    mean_u = np.mean(U)
    mean_v = np.mean(V)

    dv_dt = np.mean(np.abs(V - V_prev))
    
    l2_norm = np.linalg.norm(V) / np.sqrt(V.size)
    
    print(
        f"[Frame {frame:05d}]  "
        f"Mean U: {mean_u:.5f} | "
        f"Mean V: {mean_v:.5f} | "
        f"dV/dt: {dv_dt:.6f} | "
        f"L2-Norm: {l2_norm:.4f}"
    )

def initialize_grid(N=100):
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)
    return U, V

def inject_chaotic_noise(U, V):
    global V_prev
    N = U.shape[0]
    mid = N // 2
    
    U.fill(1.0)
    V.fill(0.0)
    
    U[mid-4:mid+4, mid-4:mid+4] = 0.50
    V[mid-4:mid+4, mid-4:mid+4] = 0.25
    
    np.random.seed(42) 
    for _ in range(15):
        rx = np.random.randint(10, N - 10)
        ry = np.random.randint(10, N - 10)
        U[rx-2:rx+2, ry-2:ry+2] = 0.50
        V[rx-2:rx+2, ry-2:ry+2] = 0.25
        
    V_prev = V.copy()
    return U, V

def inject_center_only(U, V):
    global V_prev
    N = U.shape[0]
    mid = N // 2
    
    U.fill(1.0)
    V.fill(0.0)
    
    U[mid-5:mid+5, mid-5:mid+5] = 0.50
    V[mid-5:mid+5, mid-5:mid+5] = 0.25
    
    V_prev = V.copy()
    return U, V

def calculate_laplacian(A):
    neighbor_right = np.roll(A, shift=-1, axis=1)
    neighbor_left  = np.roll(A, shift=1,  axis=1)
    neighbor_down  = np.roll(A, shift=-1, axis=0)
    neighbor_up    = np.roll(A, shift=1,  axis=0)
    return neighbor_right + neighbor_left + neighbor_down + neighbor_up - 4.0 * A

def run_simulation_step(U, V, F, k, Du=0.16, Dv=0.08, dt=1.0):
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
# SYSTEM CONFIGURATION
# =====================================================================
N = 100
dx = 1.0
dt = 1.0
Du_val = 0.16
Dv_val = 0.08

if not verify_stability_criterion(dt, dx, Du_val, Dv_val):
    print("Warning: Stability criteria violated. Proceeding with caution...")

U, V = initialize_grid(N=N)
V_prev = V.copy()
U, V = inject_chaotic_noise(U, V)

# =====================================================================
# SYSTEM PRESETS DATA STORE
# =====================================================================
PRESETS = {
    "Finger Prints":              (0.032, 0.059),
    "Labyrinth":                  (0.055, 0.062),
    "Moving Spots":               (0.030, 0.062),
    "Moving Waves":               (0.010, 0.047),
    "Random Movement":            (0.0460, 0.065)
}

# =====================================================================
# USER INTERFACE SETUP
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, right=0.72)

im = ax.imshow(V, cmap='inferno', animated=True, vmin=0.0, vmax=0.5) # maybe different cmap
ax.axis('off')

ax_radio          = plt.axes([0.76, 0.40, 0.20, 0.30])
ax_f              = plt.axes([0.15, 0.16, 0.45, 0.03])
ax_box_f          = plt.axes([0.65, 0.16, 0.08, 0.03])
ax_k              = plt.axes([0.15, 0.10, 0.45, 0.03])
ax_box_k          = plt.axes([0.65, 0.10, 0.08, 0.03])

ax_btn_center     = plt.axes([0.76, 0.20, 0.18, 0.04])
ax_btn_noise      = plt.axes([0.76, 0.15, 0.18, 0.04])
ax_btn_reset      = plt.axes([0.76, 0.10, 0.18, 0.04])

radio_menu = RadioButtons(ax_radio, list(PRESETS.keys()), active=0)

slider_F = Slider(ax_f, 'Feed (F)', 0.01, 0.09, valinit=0.032, valfmt="%.4f")
slider_k = Slider(ax_k, 'Kill (k)', 0.04, 0.07, valinit=0.059, valfmt="%.4f")
slider_F.valtext.set_visible(False)
slider_k.valtext.set_visible(False)

text_F = TextBox(ax_box_f, '', initial=f"{slider_F.val:.4f}")
text_k = TextBox(ax_box_k, '', initial=f"{slider_k.val:.4f}")

btn_center = Button(ax_btn_center, 'Inject Center Only')
btn_noise  = Button(ax_btn_noise, 'Add Noise Seeds')
btn_reset  = Button(ax_btn_reset, 'Reset System')

# =====================================================================
# INTERACTIVE CALL BACK CALLBACKS
# =====================================================================
def preset_callback(selected_label):
    global U, V
    new_F, new_k = PRESETS[selected_label]
    slider_F.set_val(new_F)
    slider_k.set_val(new_k)
    text_F.set_val(f"{new_F:.4f}")
    text_k.set_val(f"{new_k:.4f}")
    U, V = initialize_grid(N=N)
    U, V = inject_chaotic_noise(U, V)

radio_menu.on_clicked(preset_callback)

def center_callback(event):
    global U, V
    U, V = inject_center_only(U, V)
btn_center.on_clicked(center_callback)

def noise_callback(event):
    global U, V
    U, V = inject_chaotic_noise(U, V)
btn_noise.on_clicked(noise_callback)

def reset_callback(event):
    global U, V
    U, V = initialize_grid(N=N)
    U, V = inject_chaotic_noise(U, V)
    slider_F.set_val(0.032)
    slider_k.set_val(0.059)
    text_F.set_val("0.0320")
    text_k.set_val("0.0590")
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

# =====================================================================
# RENDER LOOP AND TERM METRICS LOGGING
# =====================================================================
print("--- STARTING GRAY-SCOTT SIMULATION ENGINE ---")

def update_frame(frame_number):
    global U, V, V_prev
    current_F = slider_F.val
    current_k = slider_k.val
    
    steps_per_frame = 30
    for _ in range(steps_per_frame):
        U, V = run_simulation_step(U, V, F=current_F, k=current_k, Du=Du_val, Dv=Dv_val, dt=dt)
        
    if frame_number % 10 == 0:
        actual_time_step = frame_number * steps_per_frame
        print_solver_metrics(actual_time_step, U, V, V_prev)
        V_prev = V.copy()
        
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
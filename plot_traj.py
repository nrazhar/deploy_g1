import numpy as np
import matplotlib.pyplot as plt
import argparse

# --- Constants ---

JOINT_LIMITS = {
    6:  (-2.5307, 2.8798),   # right_hip_pitch
    7:  (-2.9671, 0.5236),   # right_hip_roll
    8:  (-2.7576, 2.7576),   # right_hip_yaw
    9:  (-0.087267, 2.8798), # right_knee
    10: (-0.87267, 0.5236),  # right_ankle_pitch
    11: (-0.2618, 0.2618),   # right_ankle_roll
}

JOINT_MAP = {
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
    "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
}

JOINT_CONFIG = {
    "right_hip_yaw":   (1.0, -1.0,  0.5),
    "right_hip_roll":  (0.3, -1.0,  -0.5),
    "right_hip_pitch": (1.5, -1.0,  0.0),
    "right_knee":      (0.6, 1.0,  0.8),
    "right_ankle_pitch":(0.40, -1.0,  0.00),
    "right_ankle_roll": (0.15, -1.0,  0.00),
}

def generate_and_plot(min_freq, max_freq, duration, dt=0.002, chirp_limit_safety=0.8):
    # Note: Default dt lowered to 0.002 (500Hz) for smoother derivative calculation if needed
    print(f"Generating Trajectory: {min_freq}-{max_freq}Hz, {duration}s")
    
    # --- Timing Configuration ---
    sample_rate = 1.0 / dt
    hold_duration = 2.0
    friction_loop_duration = 15.0 
    transition_duration = 3.0
    chirp_duration = duration
    
    # NEW: Duration to smoothly return to zero after chirp
    return_duration = 3.0  
    
    # NEW: Duration of the chirp end to apply fading (decay)
    fade_duration = 3.0    

    hold_steps = int(hold_duration * sample_rate)
    friction_steps = int(friction_loop_duration * sample_rate)
    transition_steps = int(transition_duration * sample_rate)
    chirp_steps = int(chirp_duration * sample_rate)
    return_steps = int(return_duration * sample_rate)

    total_steps = hold_steps + friction_steps + transition_steps + chirp_steps + return_steps
    time_arr = np.linspace(0, total_steps * dt, total_steps)

    # --- Generate Base Chirp Signal with Fading ---
    t_chirp = np.linspace(0, chirp_duration, chirp_steps)
    # Standard Chirp Phase
    phase = 2 * np.pi * (min_freq * t_chirp + ((max_freq - min_freq) / (2 * chirp_duration)) * t_chirp**2)
    base_chirp_signal = np.sin(phase)

    # Apply Fade-Out Envelope to the last 'fade_duration' seconds of the chirp
    # This prevents the robot from stopping while moving at high velocity
    fade_steps = int(fade_duration * sample_rate)
    if fade_steps > chirp_steps: fade_steps = chirp_steps # Safety clip
    
    # Create an envelope: [1, 1, 1, ... 1, 0.9, 0.8 ... 0]
    envelope = np.ones(chirp_steps)
    decay_curve = np.linspace(1.0, 0.0, fade_steps)
    # Use a smoothstep curve (3x^2 - 2x^3) for the decay to avoid jerk at the start of decay
    # Re-mapping linear 0..1 to smooth 0..1
    decay_curve_smooth = 3*decay_curve**2 - 2*decay_curve**3 # This is actually 0->1 so we might need to flip
    # Let's stick to cosine decay for simplicity and smoothness
    t_decay = np.linspace(0, np.pi/2, fade_steps)
    decay_curve_cos = np.cos(t_decay) # Goes from 1 to 0
    
    envelope[-fade_steps:] = decay_curve_cos
    
    # Apply envelope
    base_chirp_signal = base_chirp_signal * envelope

    # --- Setup Plotting ---
    num_joints = len(JOINT_CONFIG)
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 3 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes]

    fig.suptitle(f"System ID Trajectory (Smooth Stop)", fontsize=14)

    # --- Simulation Loop ---
    for i, (name, (manual_scale, direction, manual_bias)) in enumerate(JOINT_CONFIG.items()):
        idx = JOINT_MAP.get(name)
        if idx is None: continue

        current_q = 0.0 
        trajectory = np.zeros(total_steps)

        # 1. Hardware Limits
        if idx in JOINT_LIMITS:
            lim_min, lim_max = JOINT_LIMITS[idx]
        else:
            lim_min, lim_max = (-1.0, 1.0)

        limit_center = (lim_min + lim_max) / 2.0
        limit_span = lim_max - lim_min
        
        # 2. Friction Loop Bounds
        friction_safe_scalar = 0.6 
        f_min = limit_center - (limit_span/2.0) * friction_safe_scalar
        f_max = limit_center + (limit_span/2.0) * friction_safe_scalar

        # 3. Chirp Bounds
        chirp_amp = (limit_span / 2.0) * chirp_limit_safety
        chirp_bias = limit_center
        
        c_min = chirp_bias - chirp_amp
        c_max = chirp_bias + chirp_amp

        # --- Phase Generation ---

        # Phase 1: Hold
        trajectory[0:hold_steps] = current_q

        # Phase 2: Friction Loop
        s_align = int(friction_steps * 0.2)
        s_up = int(friction_steps * 0.4)
        s_down = friction_steps - s_align - s_up
        
        # Helper for smooth friction loop (using cosine interp instead of linear)
        # linear is okay for slow friction loops, but cosine is smoother
        traj_align = np.linspace(current_q, f_min, s_align)
        traj_up = np.linspace(f_min, f_max, s_up)
        traj_down = np.linspace(f_max, f_min, s_down)
        friction_part = np.concatenate([traj_align, traj_up, traj_down])
        
        trajectory[hold_steps : hold_steps+friction_steps] = friction_part

        # Phase 3: Transition
        friction_end_q = friction_part[-1]
        
        # Dynamic Chirp Calculation
        chirp_part_raw = base_chirp_signal * direction * chirp_amp + chirp_bias
        chirp_start_q = chirp_part_raw[0]

        transition_part = np.linspace(friction_end_q, chirp_start_q, transition_steps)
        trajectory[hold_steps+friction_steps : hold_steps+friction_steps+transition_steps] = transition_part

        # Phase 4: Chirp (Now Faded)
        trajectory[hold_steps+friction_steps+transition_steps : hold_steps+friction_steps+transition_steps+chirp_steps] = chirp_part_raw

        # --- NEW Phase 5: Smooth Return to Zero ---
        # The chirp ends at 'chirp_bias' (because amplitude faded to 0)
        # We need to go from 'chirp_bias' to 0.0 smoothly
        
        return_start_q = chirp_part_raw[-1] # Should be approx chirp_bias
        return_target_q = 0.0 # Return to Home
        
        # Cosine Interpolation: 0 velocity at start, 0 velocity at end
        # Formula: y(t) = start + (target - start) * (1 - cos(pi * t / T)) / 2
        t_ret = np.linspace(0, 1, return_steps)
        return_part = return_start_q + (return_target_q - return_start_q) * (1 - np.cos(np.pi * t_ret)) / 2.0
        
        trajectory[-return_steps:] = return_part

        # --- Plotting ---
        ax = axes[i]
        
        ax.plot(time_arr, trajectory, 'b-', linewidth=1.5, label='Command')
        
        ax.axhline(lim_min, color='r', linestyle='--', alpha=0.5)
        ax.axhline(lim_max, color='r', linestyle='--', alpha=0.5)
        
        # Mark phases
        c_start = (hold_duration + friction_loop_duration + transition_duration)
        c_end = c_start + chirp_duration
        
        # Shade Chirp Fade
        ax.axvspan(c_end - fade_duration, c_end, color='orange', alpha=0.1, label='Decay')
        # Shade Return
        ax.axvspan(c_end, c_end + return_duration, color='green', alpha=0.1, label='Return')

        ax.set_ylabel(f"{name}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SysID Trajectory with Smooth Stop")
    parser.add_argument("--min-freq", type=float, default=0.01)
    parser.add_argument("--max-freq", type=float, default=3.0) # Increased default for visibility
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--safety", type=float, default=0.6)
    args = parser.parse_args()

    generate_and_plot(args.min_freq, args.max_freq, args.duration, chirp_limit_safety=args.safety)
ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
# ROBOT_SCENE = "/home/azhar/ws/g1_sysID/unitree_mujoco/unitree_robots/g1/scene_23dof.xml" # Robot scene
ROBOT_SCENE = "/home/azhar/ws/g1_sysID/ws/deploy_mujoco/g1_23dof_rev_1_0_SDK.xml" # Robot scene

SIMULATE_DT = 0.002  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

# #Joint order in simulation:
#   Joint 0: left_hip_pitch_joint
#   Joint 1: left_hip_roll_joint
#   Joint 2: left_hip_yaw_joint
#   Joint 3: left_knee_joint
#   Joint 4: left_ankle_pitch_joint
#   Joint 5: left_ankle_roll_joint
#   Joint 6: right_hip_pitch_joint
#   Joint 7: right_hip_roll_joint
#   Joint 8: right_hip_yaw_joint
#   Joint 9: right_knee_joint
#   Joint 10: right_ankle_pitch_joint
#   Joint 11: right_ankle_roll_joint
#   Joint 12: waist_yaw_joint
#   Joint 13: left_shoulder_pitch_joint
#   Joint 14: left_shoulder_roll_joint
#   Joint 15: left_shoulder_yaw_joint
#   Joint 16: left_elbow_joint
#   Joint 17: left_wrist_roll_joint
#   Joint 18: right_shoulder_pitch_joint
#   Joint 19: right_shoulder_roll_joint
#   Joint 20: right_shoulder_yaw_joint
#   Joint 21: right_elbow_joint
#   Joint 22: right_wrist_roll_joint

# -------------------------------------------------------------------------
# Calculated Heuristic Gains
# -------------------------------------------------------------------------

# 1. Inertia-Based Heuristic (Soft/Compliant)
# Formula: Kp = I * w^2, Kd = 2 * I * zeta * w
# Parameters: w = 10.0, zeta = 2.0

KPS_INERTIA = [
    # Left Leg (Hip Pitch, Roll, Yaw | Knee | Ankle Pitch, Roll)
    # Pitch(7520_14), Roll(7520_22), Yaw(7520_14), Knee(7520_22), Ankles(2x5020)
    40.18, 99.10, 40.18, 99.10, 28.50, 28.50,
    
    # Right Leg (Hip Pitch, Roll, Yaw | Knee | Ankle Pitch, Roll)
    40.18, 99.10, 40.18, 99.10, 28.50, 28.50,
    
    # Waist (Yaw)
    # Yaw(7520_14)
    40.18,
    
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist Roll)
    # All 5020
    14.25, 14.25, 14.25, 14.25, 14.25,
    
    # Right Arm (Shoulder P/R/Y, Elbow, Wrist Roll)
    # All 5020
    14.25, 14.25, 14.25, 14.25, 14.25
]

KDS_INERTIA = [
    # Left Leg (Hip Pitch, Roll, Yaw | Knee | Ankle Pitch, Roll)
    # Pitch(7520_14), Roll(7520_22), Yaw(7520_14), Knee(7520_22), Ankles(2x5020)
    2.56, 6.31, 2.56, 6.31, 1.81, 1.81,
    
    # Right Leg (Hip Pitch, Roll, Yaw | Knee | Ankle Pitch, Roll)
    2.56, 6.31, 2.56, 6.31, 1.81, 1.81,
    
    # Waist (Yaw)
    # Yaw(7520_14)
    2.56,
    
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist Roll)
    # All 5020
    0.91, 0.91, 0.91, 0.91, 0.91,
    
    # Right Arm (Shoulder P/R/Y, Elbow, Wrist Roll)
    # All 5020
    0.91, 0.91, 0.91, 0.91, 0.91
]
# 2. Conservative Exploration (Full Range)
# Formula: Kp = tau_max / |q_max - q_min|
# Formula: Kd = Kp / 20
KPS_EXPLORE_CONSERVATIVE = [
    # Left Leg (Hip Pitch, Roll, Yaw, Knee, Ankle Pitch, Roll)
    16.26, 39.82, 15.96, 46.85, 35.81, 95.49,
    # Right Leg
    16.26, 39.82, 15.96, 46.85, 35.81, 95.49,
    # Waist
    16.81,
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist)
    4.34, 6.51, 4.77, 7.96, 6.34,
    # Right Arm
    4.34, 6.51, 4.77, 7.96, 6.34
]

KDS_EXPLORE_CONSERVATIVE = [
    # Left Leg
    0.81, 1.99, 0.80, 2.34, 1.79, 4.77,
    # Right Leg
    0.81, 1.99, 0.80, 2.34, 1.79, 4.77,
    # Waist
    0.84,
    # Left Arm
    0.22, 0.33, 0.24, 0.40, 0.32,
    # Right Arm
    0.22, 0.33, 0.24, 0.40, 0.32
]

# 3. Aggressive Exploration (Half Range / Stiff)
# Formula: Kp = tau_max / (0.5 * |q_max - q_min|)
# Formula: Kd = Kp / 20
KPS_EXPLORE_AGGRESSIVE = [
    # Left Leg (Hip Pitch, Roll, Yaw, Knee, Ankle Pitch, Roll)
    32.53, 79.64, 31.91, 93.69, 71.62, 190.99,
    # Right Leg
    32.53, 79.64, 31.91, 93.69, 71.62, 190.99,
    # Waist
    33.61,
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist)
    8.68, 13.02, 9.55, 15.92, 12.68,
    # Right Arm
    8.68, 13.02, 9.55, 15.92, 12.68
]

KDS_EXPLORE_AGGRESSIVE = [
    # Left Leg
    1.63, 3.98, 1.60, 4.68, 3.58, 9.55,
    # Right Leg
    1.63, 3.98, 1.60, 4.68, 3.58, 9.55,
    # Waist
    1.68,
    # Left Arm
    0.43, 0.65, 0.48, 0.80, 0.63,
    # Right Arm
    0.43, 0.65, 0.48, 0.80, 0.63
]

KPS_STANDUP = [
    # Left Leg (Hip Pitch, Roll, Yaw: 100 | Knee: 150 | Ankle Pitch, Roll: 40)
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
    # Right Leg (Hip Pitch, Roll, Yaw: 100 | Knee: 150 | Ankle Pitch, Roll: 40)
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    # Waist (Yaw)
    200.0,
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist)
    40.0, 40.0, 40.0, 40.0, 40.0,
    # Right Arm (Shoulder P/R/Y, Elbow, Wrist)
    40.0, 40.0, 40.0, 40.0, 40.0
]

KDS_STANDUP = [
    # Left Leg (Hip Pitch, Roll, Yaw: 2 | Knee: 4 | Ankle Pitch, Roll: 2)
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
    # Right Leg (Hip Pitch, Roll, Yaw: 2 | Knee: 4 | Ankle Pitch, Roll: 2)
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    # Waist (Yaw)
    5.0,
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist)
    10.0, 10.0, 10.0, 10.0, 10.0,
    # Right Arm (Shoulder P/R/Y, Elbow, Wrist)
    10.0, 10.0, 10.0, 10.0, 10.0
]

KPS_CUSTOM = [
    # Left Leg (Hip Pitch, Roll, Yaw: 120 | Knee: 150 | Ankle Pitch, Roll: 40)
    120.0, 120.0, 120.0, 150.0, 40.0, 40.0,
    # Right Leg
    120.0, 120.0, 120.0, 150.0, 40.0, 40.0,
    # Waist (Yaw)
    200.0,
    # Left Arm (Retained defaults)
    40.0, 40.0, 40.0, 40.0, 40.0,
    # Right Arm (Retained defaults)
    40.0, 40.0, 40.0, 40.0, 40.0
]

KDS_CUSTOM = [
    # Left Leg (Hip Pitch, Roll, Yaw: 3 | Knee: 4 | Ankle Pitch, Roll: 2)
    3.0, 3.0, 3.0, 4.0, 2.0, 2.0,
    # Right Leg
    3.0, 3.0, 3.0, 4.0, 2.0, 2.0,
    # Waist (Yaw)
    5.0,
    # Left Arm (Retained defaults)
    10.0, 10.0, 10.0, 10.0, 10.0,
    # Right Arm (Retained defaults)
    10.0, 10.0, 10.0, 10.0, 10.0
]

# Default gains (alias to STANDUP for compatibility with gui_deploy.py)
KPS = KPS_STANDUP
KDS = KDS_STANDUP

DEFAULT_ANGLES = [
    # Left Leg
    -0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
    # Right Leg
    -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,
    # Waist (Yaw)
    0.0,
    # Left Arm (Shoulder P/R/Y, Elbow, Wrist Roll)
    0.0, 0.25, 0.0, 0.97, 0.15,
    # Right Arm (Shoulder P/R/Y, Elbow, Wrist Roll)
    0.0, -0.25, 0.0, 0.97, -0.15
]

# -------------------------------------------------------------------------
# GUI Configuration
# -------------------------------------------------------------------------

# Enable/disable GUI (if False, runs headless with MuJoCo viewer only)
USE_GUI = True

# Time window for plots (seconds)
PLOT_TIME_WINDOW = 10.0

# Velocity limits (rad/s)
# Source: G1_RIGHT_LEG_PACE_ACTUATOR_CFG for legs.
VELOCITY_LIMITS = [
    # Left Leg (Hip Pitch, Hip Roll, Hip Yaw, Knee, Ankle Pitch, Ankle Roll)
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    # Right Leg (Hip Pitch, Hip Roll, Hip Yaw, Knee, Ankle Pitch, Ankle Roll)
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    # Waist (Yaw)
    10.0,
    # Left Arm
    15.0, 15.0, 15.0, 20.0, 20.0,
    # Right Arm
    15.0, 15.0, 15.0, 20.0, 20.0
]

# Torque limits (Nm)
# Source: G1_RIGHT_LEG_PACE_ACTUATOR_CFG for legs.
# Mapping: Hip Pitch=88, Hip Roll=139, Hip Yaw=88, Knee=139, Ankles=50
TORQUE_LIMITS = [
    # Left Leg (Hip Pitch, Hip Roll, Hip Yaw, Knee, Ankle Pitch, Ankle Roll)
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    # Right Leg (Hip Pitch, Hip Roll, Hip Yaw, Knee, Ankle Pitch, Ankle Roll)
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    # Waist (Yaw)
    88.0,
    # Left Arm
    25.0, 25.0, 25.0, 25.0, 25.0,
    # Right Arm
    25.0, 25.0, 25.0, 25.0, 25.0
]

POSITION_LIMITS = [
    # --- Left Leg ---
    (-2.5307, 2.8798),    # left_hip_pitch_joint
    (-0.5236, 2.9671),    # left_hip_roll_joint
    (-2.7576, 2.7576),    # left_hip_yaw_joint
    (-0.087267, 2.8798),  # left_knee_joint
    (-0.87267, 0.5236),   # left_ankle_pitch_joint
    (-0.2618, 0.2618),    # left_ankle_roll_joint

    # --- Right Leg ---
    (-2.5307, 2.8798),    # right_hip_pitch_joint
    (-2.9671, 0.5236),    # right_hip_roll_joint (Note: Asymmetric limits vs Left)
    (-2.7576, 2.7576),    # right_hip_yaw_joint
    (-0.087267, 2.8798),  # right_knee_joint
    (-0.87267, 0.5236),   # right_ankle_pitch_joint
    (-0.2618, 0.2618),    # right_ankle_roll_joint

    # --- Waist ---
    (-2.618, 2.618),      # waist_yaw_joint

    # --- Left Arm ---
    (-3.0892, 2.6704),    # left_shoulder_pitch_joint
    (-1.5882, 2.2515),    # left_shoulder_roll_joint
    (-2.618, 2.618),      # left_shoulder_yaw_joint
    (-1.0472, 2.0944),    # left_elbow_joint
    (-1.97222, 1.97222),  # left_wrist_roll_joint

    # --- Right Arm ---
    (-3.0892, 2.6704),    # right_shoulder_pitch_joint
    (-2.2515, 1.5882),    # right_shoulder_roll_joint (Note: Asymmetric limits vs Left)
    (-2.618, 2.618),      # right_shoulder_yaw_joint
    (-1.0472, 2.0944),    # right_elbow_joint
    (-1.97222, 1.97222)   # right_wrist_roll_joint
]


# Defined Constants from the Code
A_5020    = 0.003609725
A_7520_14 = 0.010177520
A_7520_22 = 0.025101925
A_4010    = 0.00425     # (Not used in the standard 5-DoF arm subset)

# Calculated Derived Values
A_FEET    = 2.0 * A_5020 # 0.00721945

# Armature inertia (kg*m^2) - reflected motor inertia
# These are default estimates; actual values depend on gear ratio and rotor inertia
ARMATURES = [
    # Left Leg ----------------------------------------------------------------
    0.01017752,   # 0: left_hip_pitch_joint  (7520_14)
    0.025101925,  # 1: left_hip_roll_joint   (7520_22)
    0.01017752,   # 2: left_hip_yaw_joint    (7520_14)
    0.025101925,  # 3: left_knee_joint       (7520_22)
    0.00721945,   # 4: left_ankle_pitch_joint (2 * 5020)
    0.00721945,   # 5: left_ankle_roll_joint  (2 * 5020)

    # Right Leg ---------------------------------------------------------------
    0.01017752,   # 6: right_hip_pitch_joint  (7520_14)
    0.025101925,  # 7: right_hip_roll_joint   (7520_22)
    0.01017752,   # 8: right_hip_yaw_joint    (7520_14)
    0.025101925,  # 9: right_knee_joint       (7520_22)
    0.00721945,   # 10: right_ankle_pitch_joint (2 * 5020)
    0.00721945,   # 11: right_ankle_roll_joint  (2 * 5020)

    # Waist -------------------------------------------------------------------
    0.01017752,   # 12: waist_yaw_joint       (7520_14)

    # Left Arm ----------------------------------------------------------------
    0.003609725,  # 13: left_shoulder_pitch_joint (5020)
    0.003609725,  # 14: left_shoulder_roll_joint  (5020)
    0.003609725,  # 15: left_shoulder_yaw_joint   (5020)
    0.003609725,  # 16: left_elbow_joint          (5020)
    0.003609725,  # 17: left_wrist_roll_joint     (5020)

    # Right Arm ---------------------------------------------------------------
    0.003609725,  # 18: right_shoulder_pitch_joint (5020)
    0.003609725,  # 19: right_shoulder_roll_joint  (5020)
    0.003609725,  # 20: right_shoulder_yaw_joint   (5020)
    0.003609725,  # 21: right_elbow_joint          (5020)
    0.003609725   # 22: right_wrist_roll_joint     (5020)
]

# Default damping ratios for each joint (ζ)
# ζ=1 critical, ζ<1 underdamped, ζ>1 overdamped
DAMPING_RATIOS = [
    # Left Leg
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    # Right Leg
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    # Waist
    2.0,
    # Left Arm
    2.0, 2.0, 2.0, 2.0, 2.0,
    # Right Arm
    2.0, 2.0, 2.0, 2.0, 2.0
]

# Default natural frequency (Hz) for inertia-based gain calculation
DEFAULT_OMEGA = 10.0
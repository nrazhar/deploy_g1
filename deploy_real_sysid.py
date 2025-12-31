# deploy_joint_only_control.py
# Direct joint control deployment without RL policies
# Uses GUI sliders to control joint positions directly on a base pose
# Includes complete safety system from hybrid deployment
# Supports gravity compensation using MuJoCo physics model

import time
import numpy as np
import signal
import threading
import argparse
import sys
import os
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R
from loguru import logger

# --- MuJoCo for gravity compensation ---
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.warning("MuJoCo not available - gravity compensation disabled")

# --- Unitree SDK Imports ---
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
    from common.command_helper import init_cmd_hg, MotorMode
except ImportError as e:
    logger.error(f"Unitree SDK import failed: {e}. Please ensure the SDK is installed and sourced correctly.")
    exit()

try:
    from limxsdk.robot.Rate import Rate as LimxRate
except Exception:
    LimxRate = None

# Import gravity computation from utils
try:
    from utils import compute_qfrc_bias
    import config
    GRAVITY_COMP_AVAILABLE = MUJOCO_AVAILABLE
except ImportError as e:
    logger.warning(f"Could not import gravity compensation utils: {e}")
    GRAVITY_COMP_AVAILABLE = False

# Extracted from g1_23dof_rev_1_0.xml
# MuJoCo joint indices (0-22) - used for trajectory generation
JOINT_LIMITS = {
    # --- Left Leg ---
    0:  (-2.5307, 2.8798),   # left_hip_pitch_joint
    1:  (-0.5236, 2.9671),   # left_hip_roll_joint
    2:  (-2.7576, 2.7576),   # left_hip_yaw_joint
    3:  (-0.087267, 2.8798), # left_knee_joint
    4:  (-0.87267, 0.5236),  # left_ankle_pitch_joint
    5:  (-0.2618, 0.2618),   # left_ankle_roll_joint

    # --- Right Leg ---
    6:  (-2.5307, 2.8798),   # right_hip_pitch_joint
    7:  (-2.9671, 0.5236),   # right_hip_roll_joint
    8:  (-2.7576, 2.7576),   # right_hip_yaw_joint
    9:  (-0.087267, 2.8798), # right_knee_joint
    10: (-0.87267, 0.5236),  # right_ankle_pitch_joint
    11: (-0.2618, 0.2618),   # right_ankle_roll_joint

    # --- Waist ---
    12: (-2.618, 2.618),     # waist_yaw_joint

    # --- Left Arm ---
    13: (-3.0892, 2.6704),   # left_shoulder_pitch_joint
    14: (-1.5882, 2.2515),   # left_shoulder_roll_joint
    15: (-2.618, 2.618),     # left_shoulder_yaw_joint
    16: (-1.0472, 2.0944),   # left_elbow_joint
    17: (-1.97222, 1.97222), # left_wrist_roll_joint

    # --- Right Arm ---
    18: (-3.0892, 2.6704),   # right_shoulder_pitch_joint
    19: (-2.2515, 1.5882),   # right_shoulder_roll_joint
    20: (-2.618, 2.618),     # right_shoulder_yaw_joint
    21: (-1.0472, 2.0944),   # right_elbow_joint
    22: (-1.97222, 1.97222), # right_wrist_roll_joint
}

# Motor index limits (0-28) - maps real robot motor indices to position limits
# Motor indices differ from MuJoCo indices for arms (motors 13,14,20,21,27,28 are dummy)
MOTOR_JOINT_LIMITS = {
    # --- Left Leg (motor 0-5 = MuJoCo 0-5) ---
    0:  (-2.5307, 2.8798),   # left_hip_pitch
    1:  (-0.5236, 2.9671),   # left_hip_roll
    2:  (-2.7576, 2.7576),   # left_hip_yaw
    3:  (-0.087267, 2.8798), # left_knee
    4:  (-0.87267, 0.5236),  # left_ankle_pitch
    5:  (-0.2618, 0.2618),   # left_ankle_roll

    # --- Right Leg (motor 6-11 = MuJoCo 6-11) ---
    6:  (-2.5307, 2.8798),   # right_hip_pitch
    7:  (-2.9671, 0.5236),   # right_hip_roll
    8:  (-2.7576, 2.7576),   # right_hip_yaw
    9:  (-0.087267, 2.8798), # right_knee
    10: (-0.87267, 0.5236),  # right_ankle_pitch
    11: (-0.2618, 0.2618),   # right_ankle_roll

    # --- Waist (motor 12 = MuJoCo 12) ---
    12: (-2.618, 2.618),     # waist_yaw
    # 13, 14: dummy (waist_roll, waist_pitch) - no limits

    # --- Left Arm (motors 15-19 = MuJoCo 13-17) ---
    15: (-3.0892, 2.6704),   # left_shoulder_pitch
    16: (-1.5882, 2.2515),   # left_shoulder_roll
    17: (-2.618, 2.618),     # left_shoulder_yaw
    18: (-1.0472, 2.0944),   # left_elbow
    19: (-1.97222, 1.97222), # left_wrist_roll
    # 20, 21: dummy (left_wrist_pitch, left_wrist_yaw) - no limits

    # --- Right Arm (motors 22-26 = MuJoCo 18-22) ---
    22: (-3.0892, 2.6704),   # right_shoulder_pitch
    23: (-2.2515, 1.5882),   # right_shoulder_roll
    24: (-2.618, 2.618),     # right_shoulder_yaw
    25: (-1.0472, 2.0944),   # right_elbow
    26: (-1.97222, 1.97222), # right_wrist_roll
    # 27, 28: dummy (right_wrist_pitch, right_wrist_yaw) - no limits
}

def get_motor_joint_limits(motor_idx: int, safety_margin: float = 0.05):
    """
    Get joint position limits for a motor index with safety margin.
    
    Args:
        motor_idx: Motor index (0-28)
        safety_margin: Fraction of range to use as buffer from limits (default 5%)
    
    Returns:
        (min_pos, max_pos) tuple, or None if motor has no limits (dummy joint)
    """
    if motor_idx not in MOTOR_JOINT_LIMITS:
        return None  # Dummy joint or invalid index
    
    lim_min, lim_max = MOTOR_JOINT_LIMITS[motor_idx]
    
    # Apply safety margin (shrink the range by margin on each side)
    range_span = lim_max - lim_min
    margin = range_span * safety_margin
    safe_min = lim_min + margin
    safe_max = lim_max - margin
    
    return (safe_min, safe_max)


# ==================== PD GAIN SETS ====================
# Gains are defined in MuJoCo order (23 joints), then expanded to motor order (29)
# MuJoCo: [L_leg(6), R_leg(6), waist(1), L_arm(5), R_arm(5)] = 23
# Motor:  [L_leg(6), R_leg(6), waist(1), dummy(2), L_arm(5), dummy(2), R_arm(5), dummy(2)] = 29

def _expand_gains_to_motor(gains_23: list) -> list:
    """Expand 23-element MuJoCo gains to 29-element motor gains (insert 0 for dummies)"""
    # gains_23 order: [0-5: L_leg, 6-11: R_leg, 12: waist, 13-17: L_arm, 18-22: R_arm]
    # motor order: [0-5: L_leg, 6-11: R_leg, 12: waist, 13-14: dummy, 15-19: L_arm, 
    #               20-21: dummy, 22-26: R_arm, 27-28: dummy]
    return [
        *gains_23[0:6],    # Left leg (motors 0-5)
        *gains_23[6:12],   # Right leg (motors 6-11)
        gains_23[12],      # Waist yaw (motor 12)
        0.0, 0.0,          # Dummy waist roll/pitch (motors 13-14)
        *gains_23[13:18],  # Left arm (motors 15-19)
        0.0, 0.0,          # Dummy left wrist pitch/yaw (motors 20-21)
        *gains_23[18:23],  # Right arm (motors 22-26)
        0.0, 0.0,          # Dummy right wrist pitch/yaw (motors 27-28)
    ]

# Conservative Exploration: Kp = tau_max / |q_max - q_min|, Kd = Kp / 20
_KPS_CONSERVATIVE_23 = [
    16.26, 39.82, 15.96, 46.85, 35.81, 95.49,   # Left Leg
    16.26, 39.82, 15.96, 46.85, 35.81, 95.49,   # Right Leg
    16.81,                                       # Waist
    4.34, 6.51, 4.77, 7.96, 6.34,               # Left Arm
    4.34, 6.51, 4.77, 7.96, 6.34,               # Right Arm
]
_KDS_CONSERVATIVE_23 = [
    0.81, 1.99, 0.80, 2.34, 1.79, 4.77,         # Left Leg
    0.81, 1.99, 0.80, 2.34, 1.79, 4.77,         # Right Leg
    0.84,                                        # Waist
    0.22, 0.33, 0.24, 0.40, 0.32,               # Left Arm
    0.22, 0.33, 0.24, 0.40, 0.32,               # Right Arm
]

# Aggressive Exploration: Kp = tau_max / (0.5 * |q_max - q_min|), Kd = Kp / 20
_KPS_AGGRESSIVE_23 = [
    32.53, 79.64, 31.91, 93.69, 71.62, 190.99,  # Left Leg
    32.53, 79.64, 31.91, 93.69, 71.62, 190.99,  # Right Leg
    33.61,                                       # Waist
    8.68, 13.02, 9.55, 15.92, 12.68,            # Left Arm
    8.68, 13.02, 9.55, 15.92, 12.68,            # Right Arm
]
_KDS_AGGRESSIVE_23 = [
    1.63, 3.98, 1.60, 4.68, 3.58, 9.55,         # Left Leg
    1.63, 3.98, 1.60, 4.68, 3.58, 9.55,         # Right Leg
    1.68,                                        # Waist
    0.43, 0.65, 0.48, 0.80, 0.63,               # Left Arm
    0.43, 0.65, 0.48, 0.80, 0.63,               # Right Arm
]

# Standup gains (default from Unitree)
_KPS_STANDUP_23 = [
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,     # Left Leg
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,     # Right Leg
    200.0,                                       # Waist
    40.0, 40.0, 40.0, 40.0, 40.0,               # Left Arm
    40.0, 40.0, 40.0, 40.0, 40.0,               # Right Arm
]
_KDS_STANDUP_23 = [
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,               # Left Leg
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,               # Right Leg
    5.0,                                         # Waist
    10.0, 10.0, 10.0, 10.0, 10.0,               # Left Arm
    10.0, 10.0, 10.0, 10.0, 10.0,               # Right Arm
]

# Custom gains (stiffer hip)
_KPS_CUSTOM_23 = [
    120.0, 120.0, 120.0, 150.0, 40.0, 40.0,     # Left Leg
    120.0, 120.0, 120.0, 150.0, 40.0, 40.0,     # Right Leg
    200.0,                                       # Waist
    40.0, 40.0, 40.0, 40.0, 40.0,               # Left Arm
    40.0, 40.0, 40.0, 40.0, 40.0,               # Right Arm
]
_KDS_CUSTOM_23 = [
    3.0, 3.0, 3.0, 4.0, 2.0, 2.0,               # Left Leg
    3.0, 3.0, 3.0, 4.0, 2.0, 2.0,               # Right Leg
    5.0,                                         # Waist
    10.0, 10.0, 10.0, 10.0, 10.0,               # Left Arm
    10.0, 10.0, 10.0, 10.0, 10.0,               # Right Arm
]

# Inertia-based gains: Kp = I * w^2, Kd = 2 * I * zeta * w
# Actuator types: 7520_14 (hip P/Y, waist), 7520_22 (hip R, knee), 5020 (ankles, arms)
_KPS_INERTIA_23 = [
    40.18, 99.10, 40.18, 99.10, 28.50, 28.50,   # Left Leg (P/R/Y/Knee/Ankles)
    40.18, 99.10, 40.18, 99.10, 28.50, 28.50,   # Right Leg
    40.18,                                       # Waist (7520_14)
    14.25, 14.25, 14.25, 14.25, 14.25,          # Left Arm (all 5020)
    14.25, 14.25, 14.25, 14.25, 14.25,          # Right Arm (all 5020)
]
_KDS_INERTIA_23 = [
    2.56, 6.31, 2.56, 6.31, 1.81, 1.81,         # Left Leg
    2.56, 6.31, 2.56, 6.31, 1.81, 1.81,         # Right Leg
    2.56,                                        # Waist
    0.91, 0.91, 0.91, 0.91, 0.91,               # Left Arm
    0.91, 0.91, 0.91, 0.91, 0.91,               # Right Arm
]

# Gain sets dictionary (name -> (kp_29, kd_29))
GAIN_SETS = {
    "conservative": (_expand_gains_to_motor(_KPS_CONSERVATIVE_23), 
                     _expand_gains_to_motor(_KDS_CONSERVATIVE_23)),
    "aggressive":   (_expand_gains_to_motor(_KPS_AGGRESSIVE_23), 
                     _expand_gains_to_motor(_KDS_AGGRESSIVE_23)),
    "standup":      (_expand_gains_to_motor(_KPS_STANDUP_23), 
                     _expand_gains_to_motor(_KDS_STANDUP_23)),
    "custom":       (_expand_gains_to_motor(_KPS_CUSTOM_23), 
                     _expand_gains_to_motor(_KDS_CUSTOM_23)),
    "inertia":      (_expand_gains_to_motor(_KPS_INERTIA_23), 
                     _expand_gains_to_motor(_KDS_INERTIA_23)),
}
class _LocalRate:
    def __init__(self, hz: float):
        self.period = 1.0 / float(hz)
        self.next_t = time.perf_counter()
    def sleep(self):
        self.next_t += self.period
        delay = self.next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.perf_counter()

# ---------- Complete Safety Functions (Matching C++ FSM) ----------
def bad_orientation(low_state, limit_angle: float = 1.0) -> bool:
    """
    Check if robot orientation is dangerous (matching C++ isaaclab::mdp::bad_orientation)

    Args:
        low_state: Robot low-level state containing IMU data
        limit_angle: Maximum allowed angle from upright position in radians (default: 1.0 rad ≈ 57.3°)

    Returns:
        True if orientation is dangerous (should trigger safety mode), False otherwise
    """
    # Get quaternion from IMU (w, x, y, z format)
    quat_wxyz = np.asarray(low_state.imu_state.quaternion, dtype=np.float32)

    # Convert to scipy format (x, y, z, w)
    qw, qx, qy, qz = quat_wxyz
    quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)

    # Create rotation object
    r = R.from_quat(quat_xyzw)

    # Gravity vector in world frame (pointing down)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Transform gravity to body frame (inverse rotation)
    gravity_body = r.apply(gravity_world, inverse=True)

    # Calculate angle between body up vector and gravity
    # When robot is upright: gravity_body[2] ≈ -1.0
    # When robot is upside down: gravity_body[2] ≈ 1.0
    # When robot is tilted: gravity_body[2] is between -1 and 1

    # Calculate angle from vertical (0 = upright, π = upside down)
    angle_from_vertical = np.abs(np.arccos(-gravity_body[2]))

    # Check if angle exceeds limit
    is_bad_orientation = angle_from_vertical > limit_angle

    if is_bad_orientation:
        logger.warning(f"Bad orientation detected! Angle from vertical: {np.degrees(angle_from_vertical):.1f}° (limit: {np.degrees(limit_angle):.1f}°)")
        logger.warning(f"Gravity in body frame: [{gravity_body[0]:.3f}, {gravity_body[1]:.3f}, {gravity_body[2]:.3f}]")

    return is_bad_orientation

def check_communication_timeout(low_state, last_state_time: float, timeout_threshold: float = 0.1) -> bool:
    """
    Check if communication with robot has timed out (matching C++ lowstate->isTimeout())

    Args:
        low_state: Robot low-level state
        last_state_time: Timestamp of last received state
        timeout_threshold: Timeout threshold in seconds

    Returns:
        True if communication has timed out, False otherwise
    """
    current_time = time.time()
    time_since_last_state = current_time - last_state_time
    return time_since_last_state > timeout_threshold

def check_manual_emergency_stop(low_state) -> bool:
    """
    Check for manual emergency stop (matching C++ joystick.LT.pressed && joystick.B.on_pressed)

    Args:
        low_state: Robot low-level state containing joystick data

    Returns:
        True if manual emergency stop is pressed, False otherwise
    """
    try:
        # Check if L2 (LT) is pressed AND B button is pressed
        # This matches the C++ condition: lowstate->joystick.LT.pressed && lowstate->joystick.B.on_pressed
        joystick = low_state.joystick

        # Check for L2 + B combination (emergency stop)
        l2_pressed = hasattr(joystick, 'LT') and joystick.LT.pressed
        b_pressed = hasattr(joystick, 'B') and joystick.B.on_pressed

        emergency_stop = l2_pressed and b_pressed

        if emergency_stop:
            logger.critical("Manual emergency stop detected! L2 + B pressed.")

        return emergency_stop
    except Exception as e:
        # If joystick data is not available, return False
        return False

# ---------- Helper functions ----------
def create_damping_cmd(low_cmd):
    """Create damping command (matching C++ Passive mode)"""
    for i in range(len(low_cmd.motor_cmd)):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 3.0
        low_cmd.motor_cmd[i].tau = 0.0
    return low_cmd

def create_zero_cmd(low_cmd):
    """Create zero command"""
    for i in range(len(low_cmd.motor_cmd)):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
    return low_cmd

def gravity_projected_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    r = R.from_quat(quat_xyzw)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return r.apply(g_world, inverse=True).astype(np.float32)

# ---------- Gravity Compensation for Real Hardware ----------
class GravityCompensator:
    """
    Computes gravity compensation torques using MuJoCo physics model.
    Handles mapping between real robot motor indices and MuJoCo joint indices.
    """
    
    # Mapping from real robot motor index to MuJoCo joint index
    # Real robot has 29 motors, MuJoCo model has 23 joints
    # Dummy motors (13,14,20,21,27,28) have no corresponding MuJoCo joint
    MOTOR_TO_MUJOCO = {
        # Left leg (direct mapping)
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        # Right leg (direct mapping)
        6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        # Waist
        12: 12,
        # Left arm: motors 15-19 → MuJoCo 13-17
        15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
        # Right arm: motors 22-26 → MuJoCo 18-22
        22: 18, 23: 19, 24: 20, 25: 21, 26: 22,
    }
    
    # Reverse mapping: MuJoCo joint index to motor index
    MUJOCO_TO_MOTOR = {v: k for k, v in MOTOR_TO_MUJOCO.items()}
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize gravity compensator with MuJoCo model.
        
        Args:
            model_path: Path to MuJoCo XML model. If None, uses config.ROBOT_SCENE
        """
        if not GRAVITY_COMP_AVAILABLE:
            raise RuntimeError("MuJoCo or utils not available for gravity compensation")
        
        if model_path is None:
            model_path = config.ROBOT_SCENE
        
        logger.info(f"Loading MuJoCo model for gravity compensation: {model_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.num_mujoco_joints = self.mj_model.nv
        self.num_motors = 29  # Real robot has 29 motor slots
        
        logger.info(f"MuJoCo model loaded: {self.num_mujoco_joints} joints")
        
        # Cache for efficiency
        self._tau_grav = np.zeros(self.num_mujoco_joints)
        self._motor_tau = np.zeros(self.num_motors)
    
    def compute_gravity_torques(self, motor_positions: np.ndarray, 
                                 motor_velocities: Optional[np.ndarray] = None,
                                 include_coriolis: bool = True) -> np.ndarray:
        """
        Compute gravity (and optionally Coriolis) compensation torques for all motors.
        
        Args:
            motor_positions: Array of motor positions (29 elements)
            motor_velocities: Array of motor velocities (29 elements), optional
            include_coriolis: Include Coriolis/centrifugal compensation (default True)
            
        Returns:
            Array of compensation torques for all 29 motors
        """
        # Map motor positions to MuJoCo joint positions
        for motor_idx, mj_idx in self.MOTOR_TO_MUJOCO.items():
            if motor_idx < len(motor_positions):
                self.mj_data.qpos[mj_idx] = motor_positions[motor_idx]
        
        # Set velocities if provided (affects Coriolis computation)
        if motor_velocities is not None and include_coriolis:
            for motor_idx, mj_idx in self.MOTOR_TO_MUJOCO.items():
                if motor_idx < len(motor_velocities):
                    self.mj_data.qvel[mj_idx] = motor_velocities[motor_idx]
        else:
            self.mj_data.qvel[:] = 0
        
        # Forward kinematics to update body positions
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # Compute gravity + optionally Coriolis forces using RNE algorithm
        self._tau_grav = compute_qfrc_bias(
            self.mj_model, self.mj_data,
            include_gravity=True,
            include_coriolis=include_coriolis
        )
        
        # Map MuJoCo torques back to motor torques
        self._motor_tau[:] = 0
        for motor_idx, mj_idx in self.MOTOR_TO_MUJOCO.items():
            self._motor_tau[motor_idx] = self._tau_grav[mj_idx]
        
        return self._motor_tau.copy()
    
    def compute_gravity_torques_for_joints(self, motor_positions: np.ndarray,
                                           active_indices: set,
                                           motor_velocities: Optional[np.ndarray] = None,
                                           include_coriolis: bool = True) -> np.ndarray:
        """
        Compute gravity (and optionally Coriolis) compensation for active joints only.
        
        Args:
            motor_positions: Array of motor positions (29 elements)
            active_indices: Set of motor indices to compute torques for
            motor_velocities: Array of motor velocities (29 elements), optional
            include_coriolis: Include Coriolis/centrifugal compensation
            
        Returns:
            Array of compensation torques (non-active joints = 0)
        """
        # Full computation
        full_tau = self.compute_gravity_torques(
            motor_positions, motor_velocities, include_coriolis
        )
        
        # Zero out non-active joints
        result = np.zeros_like(full_tau)
        for idx in active_indices:
            if idx < len(result):
                result[idx] = full_tau[idx]
        
        return result


# ---------- Safety Action Clipping Functions ----------
def _get_torque_limits() -> list:
    """
    Get torque limits for all motors based on XML/URDF actuatorfrcrange values
    Order matches the motor command indices
    """
    # Torque limits from g1_23dof_rev_1_0_SDK.xml actuatorfrcrange values
    # These correspond to the motor order in the low_cmd.motor_cmd array
    return [
        88,   # left_hip_pitch
        139,  # left_hip_roll
        88,   # left_hip_yaw
        139,  # left_knee
        50,   # left_ankle_pitch
        50,   # left_ankle_roll
        88,   # right_hip_pitch
        139,  # right_hip_roll
        88,   # right_hip_yaw
        139,  # right_knee
        50,   # right_ankle_pitch
        50,   # right_ankle_roll
        88,   # waist_yaw
        0,    # waist_roll (dummy joint)
        0,    # waist_pitch (dummy joint)
        25,   # left_shoulder_pitch
        25,   # left_shoulder_roll
        25,   # left_shoulder_yaw
        25,   # left_elbow
        25,   # left_wrist_roll
        0,    # left_wrist_pitch (dummy joint)
        0,    # left_wrist_yaw (dummy joint)
        25,   # right_shoulder_pitch
        25,   # right_shoulder_roll
        25,   # right_shoulder_yaw
        25,   # right_elbow
        25,   # right_wrist_roll
        0,    # right_wrist_pitch (dummy joint)
        0     # right_wrist_yaw (dummy joint)
    ]

# ================= Direct Joint Control Deployer =================
class G1DirectJointDeployer:
    def __init__(self,
                 device_type: str = "gui",
                 step_dt: float = 0.02,
                 safety_orientation_limit: float = 1.0,
                 safety_communication_timeout: float = 0.1,
                 gravity_comp_enabled: bool = False,
                 coriolis_comp_enabled: bool = False,
                 gravity_comp_scale: float = 1.0,
                 gain_set: str = "standup"):
        # ---- Timing ----
        self.control_dt = step_dt # 1kHz control loop
        self.step_dt = step_dt   # 50Hz updates for joint control
        self.control_hz = 1.0 / self.control_dt
        self.policy_hz = 1.0 / self.step_dt

        # Calculate decimation factor
        self.decimation = int(self.step_dt / self.control_dt)

        # Rate limiter for control loop
        self.rate = LimxRate(self.control_hz) if LimxRate is not None else _LocalRate(self.control_hz)

        # ---- Complete Safety Configuration ----
        self.safety_orientation_limit = safety_orientation_limit
        self.safety_communication_timeout = safety_communication_timeout
        self.safety_mode = False
        self.last_state_time = time.time()

        # FSM State tracking
        self.current_state = "Passive"
        self.state_transition_count = 0

        # ---- Gravity & Coriolis Compensation ----
        self.gravity_comp_enabled = gravity_comp_enabled and GRAVITY_COMP_AVAILABLE
        self.coriolis_comp_enabled = coriolis_comp_enabled
        self.gravity_comp_scale = gravity_comp_scale
        self.gravity_compensator: Optional[GravityCompensator] = None
        
        if self.gravity_comp_enabled:
            try:
                self.gravity_compensator = GravityCompensator()
                logger.success("Gravity compensation initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize gravity compensation: {e}")
                self.gravity_comp_enabled = False
        
        logger.info(f"Control dt: {self.control_dt:.6f}s, Policy dt: {self.step_dt:.4f}s")
        logger.info(f"Control Hz: {self.control_hz:.0f}, Policy Hz: {self.policy_hz:.0f}")
        logger.info(f"Safety orientation limit: {self.safety_orientation_limit:.1f} rad")
        logger.info(f"Gravity compensation: {'ENABLED' if self.gravity_comp_enabled else 'DISABLED'}")
        logger.info(f"Coriolis compensation: {'ENABLED' if self.coriolis_comp_enabled else 'DISABLED'}")

        # ---- Device Configuration ----
        self.device_type = device_type
        self.device_module = self._load_device_module()
        if self.device_module is None:
            raise ImportError(f"Failed to load device module for: {device_type}")

        # ---- Control State ----
        self.joint_control_enabled = False
        self.reset_requested = False

        # ---- Default joint positions (FixStand pose) ----
        self.default_joint_pos = np.array([
            -0.1, 0, 0, 0.3, -0.2, 0,    # left leg
            -0.1, 0, 0, 0.3, -0.2, 0,    # right leg
            0, 0, 0,                      # waist (yaw, roll, pitch)
            0, 0.25, 0, 0.97, 0.15, 0, 0,  # left arm + wrist
            0, -0.25, 0, 0.97, -0.15, 0, 0 # right arm + wrist
        ], dtype=np.float32)

        # ---- Runtime state ----
        self._running = True
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_publisher = None

        # ---- PD gains (selected from GAIN_SETS) ----
        if gain_set not in GAIN_SETS:
            logger.warning(f"Unknown gain set '{gain_set}', using 'standup'")
            gain_set = "standup"
        
        self.gain_set_name = gain_set
        self.kp_fixstand, self.kd_fixstand = GAIN_SETS[gain_set]
        logger.info(f"Using gain set: {gain_set.upper()}")
        logger.info(f"  Kp sample (hip_pitch): {self.kp_fixstand[0]:.2f}")
        logger.info(f"  Kd sample (hip_pitch): {self.kd_fixstand[0]:.2f}")

        # Signal handler
        def _sigint_handler(_sig, _frame):
            logger.warning("\nCtrl+C detected. Shutting down...")
            self._running = False
        signal.signal(signal.SIGINT, _sigint_handler)

    def _load_device_module(self):
        """Load device-specific module"""
        try:
            # These modes don't need a device module
            if self.device_type in ["chirp", "gravity_test", "none"]:
                self.device_imports = None
                return self._create_stub_device_module()
            elif self.device_type == "keyboard":
                import device_keyboard as device_module
            elif self.device_type == "joystick":
                import device_joystick as device_module
            elif self.device_type == "gui":
                import device_gui_joints_only as device_module
            else:
                logger.error(f"Unknown device type: {self.device_type}")
                return None

            # Load device imports
            self.device_imports = device_module.load_device_imports()
            if self.device_imports is None:
                return None

            return device_module
        except ImportError as e:
            logger.error(f"Failed to import device module: {e}")
            return None
    
    def _create_stub_device_module(self):
        """Create a stub device module for chirp/none mode"""
        class StubDeviceModule:
            @staticmethod
            def load_device_imports():
                return {}
            @staticmethod
            def initialize_device(deployer):
                return True
            @staticmethod
            def process_device_input(deployer):
                pass
            @staticmethod
            def start_device_listener(deployer):
                pass
            @staticmethod
            def print_help():
                pass
            @staticmethod
            def shutdown_device(deployer):
                pass
        return StubDeviceModule()

    def _initialize_device(self):
        """Initialize device-specific components"""
        return self.device_module.initialize_device(self)

    def _process_device_input(self):
        """Process device-specific input"""
        self.device_module.process_device_input(self)

    def _start_device_listener(self):
        """Start device-specific listener"""
        self.device_module.start_device_listener(self)

    def print_help(self):
        """Print device-specific help"""
        self.device_module.print_help()

    def _check_all_safety_conditions(self) -> Tuple[bool, str]:
        """
        Check ALL safety conditions exactly like C++ FSM
        """
        # 1. Manual Emergency Stop
        if check_manual_emergency_stop(self.low_state):
            return True, "Manual emergency stop (L2 + B)"

        # 2. Communication Timeout
        if check_communication_timeout(self.low_state, self.last_state_time, self.safety_communication_timeout):
            return True, "Communication timeout"

        # # 3. Orientation Safety
        # if bad_orientation(self.low_state, self.safety_orientation_limit):
        #     return True, "Dangerous orientation"

        return False, ""

    def _enter_safety_mode(self, reason: str):
        """Enter safety mode (damping)"""
        if not self.safety_mode:
            logger.critical("=== ENTERING SAFETY MODE (DAMPING) ===")
            logger.critical(f"Reason: {reason}")
            logger.critical("Robot will be put into passive damping mode for safety.")
            self.safety_mode = True
            self.joint_control_enabled = False

            # Set damping command for safety
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)

            # Log state transition
            self.state_transition_count += 1
            logger.info(f"FSM: Change state from {self.current_state} to Passive")
            self.current_state = "Passive"

    def _exit_safety_mode(self):
        """Exit safety mode"""
        if self.safety_mode:
            logger.info("=== EXITING SAFETY MODE ===")
            logger.info("Returning to normal operation.")
            self.safety_mode = False

    def _initialize(self):
        logger.info("Initializing Unitree DDS SDK...")
        ChannelFactoryInitialize(0, "lo")

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher.Init()
        lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        lowstate_subscriber.Init(self._low_state_callback, 10)
        init_cmd_hg(self.low_cmd, 4, MotorMode.PR)

    def _low_state_callback(self, msg):
        self.low_state = msg
        self.last_state_time = time.time()

    def send_cmd(self, cmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

    def _set_gains_fixstand(self):
        """Set PD gains for FixStand mode"""
        for i in range(len(self.kp_fixstand)):
            if i < len(self.low_cmd.motor_cmd):
                mc = self.low_cmd.motor_cmd[i]
                mc.kp = float(self.kp_fixstand[i])
                mc.kd = float(self.kd_fixstand[i])
                mc.dq = 0.0
                mc.tau = 0.0

    def run(self):
        # Initialize device first
        if not self._initialize_device():
            logger.error("Failed to initialize device. Exiting.")
            return

        self._initialize()

        logger.info("Connecting to robot... Waiting for first state message.")
        while self.low_state.motor_state[0].q == 0.0 and self._running:
            time.sleep(0.1)
        if not self._running:
            return
        logger.success("Robot connected.")

        try:
            self._safe_startup()
            if self._running:
                if self.device_type == "gui":
                    self._run_with_gui()
                else:
                    self._control_loop()
        except Exception as e:
            logger.error(f"An exception occurred in the main loop: {e}")
        finally:
            self.shutdown()

    def _run_with_gui(self):
        """Run with GUI in main thread"""
        import device_gui_joints_only as device_gui_joints_only
        RobotControlGUI = device_gui_joints_only.RobotControlGUI

        self.gui = RobotControlGUI(self)
        control_thread = threading.Thread(target=self._control_loop, daemon=True)
        control_thread.start()

        self.gui.run()

    def _safe_startup(self):
        """Safe startup sequence"""
        logger.info("--- Initiating Safe Startup ---")

        # Step 1: Zero torque
        logger.info("1. Zero torque...")
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        for _ in range(int(3.0 / self.control_dt)):
            self.rate.sleep()

        # Step 2: FixStand mode
        logger.info("2. Entering FixStand mode...")
        self._set_gains_fixstand()
        self.send_cmd(self.low_cmd)

        move_time = 3.0
        steps = int(move_time / self.control_dt)
        initial_q = np.array([self.low_state.motor_state[i].q for i in range(len(self.default_joint_pos))])

        for i in range(steps):
            if not self._running:
                break
            alpha = (i + 1) / steps
            interp_q = initial_q * (1 - alpha) + self.default_joint_pos * alpha
            self._apply_fixstand_action(interp_q)
            self.rate.sleep()

        # Step 3: Hold FixStand position
        logger.info("3. Holding FixStand position...")
        self._apply_fixstand_action(self.default_joint_pos)
        for _ in range(int(3.0 / self.control_dt)):
            self.rate.sleep()

        logger.info("Robot is ready for direct joint control.")
        if self.device_type == "gui":
            logger.info("Use the GUI to enable joint control and adjust joint positions.")
        elif self.device_type == "keyboard":
            logger.info("Press ENTER to enable joint control, H for help, or X for emergency stop.")

    def _safe_startup_sysid(self, active_motor_indices: set):
        """Safe startup sequence for SysID - only activates specified joints, zeros others"""
        logger.info("--- Initiating SysID Safe Startup ---")
        logger.info(f"Active motor indices: {sorted(active_motor_indices)}")
        logger.info(f"Gravity compensation: {'ENABLED' if self.gravity_comp_enabled else 'DISABLED'}")

        # Step 1: Zero torque for ALL joints
        logger.info("1. Zero torque for all joints...")
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        for _ in range(int(3.0 / self.control_dt)):
            self.rate.sleep()

        # Step 2: Set gains ONLY for active joints, keep zero for others
        logger.info("2. Setting gains for active joints only...")
        tau_gravity = self._compute_gravity_compensation(active_motor_indices)
        for i in range(len(self.low_cmd.motor_cmd)):
            mc = self.low_cmd.motor_cmd[i]
            if i in active_motor_indices:
                mc.kp = float(self.kp_fixstand[i])
                mc.kd = float(self.kd_fixstand[i])
                mc.tau = float(tau_gravity[i])
            else:
                mc.kp = 0.0
                mc.kd = 0.0
                mc.tau = 0.0
            mc.q = 0.0
            mc.dq = 0.0
        self.send_cmd(self.low_cmd)

        # Step 3: Ramp ONLY active joints to default positions
        logger.info("3. Ramping active joints to default positions...")
        move_time = 3.0
        steps = int(move_time / self.control_dt)
        initial_q = np.array([self.low_state.motor_state[i].q 
                              for i in range(len(self.default_joint_pos))])

        for step in range(steps):
            if not self._running:
                break
            alpha = (step + 1) / steps
            tau_gravity = self._compute_gravity_compensation(active_motor_indices)
            
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in active_motor_indices:
                    # Interpolate active joints to default position
                    target_q = initial_q[i] * (1 - alpha) + self.default_joint_pos[i] * alpha
                    mc.q = float(target_q)
                    mc.kp = float(self.kp_fixstand[i])
                    mc.kd = float(self.kd_fixstand[i])
                    mc.tau = float(tau_gravity[i])
                else:
                    # Keep non-active joints zeroed
                    mc.q = 0.0
                    mc.kp = 0.0
                    mc.kd = 0.0
                    mc.tau = 0.0
                mc.dq = 0.0
            self.send_cmd(self.low_cmd)
            self.rate.sleep()

        # Step 4: Hold position for active joints
        logger.info("4. Holding position for active joints...")
        for _ in range(int(3.0 / self.control_dt)):
            tau_gravity = self._compute_gravity_compensation(active_motor_indices)
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in active_motor_indices:
                    mc.q = float(self.default_joint_pos[i])
                    mc.kp = float(self.kp_fixstand[i])
                    mc.kd = float(self.kd_fixstand[i])
                    mc.tau = float(tau_gravity[i])
                else:
                    mc.q = 0.0; mc.kp = 0.0; mc.kd = 0.0; mc.tau = 0.0
                mc.dq = 0.0
            self.send_cmd(self.low_cmd)
            self.rate.sleep()

        logger.info("SysID startup complete. Only active joints are controlled.")

    def _smooth_winddown_sysid(self, active_motor_indices: set):
        """Smooth wind-down after SysID - ramps active joints back to default, keeps others zeroed"""
        pause_at_end_time = 3.0  # seconds to hold at last chirp position before ramping
        winddown_time = 4.0  # seconds for ramp
        hold_time = 2.0  # seconds to hold at default after ramp
        
        # Capture current position (last chirp command position)
        current_q = np.array([m.q for m in self.low_state.motor_state], dtype=np.float32)
        
        # Phase 1: Hold at last chirp position to avoid sudden jerk
        logger.info(f"Holding at last chirp position for {pause_at_end_time}s...")
        pause_steps = int(pause_at_end_time / self.control_dt)
        for _ in range(pause_steps):
            if not self._running:
                break
            tau_gravity = self._compute_gravity_compensation(active_motor_indices)
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in active_motor_indices:
                    mc.q = float(current_q[i])
                    mc.kp = float(self.kp_fixstand[i])
                    mc.kd = float(self.kd_fixstand[i])
                    mc.tau = float(tau_gravity[i])
                else:
                    mc.q = 0.0
                    mc.kp = 0.0
                    mc.kd = 0.0
                    mc.tau = 0.0
                mc.dq = 0.0
            self.send_cmd(self.low_cmd)
            self.rate.sleep()
        
        # Phase 2: Smooth ramp from current position to default
        logger.info(f"Ramping active joints back to default over {winddown_time}s...")
        ramp_steps = int(winddown_time / self.control_dt)
        
        for step in range(ramp_steps):
            if not self._running:
                break
            alpha = (step + 1) / ramp_steps
            tau_gravity = self._compute_gravity_compensation(active_motor_indices)
            
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in active_motor_indices:
                    # Interpolate active joints to default position
                    target_q = current_q[i] * (1 - alpha) + self.default_joint_pos[i] * alpha
                    mc.q = float(target_q)
                    mc.kp = float(self.kp_fixstand[i])
                    mc.kd = float(self.kd_fixstand[i])
                    mc.tau = float(tau_gravity[i])
                else:
                    # Keep non-active joints zeroed
                    mc.q = 0.0
                    mc.kp = 0.0
                    mc.kd = 0.0
                    mc.tau = 0.0
                mc.dq = 0.0
            self.send_cmd(self.low_cmd)
            self.rate.sleep()
        
        # Phase 3: Hold at default position briefly
        logger.info(f"Holding at default position for {hold_time}s...")
        for _ in range(int(hold_time / self.control_dt)):
            if not self._running:
                break
            tau_gravity = self._compute_gravity_compensation(active_motor_indices)
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in active_motor_indices:
                    mc.q = float(self.default_joint_pos[i])
                    mc.kp = float(self.kp_fixstand[i])
                    mc.kd = float(self.kd_fixstand[i])
                    mc.tau = float(tau_gravity[i])
                else:
                    mc.q = 0.0; mc.kp = 0.0; mc.kd = 0.0; mc.tau = 0.0
                mc.dq = 0.0
            self.send_cmd(self.low_cmd)
            self.rate.sleep()
        
        logger.info("Wind-down complete.")

    def _control_loop(self):
        """Main control loop"""
        logger.info(f"--- Starting direct joint control loop with {self.device_type} ---")
        tick = 0
        start_wall = time.time()

        try:
            while self._running and (time.time() - start_wall < 300.0):
                if self.low_state.motor_state[0].q == 0.0:
                    time.sleep(0.01)
                    continue

                wall_t = time.time() - start_wall

                # Process device input
                self._process_device_input()

                # Update GUI smooth interpolation if using GUI
                if hasattr(self, 'gui') and self.gui and hasattr(self.gui, 'update_smooth_interpolation'):
                    self.gui.update_smooth_interpolation()

                # === SAFETY CHECKS ===
                should_enter_safety, safety_reason = self._check_all_safety_conditions()

                if should_enter_safety:
                    self._enter_safety_mode(safety_reason)

                # If in safety mode, hold safe position
                if self.safety_mode:
                    self._apply_fixstand_action(self.default_joint_pos)
                    tick += 1
                    self.rate.sleep()
                    continue

                # Check for reset request
                if self.reset_requested or (hasattr(self, 'gui') and self.gui and self.gui.reset_requested):
                    logger.info("Resetting robot to FixStand...")
                    self._reset_to_fixstand()
                    self.reset_requested = False
                    if hasattr(self, 'gui') and self.gui:
                        self.gui.reset_requested = False
                        self.gui.status_label.config(text="Status: Reset Complete")

                # Apply joint commands
                self._apply_joint_control()

                tick += 1
                self.rate.sleep()

        finally:
            self.shutdown()

    def _reset_to_fixstand(self):
        """Reset robot to FixStand pose"""
        logger.info("Resetting to FixStand...")

        self._set_gains_fixstand()
        self.send_cmd(self.low_cmd)

        move_time = 2.0
        steps = int(move_time / self.control_dt)
        initial_q = np.array([self.low_state.motor_state[i].q for i in range(len(self.default_joint_pos))])

        for i in range(steps):
            if not self._running:
                break
            alpha = (i + 1) / steps
            interp_q = initial_q * (1 - alpha) + self.default_joint_pos * alpha
            self._apply_fixstand_action(interp_q)
            self.rate.sleep()

        self._apply_fixstand_action(self.default_joint_pos)
        for _ in range(int(1.0 / self.control_dt)):
            self.rate.sleep()

        logger.info("Reset to FixStand complete")

    def _apply_fixstand_action(self, target_joint_pos: np.ndarray, use_gravity_comp: bool = True):
        """Apply FixStand joint position commands with optional gravity compensation"""
        # Compute gravity compensation if enabled
        tau_gravity = np.zeros(29)
        if use_gravity_comp and self.gravity_comp_enabled:
            tau_gravity = self._compute_gravity_compensation()
        
        for i in range(len(target_joint_pos)):
            if i < len(self.low_cmd.motor_cmd):
                mc = self.low_cmd.motor_cmd[i]
                mc.q = float(target_joint_pos[i])
                mc.tau = float(tau_gravity[i])

        self.send_cmd(self.low_cmd)

    def _apply_joint_control(self):
        """Apply direct joint control with offsets from GUI for all 29 joints"""
        # Start with default joint positions (29 joints total)
        target_joint_pos = self.default_joint_pos.copy()

        # Ensure we have 29 joints (robot has 29 motors total)
        if len(target_joint_pos) != 29:
            target_joint_pos = np.zeros(29, dtype=np.float32)
            logger.warning(f"Default joint pos length mismatch, using zeros for 29 joints")

        # Get joint offsets from GUI if available and joint control is enabled
        if hasattr(self, 'gui') and self.gui and self.gui.joint_control_enabled.get():
            all_joint_offsets = self.gui.get_joint_offsets()

            # GUI returns 23 offsets for controllable joints (12 lower + 11 upper)
            # Map these to the correct motor indices (skipping dummy joints)

            # Lower body joints (indices 0-11) - direct mapping: GUI 0-11 -> Motor 0-11
            lower_body_motor_indices = list(range(12))  # Motors 0-11
            for i in range(min(12, len(all_joint_offsets))):
                motor_idx = lower_body_motor_indices[i]
                target_joint_pos[motor_idx] += all_joint_offsets[i]
                if abs(all_joint_offsets[i]) > 0.01:
                    lower_body_names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
                                      "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
                                      "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll"]
                    joint_name = lower_body_names[i] if i < len(lower_body_names) else f"lower_{i}"
                    logger.info(f"Controlling {joint_name} (motor {motor_idx}): offset {all_joint_offsets[i]:.3f}")

            # Upper body joints: Direct mapping from GUI order to motor indices
            # GUI upper body order: [waist_yaw, left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
            #                        left_elbow, right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
            #                        right_elbow, left_wrist_roll, right_wrist_roll]
            # Motor indices:        [12,       15,                  16,                 17,
            #                        18,       22,                   23,                 24,
            #                        25,       19,                  26]
            upper_body_motor_indices = [12, 15, 16, 17, 18, 22, 23, 24, 25, 19, 26]

            for i in range(11):  # 11 upper body joints
                gui_idx = 12 + i  # GUI indices 12-22 for upper body
                if gui_idx < len(all_joint_offsets) and i < len(upper_body_motor_indices):
                    motor_idx = upper_body_motor_indices[i]
                    target_joint_pos[motor_idx] += all_joint_offsets[gui_idx]
                    if abs(all_joint_offsets[gui_idx]) > 0.01:
                        upper_body_names = ["waist_yaw", "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                                          "left_elbow", "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                                          "right_elbow", "left_wrist_roll", "right_wrist_roll"]
                        joint_name = upper_body_names[i] if i < len(upper_body_names) else f"upper_{i}"
                        logger.info(f"GUI slider {i} ({joint_name}) → Motor {motor_idx}: offset {all_joint_offsets[gui_idx]:.3f}")

        # Ensure dummy joints are set to 0 (motors 13,14,20,21,27,28 don't exist physically)
        dummy_joints = [13, 14, 20, 21, 27, 28]
        for dummy_idx in dummy_joints:
            if dummy_idx < len(target_joint_pos):
                target_joint_pos[dummy_idx] = 0.0

        # Apply safety clipping to all joints
        for i in range(len(target_joint_pos)):
            target_joint_pos[i] = self._apply_safety_clipping(
                target_joint_pos[i],
                self.low_state.motor_state[i].q,
                self.low_state.motor_state[i].dq,
                i
            )

        # Compute gravity compensation torques if enabled
        tau_gravity = self._compute_gravity_compensation()

        # Set commands for all 29 joints
        for i in range(min(len(target_joint_pos), len(self.low_cmd.motor_cmd))):
            mc = self.low_cmd.motor_cmd[i]
            mc.q = float(target_joint_pos[i])
            # Add gravity compensation feedforward torque
            mc.tau = float(tau_gravity[i])

        self.send_cmd(self.low_cmd)

    def set_gravity_comp_enabled(self, enabled: bool):
        """Enable or disable gravity compensation at runtime"""
        if enabled and not GRAVITY_COMP_AVAILABLE:
            logger.warning("Cannot enable gravity compensation - MuJoCo not available")
            return
        
        if enabled and self.gravity_compensator is None:
            try:
                self.gravity_compensator = GravityCompensator()
                logger.success("Gravity compensation initialized")
            except Exception as e:
                logger.error(f"Failed to initialize gravity compensation: {e}")
                return
        
        self.gravity_comp_enabled = enabled
        logger.info(f"Gravity compensation: {'ENABLED' if enabled else 'DISABLED'}")

    def _compute_gravity_compensation(self, active_indices: Optional[set] = None) -> np.ndarray:
        """
        Compute gravity (and optionally Coriolis) compensation torques.
        
        Args:
            active_indices: If provided, only compute for these motor indices
            
        Returns:
            Array of compensation torques (29 elements)
        """
        if not self.gravity_comp_enabled or self.gravity_compensator is None:
            return np.zeros(29)
        
        # Get current motor positions and velocities from low_state
        motor_positions = np.array([m.q for m in self.low_state.motor_state], dtype=np.float32)
        motor_velocities = np.array([m.dq for m in self.low_state.motor_state], dtype=np.float32)
        
        if active_indices is not None:
            tau_grav = self.gravity_compensator.compute_gravity_torques_for_joints(
                motor_positions, active_indices,
                motor_velocities=motor_velocities,
                include_coriolis=self.coriolis_comp_enabled
            )
        else:
            tau_grav = self.gravity_compensator.compute_gravity_torques(
                motor_positions, motor_velocities,
                include_coriolis=self.coriolis_comp_enabled
            )
        
        return tau_grav * self.gravity_comp_scale
    
    def set_coriolis_comp_enabled(self, enabled: bool):
        """Enable or disable Coriolis compensation at runtime"""
        self.coriolis_comp_enabled = enabled
        logger.info(f"Coriolis compensation: {'ENABLED' if enabled else 'DISABLED'}")

    def _apply_safety_clipping(self, target_pos: float, current_pos: float, current_vel: float, motor_idx: int) -> float:
        """
        Apply safety clipping based on joint limits and torque limits.
        
        Enforces:
        1. Joint position limits (from MOTOR_JOINT_LIMITS with safety margin)
        2. Torque-based velocity limits (prevents excessive torque commands)
        """
        clipped_target = target_pos
        
        # === 1. Enforce Joint Position Limits ===
        joint_limits = get_motor_joint_limits(motor_idx, safety_margin=0.05)
        if joint_limits is not None:
            pos_min, pos_max = joint_limits
            if target_pos < pos_min:
                clipped_target = pos_min
                logger.warning(f"Motor {motor_idx}: position {target_pos:.3f} below limit, "
                              f"clamped to {pos_min:.3f}")
            elif target_pos > pos_max:
                clipped_target = pos_max
                logger.warning(f"Motor {motor_idx}: position {target_pos:.3f} above limit, "
                              f"clamped to {pos_max:.3f}")
        
        # === 2. Enforce Torque-Based Limits ===
        kp = self.low_cmd.motor_cmd[motor_idx].kp
        kd = self.low_cmd.motor_cmd[motor_idx].kd

        # Get torque limits
        torque_limits = _get_torque_limits()
        if motor_idx < len(torque_limits):
            max_torque = torque_limits[motor_idx]
        else:
            max_torque = 50.0

        # Soft torque limit factor
        soft_torque_limit = 0.95
        effective_torque_limit = max_torque * soft_torque_limit

        # Compute torque-based safety bounds
        if kp > 0:
            action_min = current_pos + (kd * current_vel - effective_torque_limit) / kp
            action_max = current_pos + (kd * current_vel + effective_torque_limit) / kp

            # Also respect joint limits in torque bounds
            if joint_limits is not None:
                action_min = max(action_min, joint_limits[0])
                action_max = min(action_max, joint_limits[1])

            # Clip target position within safety bounds
            torque_clipped = max(action_min, min(action_max, clipped_target))

            # Log clipping if significant difference
            if abs(torque_clipped - clipped_target) > 0.01:
                logger.debug(f"Motor {motor_idx}: torque-clipped {clipped_target:.3f} -> "
                            f"{torque_clipped:.3f} (bounds: [{action_min:.3f}, {action_max:.3f}])")
            
            clipped_target = torque_clipped

        return clipped_target

    def shutdown(self):
        logger.info("Sending damping command to robot...")
        
        # If in sysid mode, only send damping to active joints
        if hasattr(self, 'sysid_active_motor_indices') and self.sysid_active_motor_indices:
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in self.sysid_active_motor_indices:
                    # Active joint: apply damping
                    mc.q = 0.0
                    mc.dq = 0.0
                    mc.kp = 0.0
                    mc.kd = 3.0
                    mc.tau = 0.0
                else:
                    # Inactive joint: keep zero (no control)
                    mc.q = 0.0
                    mc.dq = 0.0
                    mc.kp = 0.0
                    mc.kd = 0.0
                    mc.tau = 0.0
            self.send_cmd(self.low_cmd)
        else:
            # Normal mode: damping for all joints
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
        
        time.sleep(0.5)
        # Wait for any background save to complete before shutdown
        if hasattr(self, '_save_thread') and self._save_thread.is_alive():
            logger.info("Waiting for data save to complete...")
            self._save_thread.join(timeout=60)  # Wait up to 60s for save
            if self._save_thread.is_alive():
                logger.warning("Save thread still running after timeout!")
                
        if hasattr(self.device_module, 'shutdown_device'):
            self.device_module.shutdown_device(self)

        logger.info("Shutdown complete.")

    # ... [Inside G1DirectJointDeployer class] ...

    def run_sysid_chirp(self, min_freq, max_freq, duration, log_dir="logs"):
        """
        Executes the Warmup -> Ramp -> Chirp trajectory for System Identification.
        Updated: Friction phase now sweeps full safe range of joint limits.
        """
        logger.info(f"--- Starting System ID Chirp (F: {min_freq}-{max_freq}Hz, T: {duration}s) ---")
        
        # 1. Configuration
        joint_map = {
            # Left leg
            "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
            "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
            # Right leg
            "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
            "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
            # Waist
            "waist_yaw": 12,
            # Left arm
            "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
            "left_elbow": 18, "left_wrist_roll": 19,
            # Right arm
            "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
            "right_elbow": 25, "right_wrist_roll": 26,
        }

        # Config: "name": (Scale, Direction, Bias)
        JOINT_CONFIG = {
            "right_hip_yaw":   (1.0, -1.0,  0.5),
            "right_hip_roll":  (0.3, -1.0,  -0.5),
            "right_hip_pitch": (1.2, -1.0,  0.0),
            "right_knee":      (0.8, 1.0,  1.0),
            "right_ankle_pitch":(0.40, -1.0,  0.00),
            "right_ankle_roll": (0.15, -1.0,  0.00),
        }
        
        # Build set of active motor indices
        active_motor_indices = set()
        for name in JOINT_CONFIG.keys():
            if name in joint_map:
                active_motor_indices.add(joint_map[name])
        
        self.sysid_active_motor_indices = active_motor_indices
        logger.info(f"Active joints for chirp: {list(JOINT_CONFIG.keys())}")
        
        # --- PRE-STARTUP COMMUNICATION CHECK ---
        logger.info("=" * 60)
        logger.info("PRE-STARTUP COMMUNICATION CHECK")
        logger.info("=" * 60)
        
        # Wait for valid state data
        logger.info("Waiting for valid robot state data...")
        timeout_start = time.time()
        while self.low_state.motor_state[0].q == 0.0 and self._running:
            if time.time() - timeout_start > 10.0:
                logger.error("Timeout waiting for robot state data!")
                return
            time.sleep(0.1)
        
        # Display current joint states for active joints
        logger.info("Communication OK. Current state of ACTIVE joints:")
        for name in JOINT_CONFIG.keys():
            if name in joint_map:
                idx = joint_map[name]
                q = self.low_state.motor_state[idx].q
                dq = self.low_state.motor_state[idx].dq
                tau = self.low_state.motor_state[idx].tau_est
                logger.info(f"  {name:20s} (idx {idx:2d}): q={q:+7.3f} rad, dq={dq:+7.3f} rad/s, tau={tau:+7.2f} Nm")
        
        # Check communication timing
        time_since_last = time.time() - self.last_state_time
        logger.info(f"Time since last state update: {time_since_last*1000:.1f} ms")
        
        if time_since_last > 0.1:
            logger.warning("State data may be stale! Check connection.")
        else:
            logger.success("Communication timing looks good.")
        
        # Wait for user confirmation before startup
        logger.info("-" * 60)
        logger.info("Review the joint states above.")
        logger.info("Press ENTER to proceed with safe startup sequence...")
        logger.info("(Robot will move to default position for active joints)")
        logger.info("-" * 60)
        
        import threading
        pre_startup_enter = threading.Event()
        def wait_pre_startup():
            input()
            pre_startup_enter.set()
        pre_input_thread = threading.Thread(target=wait_pre_startup, daemon=True)
        pre_input_thread.start()
        
        # Keep checking communication while waiting
        while self._running and not pre_startup_enter.is_set():
            # Periodically verify communication is still active
            time_since_last = time.time() - self.last_state_time
            if time_since_last > self.safety_communication_timeout:
                logger.warning(f"Communication timeout during wait! ({time_since_last*1000:.1f} ms)")
            time.sleep(0.1)
        
        if not self._running:
            logger.info("Aborted before startup.")
            return
        
        logger.info("User confirmed. Proceeding with safe startup...")
        
        # Perform SysID-specific startup
        self._safe_startup_sysid(active_motor_indices)

        # Wait for user input
        logger.info("Safe startup complete. Robot is holding position.")
        logger.info("Press Enter to start the chirp sequence...")
        
        import threading
        enter_pressed = threading.Event()
        def wait_for_enter():
            input()
            enter_pressed.set()
        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()
        
        hold_positions = np.array([m.q for m in self.low_state.motor_state], dtype=np.float32)
        
        while self._running and not enter_pressed.is_set():
            # Compute gravity compensation for active joints during hold
            tau_gravity = self._compute_gravity_compensation(active_motor_indices)
            
            for i in range(len(self.low_cmd.motor_cmd)):
                mc = self.low_cmd.motor_cmd[i]
                if i in active_motor_indices:
                    mc.q = float(hold_positions[i])
                    mc.kp = float(self.kp_fixstand[i])
                    mc.kd = float(self.kd_fixstand[i])
                    mc.tau = float(tau_gravity[i])
                else:
                    mc.q = 0.0; mc.kp = 0.0; mc.kd = 0.0; mc.tau = 0.0
                mc.dq = 0.0
            self.send_cmd(self.low_cmd)
            self.rate.sleep()
        
        logger.info("Starting chirp sequence...")

        sample_rate = self.policy_hz
        
        # --- TIMING CONFIGURATION ---
        hold_duration = 2.0
        friction_loop_duration = 15.0 
        transition_duration = 3.0
        chirp_duration = duration
        fade_duration = 3.0      # Fade-out at end of chirp
        return_duration = 3.0    # Smooth return to default after chirp

        hold_steps = int(hold_duration * sample_rate)
        friction_steps = int(friction_loop_duration * sample_rate)
        transition_steps = int(transition_duration * sample_rate)
        chirp_steps = int(chirp_duration * sample_rate)
        return_steps = int(return_duration * sample_rate)

        total_steps = hold_steps + friction_steps + transition_steps + chirp_steps + return_steps

        # Initialize buffer
        start_pos_actual = np.array([m.q for m in self.low_state.motor_state], dtype=np.float32)
        full_trajectory = np.tile(start_pos_actual, (total_steps, 1))

        # Generate Chirp Signal with Fade-Out Envelope (Phase 4)
        t_chirp = np.linspace(0, chirp_duration, chirp_steps)
        phase = 2 * np.pi * (min_freq * t_chirp + ((max_freq - min_freq) / (2 * chirp_duration)) * t_chirp**2)
        base_chirp_signal = np.sin(phase)
        
        # Apply fade-out envelope to last 'fade_duration' seconds of chirp
        # This prevents stopping while moving at high velocity
        fade_steps = int(fade_duration * sample_rate)
        fade_steps = min(fade_steps, chirp_steps)  # Safety clip
        
        envelope = np.ones(chirp_steps)
        t_decay = np.linspace(0, np.pi/2, fade_steps)
        decay_curve = np.cos(t_decay)  # Smooth cosine decay from 1 to 0
        envelope[-fade_steps:] = decay_curve
        
        chirp_signal = base_chirp_signal * envelope
        logger.info(f"Chirp with {fade_duration}s fade-out + {return_duration}s return to default")

        logger.info("Generating enhanced trajectory with FULL RANGE friction loop...")

        for name, (scale, direction, bias_val) in JOINT_CONFIG.items():
            if name not in joint_map: continue
            idx = joint_map[name]
            
            current_q = start_pos_actual[idx]
            
            # --- 1. CALCULATE LIMITS FOR FRICTION LOOP ---
            # Get Hardware limits from MOTOR_JOINT_LIMITS (motor indices)
            if idx in MOTOR_JOINT_LIMITS:
                lim_min, lim_max = MOTOR_JOINT_LIMITS[idx]
            else:
                logger.warning(f"No joint limits for motor {idx}, using fallback [-1, 1]")
                lim_min, lim_max = (-1.0, 1.0)  # Fallback

            # Calculate physical center and total span
            limit_center = (lim_min + lim_max) / 2.0
            limit_span = lim_max - lim_min

            # Safety Multiplier: 0.90 = Use 90% of the total range (5% buffer on each side)
            safety_scalar = 0.5
            
            # Calculate safe bounds centered in the physical range
            half_span_safe = (limit_span / 2.0) * safety_scalar
            safe_min = limit_center - half_span_safe
            safe_max = limit_center + half_span_safe

            # --- FILL PHASES ---
            
            # Phase 1: HOLD
            full_trajectory[0:hold_steps, idx] = current_q
            
            # Phase 2: FRICTION LOOP (Full Range)
            # We split friction_steps into 3 segments:
            # A. Current -> Safe Min (Align) - 20% of time
            # B. Safe Min -> Safe Max (Sweep Up) - 40% of time
            # C. Safe Max -> Safe Min (Sweep Down) - 40% of time
            
            s_align = int(friction_steps * 0.2)
            s_up = int(friction_steps * 0.4)
            s_down = friction_steps - s_align - s_up # Remainder
            
            traj_align = np.linspace(current_q, safe_min, s_align)
            traj_up = np.linspace(safe_min, safe_max, s_up)
            traj_down = np.linspace(safe_max, safe_min, s_down)
            
            friction_part = np.concatenate([traj_align, traj_up, traj_down])
            
            start_idx = hold_steps
            end_idx = start_idx + friction_steps
            full_trajectory[start_idx:end_idx, idx] = friction_part
            
            # Phase 3: TRANSITION
            # From end of friction loop (safe_min) to start of chirp (bias)
            friction_end_q = friction_part[-1]
            
            # Calculate where chirp starts (with envelope applied)
            chirp_part = chirp_signal * direction * scale + bias_val
            chirp_start_q = chirp_part[0]
            
            transition_part = np.linspace(friction_end_q, chirp_start_q, transition_steps)
            
            start_idx = end_idx
            end_idx = start_idx + transition_steps
            full_trajectory[start_idx:end_idx, idx] = transition_part
            
            # Phase 4: CHIRP (with fade-out envelope)
            start_idx = end_idx
            end_idx = start_idx + chirp_steps
            full_trajectory[start_idx:end_idx, idx] = chirp_part
            
            # Phase 5: SMOOTH RETURN to default position
            # The chirp ends near bias_val due to envelope fade-out
            # Smooth cosine interpolation back to default joint position
            chirp_end_q = chirp_part[-1]
            return_target_q = self.default_joint_pos[idx]
            
            # Cosine interpolation: 0 velocity at start and end
            t_ret = np.linspace(0, 1, return_steps)
            return_part = chirp_end_q + (return_target_q - chirp_end_q) * (1 - np.cos(np.pi * t_ret)) / 2.0
            
            start_idx = end_idx
            full_trajectory[start_idx:, idx] = return_part

        # Zero out inactive joints in trajectory
        for i in range(full_trajectory.shape[1]):
            if i not in active_motor_indices:
                full_trajectory[:, i] = 0.0

        # --- EXECUTION LOOP ---
        logger.info(f"Starting execution. Total steps: {total_steps}")
        logger.info(f"Gravity compensation: {'ENABLED' if self.gravity_comp_enabled else 'DISABLED'}")
        
        log_data = {
            "time": [], "cmd_q": [], "meas_q": [], "meas_dq": [], 
            "meas_tau": [], "cmd_kp": [], "cmd_kd": [], "cmd_tau_ff": []
        }
        
        start_time = time.time()
        
        try:
            for step in range(total_steps):
                if not self._running: break
                
                # Safety Checks
                is_unsafe, reason = self._check_all_safety_conditions()
                if is_unsafe:
                    self._enter_safety_mode(reason)
                    break
                
                target_q_vec = full_trajectory[step]
                
                # Compute gravity compensation for active joints
                tau_gravity = self._compute_gravity_compensation(active_motor_indices)
                
                for i in range(min(len(target_q_vec), len(self.low_cmd.motor_cmd))):
                    mc = self.low_cmd.motor_cmd[i]
                    if i in active_motor_indices:
                        # Apply safety clipping to enforce joint limits
                        current_pos = self.low_state.motor_state[i].q
                        current_vel = self.low_state.motor_state[i].dq
                        safe_target = self._apply_safety_clipping(
                            target_q_vec[i], current_pos, current_vel, i
                        )
                        mc.q = float(safe_target)
                        mc.dq = 0.0
                        mc.kp = float(self.kp_fixstand[i])
                        mc.kd = float(self.kd_fixstand[i])
                        mc.tau = float(tau_gravity[i])  # Add gravity compensation torque
                    else:
                        mc.q = 0.0; mc.dq = 0.0; mc.kp = 0.0; mc.kd = 0.0; mc.tau = 0.0
                
                self.send_cmd(self.low_cmd)
                
                # Logging
                current_t = time.time() - start_time
                log_data["time"].append(current_t)
                log_data["cmd_q"].append(np.array([mc.q for mc in self.low_cmd.motor_cmd]))
                log_data["meas_q"].append(np.array([m.q for m in self.low_state.motor_state]))
                log_data["meas_dq"].append(np.array([m.dq for m in self.low_state.motor_state]))
                log_data["meas_tau"].append(np.array([m.tau_est for m in self.low_state.motor_state]))
                log_data["cmd_kp"].append(np.array([mc.kp for mc in self.low_cmd.motor_cmd]))
                log_data["cmd_kd"].append(np.array([mc.kd for mc in self.low_cmd.motor_cmd]))
                log_data["cmd_tau_ff"].append(np.array([mc.tau for mc in self.low_cmd.motor_cmd]))

                self.rate.sleep()
                
                if step % 100 == 0:
                    chirp_end = hold_steps + friction_steps + transition_steps + chirp_steps
                    if step < hold_steps:
                        phase_name = "Hold"
                    elif step < hold_steps + friction_steps:
                        phase_name = "Friction"
                    elif step < hold_steps + friction_steps + transition_steps:
                        phase_name = "Transition"
                    elif step < chirp_end:
                        # Show fade info in last part of chirp
                        chirp_start = hold_steps + friction_steps + transition_steps
                        chirp_progress = step - chirp_start
                        fade_start = chirp_steps - int(fade_duration * sample_rate)
                        if chirp_progress >= fade_start:
                            phase_name = "Chirp (Fading)"
                        else:
                            phase_name = "Chirp"
                    else:
                        phase_name = "Return"
                    logger.info(f"Progress: {step}/{total_steps} ({phase_name})")

        except Exception as e:
            logger.error(f"Exception during chirp: {e}")
        finally:
            if self._running:
                logger.info("Trajectory complete (includes smooth return). Holding position...")
                # Brief hold at final position before wind-down
                self._smooth_winddown_sysid(active_motor_indices)
            else:
                logger.info("Interrupted - skipping wind-down...")
            
            joint_names = {idx: name for name, idx in joint_map.items()}
            self._save_thread = threading.Thread(
                target=self._save_chirp_data,
                args=(log_data, log_dir, active_motor_indices, joint_names),
                daemon=False
            )
            self._save_thread.start()
            logger.info("Data saving started in background...")

    def _save_chirp_data(self, data, log_dir, active_motor_indices=None, joint_names=None,
                         trim_steps=5):
        """Saves the logged trajectory data directly to .pt format for analysis"""
        import os
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for thread safety
        import matplotlib.pyplot as plt
        import torch
        
        # Create timestamped folder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Saving run data to: {run_dir}")
        
        # Convert to numpy arrays
        time_arr = np.array(data["time"])
        cmd_q = np.array(data["cmd_q"])
        meas_q = np.array(data["meas_q"])
        meas_dq = np.array(data["meas_dq"])
        meas_tau = np.array(data["meas_tau"])
        cmd_kp = np.array(data.get("cmd_kp", []))
        cmd_kd = np.array(data.get("cmd_kd", []))
        cmd_tau_ff = np.array(data.get("cmd_tau_ff", []))
        
        # --- Save in PyTorch .pt format (analysis-ready) ---
        # Define joint names for active joints in the expected order
        # SDK indices: 6=right_hip_pitch, 7=right_hip_roll, 8=right_hip_yaw, 
        #              9=right_knee, 10=right_ankle_pitch, 11=right_ankle_roll
        SDK_TO_NAME = {
            0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_hip_yaw_joint",
            3: "left_knee_joint", 4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
            6: "right_hip_pitch_joint", 7: "right_hip_roll_joint", 8: "right_hip_yaw_joint",
            9: "right_knee_joint", 10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
            12: "waist_yaw_joint",
            15: "left_shoulder_pitch_joint", 16: "left_shoulder_roll_joint", 
            17: "left_shoulder_yaw_joint", 18: "left_elbow_joint", 19: "left_wrist_roll_joint",
            22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint",
            24: "right_shoulder_yaw_joint", 25: "right_elbow_joint", 26: "right_wrist_roll_joint",
        }
        
        # Get sorted active indices and their names
        active_indices_sorted = sorted(active_motor_indices) if active_motor_indices else []
        target_joint_names = [SDK_TO_NAME.get(idx, f"joint_{idx}") for idx in active_indices_sorted]
        
        # Trim initial steps (startup transient)
        total_steps = len(time_arr)
        if total_steps > trim_steps:
            trim_slice = slice(trim_steps, None)
            time_trimmed = time_arr[trim_slice] - time_arr[trim_steps]  # Re-zero time
            cmd_q_trimmed = cmd_q[trim_slice]
            meas_q_trimmed = meas_q[trim_slice]
            meas_dq_trimmed = meas_dq[trim_slice]
            meas_tau_trimmed = meas_tau[trim_slice]
            cmd_kp_trimmed = cmd_kp[trim_slice] if len(cmd_kp) > 0 else cmd_kp
            cmd_kd_trimmed = cmd_kd[trim_slice] if len(cmd_kd) > 0 else cmd_kd
            cmd_tau_ff_trimmed = cmd_tau_ff[trim_slice] if len(cmd_tau_ff) > 0 else cmd_tau_ff
        else:
            time_trimmed = time_arr - time_arr[0]
            cmd_q_trimmed = cmd_q
            meas_q_trimmed = meas_q
            meas_dq_trimmed = meas_dq
            meas_tau_trimmed = meas_tau
            cmd_kp_trimmed = cmd_kp
            cmd_kd_trimmed = cmd_kd
            cmd_tau_ff_trimmed = cmd_tau_ff
        
        # Extract only active joint columns
        def select_joints(arr):
            if len(arr) == 0 or len(active_indices_sorted) == 0:
                return torch.zeros((len(time_trimmed), len(active_indices_sorted)), dtype=torch.float32)
            return torch.tensor(arr[:, active_indices_sorted], dtype=torch.float32)
        
        # Build PyTorch payload (analysis-ready format)
        pt_payload = {
            "time": torch.tensor(time_trimmed, dtype=torch.float32),
            "dof_pos": select_joints(meas_q_trimmed),           # measured positions
            "des_dof_pos": select_joints(cmd_q_trimmed),        # commanded positions
            "dof_vel": select_joints(meas_dq_trimmed),          # measured velocities
            "dof_torque": select_joints(meas_tau_trimmed),      # measured torques
            "joint_names": target_joint_names,
            # Additional data for extended analysis
            "cmd_kp": select_joints(cmd_kp_trimmed) if len(cmd_kp_trimmed) > 0 else None,
            "cmd_kd": select_joints(cmd_kd_trimmed) if len(cmd_kd_trimmed) > 0 else None,
            "cmd_tau_ff": select_joints(cmd_tau_ff_trimmed) if len(cmd_tau_ff_trimmed) > 0 else None,
            "active_indices": active_indices_sorted,
            "sample_rate_hz": self.policy_hz,
        }
        
        # Remove None values
        pt_payload = {k: v for k, v in pt_payload.items() if v is not None}
        
        # Save .pt file
        pt_filename = os.path.join(run_dir, "chirp_data.pt")
        torch.save(pt_payload, pt_filename)
        logger.success(f"PyTorch data saved to {pt_filename}")
        
        # Log payload info
        logger.info("Saved data shapes:")
        for k, v in pt_payload.items():
            if torch.is_tensor(v):
                logger.info(f"  {k}: {list(v.shape)}")
            elif isinstance(v, list):
                logger.info(f"  {k}: {v}")
        
        # Generate plots if we have active joint info
        if active_motor_indices and joint_names and len(time_arr) > 0:
            active_indices = sorted(active_motor_indices)
            n_joints = len(active_indices)
            
            # Plot 1: Position (Commanded vs Measured)
            fig1, axes1 = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints), sharex=True)
            if n_joints == 1:
                axes1 = [axes1]
            fig1.suptitle('Joint Positions: Commanded vs Measured vs Limits', fontsize=14)
            
            for i, idx in enumerate(active_indices):
                name = joint_names.get(idx, f"Joint {idx}")
                axes1[i].plot(time_arr, cmd_q[:, idx], 'b-', label='Commanded', linewidth=1.5)
                axes1[i].plot(time_arr, meas_q[:, idx], 'r--', label='Measured', linewidth=1.5)
                
                # --- Add Joint Limits to plot ---
                if idx in MOTOR_JOINT_LIMITS:
                    low_lim, high_lim = MOTOR_JOINT_LIMITS[idx]
                    # Draw limits as horizontal dashed lines
                    axes1[i].axhline(y=low_lim, color='k', linestyle=':', linewidth=2.0, alpha=0.6, label='Limit')
                    axes1[i].axhline(y=high_lim, color='k', linestyle=':', linewidth=2.0, alpha=0.6)
                    
                    # Optional: Shade the forbidden region if close enough to be visible
                    # Get current plot limits to ensure we don't mess up zoom level too much
                    # y_min, y_max = axes1[i].get_ylim()
                    # buffer = (y_max - y_min) * 0.1
                    # axes1[i].fill_between(time_arr, high_lim, high_lim + 10, color='red', alpha=0.1)
                    # axes1[i].fill_between(time_arr, low_lim - 10, low_lim, color='red', alpha=0.1)
                    # axes1[i].set_ylim(max(y_min, low_lim - buffer), min(y_max, high_lim + buffer))
                # -----------------------------

                axes1[i].set_ylabel(f'{name}\n[rad]')
                # Handle duplicate labels in legend
                handles, labels = axes1[i].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes1[i].legend(by_label.values(), by_label.keys(), loc='upper right')
                axes1[i].grid(True, alpha=0.3)

            axes1[-1].set_xlabel('Time [s]')
            plt.tight_layout()
            pos_plot_file = os.path.join(run_dir, "chirp_position.png")
            fig1.savefig(pos_plot_file, dpi=150)
            plt.close(fig1)
            logger.info(f"Position plot saved to {pos_plot_file}")
            
            # Plot 2: Velocity
            fig2, axes2 = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints), sharex=True)
            if n_joints == 1:
                axes2 = [axes2]
            fig2.suptitle('Joint Velocities', fontsize=14)
            
            for i, idx in enumerate(active_indices):
                name = joint_names.get(idx, f"Joint {idx}")
                axes2[i].plot(time_arr, meas_dq[:, idx], 'g-', linewidth=1.5)
                axes2[i].set_ylabel(f'{name}\n[rad/s]')
                axes2[i].grid(True, alpha=0.3)
            axes2[-1].set_xlabel('Time [s]')
            plt.tight_layout()
            vel_plot_file = os.path.join(run_dir, "chirp_velocity.png")
            fig2.savefig(vel_plot_file, dpi=150)
            plt.close(fig2)
            logger.info(f"Velocity plot saved to {vel_plot_file}")
            
            # Plot 3: Torque
            fig3, axes3 = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints), sharex=True)
            if n_joints == 1:
                axes3 = [axes3]
            fig3.suptitle('Joint Torques', fontsize=14)
            
            for i, idx in enumerate(active_indices):
                name = joint_names.get(idx, f"Joint {idx}")
                axes3[i].plot(time_arr, meas_tau[:, idx], 'm-', linewidth=1.5)
                axes3[i].set_ylabel(f'{name}\n[Nm]')
                axes3[i].grid(True, alpha=0.3)
            axes3[-1].set_xlabel('Time [s]')
            plt.tight_layout()
            tau_plot_file = os.path.join(run_dir, "chirp_torque.png")
            fig3.savefig(tau_plot_file, dpi=150)
            plt.close(fig3)
            logger.info(f"Torque plot saved to {tau_plot_file}")
            
            # Joint name lookup for all joints
            all_joint_names = {
                0: "left_hip_pitch", 1: "left_hip_roll", 2: "left_hip_yaw",
                3: "left_knee", 4: "left_ankle_pitch", 5: "left_ankle_roll",
                6: "right_hip_pitch", 7: "right_hip_roll", 8: "right_hip_yaw",
                9: "right_knee", 10: "right_ankle_pitch", 11: "right_ankle_roll",
                12: "waist_yaw", 15: "left_shoulder_pitch", 16: "left_shoulder_roll",
                17: "left_shoulder_yaw", 18: "left_elbow", 19: "left_wrist_roll",
                22: "right_shoulder_pitch", 23: "right_shoulder_roll", 24: "right_shoulder_yaw",
                25: "right_elbow", 26: "right_wrist_roll"
            }
            
            # Plot 4: Active joints detailed (cmd_q, kp, kd, meas_tau)
            if len(cmd_kp) > 0:
                fig4, axes4 = plt.subplots(n_joints, 4, figsize=(16, 2.5*n_joints), sharex=True)
                if n_joints == 1:
                    axes4 = axes4.reshape(1, -1)
                fig4.suptitle('Active Joints: Commands & Gains', fontsize=14)
                
                for i, idx in enumerate(active_indices):
                    name = joint_names.get(idx, all_joint_names.get(idx, f"Joint {idx}"))
                    # Cmd q
                    axes4[i, 0].plot(time_arr, cmd_q[:, idx], 'b-', linewidth=1)
                    axes4[i, 0].set_ylabel(f'{name}')
                    axes4[i, 0].set_title('cmd_q [rad]' if i == 0 else '')
                    axes4[i, 0].grid(True, alpha=0.3)
                    # Cmd kp
                    axes4[i, 1].plot(time_arr, cmd_kp[:, idx], 'r-', linewidth=1)
                    axes4[i, 1].set_title('kp' if i == 0 else '')
                    axes4[i, 1].grid(True, alpha=0.3)
                    # Cmd kd
                    axes4[i, 2].plot(time_arr, cmd_kd[:, idx], 'g-', linewidth=1)
                    axes4[i, 2].set_title('kd' if i == 0 else '')
                    axes4[i, 2].grid(True, alpha=0.3)
                    # Meas tau
                    axes4[i, 3].plot(time_arr, meas_tau[:, idx], 'm-', linewidth=1)
                    axes4[i, 3].set_title('meas_tau [Nm]' if i == 0 else '')
                    axes4[i, 3].grid(True, alpha=0.3)
                
                for ax in axes4[-1, :]:
                    ax.set_xlabel('Time [s]')
                plt.tight_layout()
                active_detail_file = os.path.join(run_dir, "chirp_active_detail.png")
                fig4.savefig(active_detail_file, dpi=150)
                plt.close(fig4)
                logger.info(f"Active joints detail plot saved to {active_detail_file}")
            
            # Plot 5: ALL Inactive joints verification (q, kp, kd, tau should all be ~0)
            # Get inactive indices (exclude dummy joints 13,14,20,21,27,28)
            dummy_joints = {13, 14, 20, 21, 27, 28}
            all_real_joints = set(range(min(cmd_q.shape[1], 29))) - dummy_joints
            inactive_indices = sorted(all_real_joints - set(active_indices))
            
            if len(inactive_indices) > 0 and len(cmd_kp) > 0:
                n_inactive = len(inactive_indices)
                
                fig5, axes5 = plt.subplots(n_inactive, 4, figsize=(16, 2*n_inactive), sharex=True)
                if n_inactive == 1:
                    axes5 = axes5.reshape(1, -1)
                fig5.suptitle('ALL Inactive Joints Verification (should be ~0)', fontsize=14)
                
                for i, idx in enumerate(inactive_indices):
                    name = all_joint_names.get(idx, f"Joint {idx}")
                    # Cmd q
                    axes5[i, 0].plot(time_arr, cmd_q[:, idx], 'b-', linewidth=1)
                    axes5[i, 0].set_ylabel(f'{name}', fontsize=8)
                    axes5[i, 0].set_title('cmd_q' if i == 0 else '')
                    axes5[i, 0].grid(True, alpha=0.3)
                    axes5[i, 0].tick_params(axis='y', labelsize=7)
                    # Cmd kp
                    axes5[i, 1].plot(time_arr, cmd_kp[:, idx], 'r-', linewidth=1)
                    axes5[i, 1].set_title('kp' if i == 0 else '')
                    axes5[i, 1].grid(True, alpha=0.3)
                    axes5[i, 1].tick_params(axis='y', labelsize=7)
                    # Cmd kd
                    axes5[i, 2].plot(time_arr, cmd_kd[:, idx], 'g-', linewidth=1)
                    axes5[i, 2].set_title('kd' if i == 0 else '')
                    axes5[i, 2].grid(True, alpha=0.3)
                    axes5[i, 2].tick_params(axis='y', labelsize=7)
                    # Meas tau
                    axes5[i, 3].plot(time_arr, meas_tau[:, idx], 'm-', linewidth=1)
                    axes5[i, 3].set_title('meas_tau' if i == 0 else '')
                    axes5[i, 3].grid(True, alpha=0.3)
                    axes5[i, 3].tick_params(axis='y', labelsize=7)
                
                for ax in axes5[-1, :]:
                    ax.set_xlabel('Time [s]')
                plt.tight_layout()
                inactive_plot_file = os.path.join(run_dir, "chirp_inactive_verify.png")
                fig5.savefig(inactive_plot_file, dpi=150)
                plt.close(fig5)
                logger.info(f"Inactive joints verification plot saved to {inactive_plot_file}")

    def run_gravity_test(self, hold_duration: float = 5.0, log_dir: str = "logs"):
        """
        Test gravity compensation by holding static poses and comparing torques.
        
        Trajectory: Move through several static poses, hold each for hold_duration seconds.
        Logs: tau_gravity (computed), tau_pd (computed), tau_measured (from motor)
        """
        logger.info("=" * 60)
        logger.info("GRAVITY COMPENSATION TEST MODE")
        logger.info("=" * 60)
        
        if not self.gravity_comp_enabled:
            logger.warning("Gravity compensation is DISABLED. Enabling for test...")
            self.set_gravity_comp_enabled(True)
        
        # Joint configuration (same as chirp)
        joint_map = {
            "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
            "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
            "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
            "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
            "waist_yaw": 12,
            "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
            "left_elbow": 18, "left_wrist_roll": 19,
            "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
            "right_elbow": 25, "right_wrist_roll": 26,
        }
        
        # Use ALL active joints from chirp configuration
        JOINT_CONFIG = {
            "right_hip_yaw":   (1.0, -1.0,  0.5),
            "right_hip_roll":  (0.3, -1.0,  -0.5),
            "right_hip_pitch": (1.2, -1.0,  0.0),
            "right_knee":      (0.8, 1.0,  1.0),
            "right_ankle_pitch":(0.40, -1.0,  0.00),
            "right_ankle_roll": (0.15, -1.0,  0.00),
        }
        
        TEST_JOINTS = list(JOINT_CONFIG.keys())
        active_motor_indices = set(joint_map[name] for name in TEST_JOINTS)
        self.sysid_active_motor_indices = active_motor_indices
        
        logger.info(f"Test joints (ALL active): {TEST_JOINTS}")
        logger.info(f"Active motor indices: {sorted(active_motor_indices)}")
        
        # Right hip roll constraint: must not exceed -0.2
        RIGHT_HIP_ROLL_MAX = -0.2
        right_hip_roll_idx = joint_map.get("right_hip_roll")
        right_hip_roll_active = right_hip_roll_idx is not None and right_hip_roll_idx in active_motor_indices
        
        # Helper function to clamp right_hip_roll
        def clamp_right_hip_roll(pos_array, idx=right_hip_roll_idx):
            """Clamp right_hip_roll to not exceed RIGHT_HIP_ROLL_MAX"""
            if right_hip_roll_active and idx is not None:
                if isinstance(pos_array, np.ndarray) and pos_array.ndim == 1:
                    pos_array[idx] = min(RIGHT_HIP_ROLL_MAX, pos_array[idx])
                elif isinstance(pos_array, np.ndarray) and pos_array.ndim == 2:
                    pos_array[:, idx] = np.minimum(RIGHT_HIP_ROLL_MAX, pos_array[:, idx])
                else:
                    pos_array[idx] = min(RIGHT_HIP_ROLL_MAX, pos_array[idx])
            return pos_array
        
        # Get safe joint limits for all active joints
        safe_limits = {}
        for joint_name in TEST_JOINTS:
            motor_idx = joint_map[joint_name]
            limits = get_motor_joint_limits(motor_idx, safety_margin=0.10)  # 20% margin (increased from 10%)
            if limits:
                lim_min, lim_max = limits
                # Special constraint: right_hip_roll must stay <= RIGHT_HIP_ROLL_MAX
                if joint_name == "right_hip_roll":
                    lim_max = min(RIGHT_HIP_ROLL_MAX, lim_max)
                    logger.info(f"  {joint_name} (motor {motor_idx}): safe range [{lim_min:.3f}, {lim_max:.3f}] (capped at {RIGHT_HIP_ROLL_MAX})")
                else:
                    logger.info(f"  {joint_name} (motor {motor_idx}): safe range [{lim_min:.3f}, {lim_max:.3f}]")
                safe_limits[joint_name] = (lim_min, lim_max)
            else:
                logger.warning(f"  {joint_name} (motor {motor_idx}): no limits found")
        
        # Generate test poses within safe limits
        # Strategy: Test different combinations of joint positions
        # Each pose uses a fraction of the safe range from default position
        TEST_POSES = []
        
        # Pose 0: Default position
        pose_0 = {name: 0.0 for name in TEST_JOINTS}
        TEST_POSES.append(pose_0)
        
        # Pose 1-6: Single joint variations (move one joint at a time)
        for joint_name in TEST_JOINTS:
            if joint_name in safe_limits:
                lim_min, lim_max = safe_limits[joint_name]
                default_q = self.default_joint_pos[joint_map[joint_name]]
                
                # Special constraint: right_hip_roll must stay <= RIGHT_HIP_ROLL_MAX
                if joint_name == "right_hip_roll":
                    # Only test negative direction (below RIGHT_HIP_ROLL_MAX)
                    range_span = lim_max - lim_min
                    offset_neg = -0.3 * range_span
                    target_neg = default_q + offset_neg
                    # Clamp: ensure <= RIGHT_HIP_ROLL_MAX and within limits
                    target_neg = max(lim_min, min(RIGHT_HIP_ROLL_MAX, target_neg))
                    
                    pose = {name: 0.0 for name in TEST_JOINTS}
                    pose[joint_name] = target_neg - default_q
                    TEST_POSES.append(pose.copy())
                else:
                    # Move to 30% of safe range in negative direction
                    range_span = lim_max - lim_min
                    offset_neg = -0.3 * range_span
                    target_neg = default_q + offset_neg
                    target_neg = max(lim_min, min(lim_max, target_neg))  # Clamp
                    
                    pose = {name: 0.0 for name in TEST_JOINTS}
                    pose[joint_name] = target_neg - default_q
                    TEST_POSES.append(pose.copy())
                    
                    # Move to 30% of safe range in positive direction
                    offset_pos = 0.3 * range_span
                    target_pos = default_q + offset_pos
                    target_pos = max(lim_min, min(lim_max, target_pos))  # Clamp
                    
                    pose[joint_name] = target_pos - default_q
                    TEST_POSES.append(pose.copy())
        
        # Pose 7-9: Multi-joint combinations (test coupling effects)
        # Combination 1: Hip pitch + knee
        if "right_hip_pitch" in safe_limits and "right_knee" in safe_limits:
            pose = {name: 0.0 for name in TEST_JOINTS}
            hp_lim = safe_limits["right_hip_pitch"]
            k_lim = safe_limits["right_knee"]
            hp_default = self.default_joint_pos[joint_map["right_hip_pitch"]]
            k_default = self.default_joint_pos[joint_map["right_knee"]]
            
            pose["right_hip_pitch"] = -0.2 * (hp_lim[1] - hp_lim[0])
            pose["right_knee"] = 0.3 * (k_lim[1] - k_lim[0])
            TEST_POSES.append(pose.copy())
        
        # Combination 2: Hip roll + hip yaw
        if "right_hip_roll" in safe_limits and "right_hip_yaw" in safe_limits:
            pose = {name: 0.0 for name in TEST_JOINTS}
            hr_lim = safe_limits["right_hip_roll"]
            hy_lim = safe_limits["right_hip_yaw"]
            
            pose["right_hip_roll"] = -0.2 * (hr_lim[1] - hr_lim[0])
            pose["right_hip_yaw"] = 0.2 * (hy_lim[1] - hy_lim[0])
            TEST_POSES.append(pose.copy())
        
        # Combination 3: Knee + ankle pitch
        if "right_knee" in safe_limits and "right_ankle_pitch" in safe_limits:
            pose = {name: 0.0 for name in TEST_JOINTS}
            k_lim = safe_limits["right_knee"]
            ap_lim = safe_limits["right_ankle_pitch"]
            
            pose["right_knee"] = 0.4 * (k_lim[1] - k_lim[0])
            pose["right_ankle_pitch"] = -0.2 * (ap_lim[1] - ap_lim[0])
            TEST_POSES.append(pose.copy())
        
        # Final pose: Return to default
        pose_final = {name: 0.0 for name in TEST_JOINTS}
        TEST_POSES.append(pose_final)
        
        logger.info(f"Generated {len(TEST_POSES)} test poses within safe joint limits")
        
        # Log pose details
        logger.info("Test pose summary:")
        for i, pose_offsets in enumerate(TEST_POSES):
            pose_desc = ", ".join([f"{name}={offset:+.3f}" for name, offset in pose_offsets.items() if abs(offset) > 0.001])
            if not pose_desc:
                pose_desc = "default"
            logger.info(f"  Pose {i}: {pose_desc}")
        
        # Perform SysID-specific startup
        self._safe_startup_sysid(active_motor_indices)
        
        # Wait for user
        logger.info("Startup complete. Press ENTER to begin gravity test...")
        input()
        
        sample_rate = self.policy_hz
        transition_duration = 2.0  # Time to move between poses
        
        transition_steps = int(transition_duration * sample_rate)
        hold_steps = int(hold_duration * sample_rate)
        
        # Calculate total steps
        n_poses = len(TEST_POSES)
        total_steps = n_poses * hold_steps + (n_poses - 1) * transition_steps
        
        logger.info(f"Test configuration:")
        logger.info(f"  Poses: {n_poses}")
        logger.info(f"  Hold duration: {hold_duration}s per pose")
        logger.info(f"  Transition duration: {transition_duration}s between poses")
        logger.info(f"  Total duration: {total_steps / sample_rate:.1f}s")
        
        # Initialize logging
        log_data = {
            "time": [],
            "pose_idx": [],
            "phase": [],  # "hold" or "transition"
            "cmd_q": [],
            "meas_q": [],
            "meas_dq": [],
            "tau_gravity": [],   # Computed gravity compensation torque
            "tau_pd": [],        # Computed PD torque = kp*(q_des-q) - kd*dq
            "tau_measured": [],  # Measured total torque from motor
            "tau_ff_sent": [],   # Feedforward torque actually sent
        }
        
        start_time = time.time()
        current_pose_idx = 0
        
        # Build full trajectory
        full_trajectory = np.tile(self.default_joint_pos, (total_steps, 1))
        pose_labels = []
        phase_labels = []
        
        step_idx = 0
        for pose_idx, pose_offsets in enumerate(TEST_POSES):
            # Compute target positions for this pose
            target_pos = self.default_joint_pos.copy()
            for joint_name, offset in pose_offsets.items():
                motor_idx = joint_map[joint_name]
                target_pos[motor_idx] += offset
                
                # Clamp to safe limits
                if joint_name in safe_limits:
                    lim_min, lim_max = safe_limits[joint_name]
                    target_pos[motor_idx] = max(lim_min, min(lim_max, target_pos[motor_idx]))
            
            # Ensure right_hip_roll never exceeds limit for ALL poses
            clamp_right_hip_roll(target_pos)
            
            # Transition from previous pose (except for first pose)
            if pose_idx > 0:
                prev_end = step_idx
                for t_step in range(transition_steps):
                    alpha = (t_step + 1) / transition_steps
                    # Cosine interpolation for smooth transition
                    alpha_smooth = (1 - np.cos(np.pi * alpha)) / 2.0
                    interp_pos = prev_target * (1 - alpha_smooth) + target_pos * alpha_smooth
                    
                    # Clamp interpolated positions to safe limits
                    for joint_name in TEST_JOINTS:
                        motor_idx = joint_map[joint_name]
                        if joint_name in safe_limits:
                            lim_min, lim_max = safe_limits[joint_name]
                            interp_pos[motor_idx] = max(lim_min, min(lim_max, interp_pos[motor_idx]))
                    
                    # Ensure right_hip_roll never exceeds limit during transitions
                    clamp_right_hip_roll(interp_pos)
                    
                    full_trajectory[step_idx] = interp_pos
                    pose_labels.append(pose_idx)
                    phase_labels.append("transition")
                    step_idx += 1
            
            # Hold at this pose
            prev_target = target_pos.copy()
            for _ in range(hold_steps):
                full_trajectory[step_idx] = target_pos
                pose_labels.append(pose_idx)
                phase_labels.append("hold")
                step_idx += 1
        
        # Zero out inactive joints
        for i in range(full_trajectory.shape[1]):
            if i not in active_motor_indices:
                full_trajectory[:, i] = 0.0
        
        # Ensure right_hip_roll never exceeds limit in the entire trajectory
        clamp_right_hip_roll(full_trajectory)
        
        logger.info("Starting gravity test execution...")
        
        try:
            for step in range(min(step_idx, total_steps)):
                if not self._running:
                    break
                
                # Safety check
                is_unsafe, reason = self._check_all_safety_conditions()
                if is_unsafe:
                    self._enter_safety_mode(reason)
                    break
                
                target_q_vec = full_trajectory[step]
                
                # Ensure right_hip_roll never exceeds limit during execution
                clamp_right_hip_roll(target_q_vec)
                
                # Compute gravity compensation
                tau_gravity = self._compute_gravity_compensation(active_motor_indices)
                
                # Get current state
                meas_q = np.array([m.q for m in self.low_state.motor_state])
                meas_dq = np.array([m.dq for m in self.low_state.motor_state])
                meas_tau = np.array([m.tau_est for m in self.low_state.motor_state])
                
                # Compute PD torque (what the PD controller would produce)
                tau_pd = np.zeros(29)
                for i in active_motor_indices:
                    kp = float(self.kp_fixstand[i])
                    kd = float(self.kd_fixstand[i])
                    tau_pd[i] = kp * (target_q_vec[i] - meas_q[i]) - kd * meas_dq[i]
                
                # Apply commands
                for i in range(len(self.low_cmd.motor_cmd)):
                    mc = self.low_cmd.motor_cmd[i]
                    if i in active_motor_indices:
                        # Apply safety clipping
                        safe_target = self._apply_safety_clipping(
                            target_q_vec[i], meas_q[i], meas_dq[i], i
                        )
                        mc.q = float(safe_target)
                        mc.dq = 0.0
                        mc.kp = float(self.kp_fixstand[i])
                        mc.kd = float(self.kd_fixstand[i])
                        mc.tau = float(tau_gravity[i])
                    else:
                        mc.q = 0.0
                        mc.dq = 0.0
                        mc.kp = 0.0
                        mc.kd = 0.0
                        mc.tau = 0.0
                
                self.send_cmd(self.low_cmd)
                
                # Log data
                current_t = time.time() - start_time
                log_data["time"].append(current_t)
                log_data["pose_idx"].append(pose_labels[step] if step < len(pose_labels) else -1)
                log_data["phase"].append(phase_labels[step] if step < len(phase_labels) else "unknown")
                log_data["cmd_q"].append(target_q_vec.copy())
                log_data["meas_q"].append(meas_q.copy())
                log_data["meas_dq"].append(meas_dq.copy())
                log_data["tau_gravity"].append(tau_gravity.copy())
                log_data["tau_pd"].append(tau_pd.copy())
                log_data["tau_measured"].append(meas_tau.copy())
                log_data["tau_ff_sent"].append(np.array([mc.tau for mc in self.low_cmd.motor_cmd]))
                
                self.rate.sleep()
                
                # Progress logging
                if step % 200 == 0:
                    pose_idx = pose_labels[step] if step < len(pose_labels) else -1
                    phase = phase_labels[step] if step < len(phase_labels) else "?"
                    logger.info(f"Step {step}/{step_idx} - Pose {pose_idx}, Phase: {phase}")
        
        except Exception as e:
            logger.error(f"Exception during gravity test: {e}")
        finally:
            if self._running:
                self._smooth_winddown_sysid(active_motor_indices)
            
            # Save data in background
            self._save_thread = threading.Thread(
                target=self._save_gravity_test_data,
                args=(log_data, log_dir, active_motor_indices, joint_map, TEST_JOINTS),
                daemon=False
            )
            self._save_thread.start()
            logger.info("Data saving started in background...")

    def _save_gravity_test_data(self, data, log_dir, active_motor_indices, joint_map, test_joints):
        """Save gravity test data with torque comparison plots"""
        import os
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import torch
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, f"gravity_test_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Saving gravity test data to: {run_dir}")
        
        # Convert to numpy
        time_arr = np.array(data["time"])
        pose_idx = np.array(data["pose_idx"])
        phase = np.array(data["phase"])
        cmd_q = np.array(data["cmd_q"])
        meas_q = np.array(data["meas_q"])
        meas_dq = np.array(data["meas_dq"])
        tau_gravity = np.array(data["tau_gravity"])
        tau_pd = np.array(data["tau_pd"])
        tau_measured = np.array(data["tau_measured"])
        tau_ff_sent = np.array(data["tau_ff_sent"])
        
        active_indices = sorted(active_motor_indices)
        n_joints = len(active_indices)
        
        # Get joint names
        idx_to_name = {v: k for k, v in joint_map.items()}
        joint_names = [idx_to_name.get(idx, f"Joint {idx}") for idx in active_indices]
        
        # --- Plot 1: Torque Comparison (tau_gravity vs tau_pd vs tau_measured) ---
        fig1, axes1 = plt.subplots(n_joints, 1, figsize=(14, 4*n_joints), sharex=True)
        if n_joints == 1:
            axes1 = [axes1]
        fig1.suptitle('Torque Comparison: Gravity Comp vs PD vs Measured', fontsize=14)
        
        for i, idx in enumerate(active_indices):
            name = joint_names[i]
            
            # Compute expected total = tau_pd + tau_gravity
            tau_expected = tau_pd[:, idx] + tau_gravity[:, idx]
            
            axes1[i].plot(time_arr, tau_gravity[:, idx], 'b-', label='τ_gravity (computed)', linewidth=1.5, alpha=0.8)
            axes1[i].plot(time_arr, tau_pd[:, idx], 'g-', label='τ_pd (computed)', linewidth=1.5, alpha=0.8)
            axes1[i].plot(time_arr, tau_expected, 'c--', label='τ_expected (pd+grav)', linewidth=1.5, alpha=0.8)
            axes1[i].plot(time_arr, tau_measured[:, idx], 'r-', label='τ_measured (motor)', linewidth=1.5, alpha=0.8)
            
            axes1[i].set_ylabel(f'{name}\n[Nm]')
            axes1[i].legend(loc='upper right', fontsize=8)
            axes1[i].grid(True, alpha=0.3)
        
        axes1[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        fig1.savefig(os.path.join(run_dir, "torque_comparison.png"), dpi=150)
        plt.close(fig1)
        logger.info("Saved torque_comparison.png")
        
        # --- Plot 2: Torque Error (measured - expected) ---
        fig2, axes2 = plt.subplots(n_joints, 1, figsize=(14, 3*n_joints), sharex=True)
        if n_joints == 1:
            axes2 = [axes2]
        fig2.suptitle('Torque Error: Measured - Expected (should be ~0 if model is accurate)', fontsize=14)
        
        for i, idx in enumerate(active_indices):
            name = joint_names[i]
            tau_expected = tau_pd[:, idx] + tau_gravity[:, idx]
            tau_error = tau_measured[:, idx] - tau_expected
            
            axes2[i].plot(time_arr, tau_error, 'r-', linewidth=1)
            axes2[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes2[i].set_ylabel(f'{name}\n[Nm]')
            axes2[i].grid(True, alpha=0.3)
            
            # Stats for hold phases only
            hold_mask = phase == "hold"
            if np.any(hold_mask):
                mean_err = np.mean(tau_error[hold_mask])
                std_err = np.std(tau_error[hold_mask])
                axes2[i].set_title(f'Hold phase: mean={mean_err:.2f}, std={std_err:.2f} Nm', fontsize=10)
        
        axes2[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        fig2.savefig(os.path.join(run_dir, "torque_error.png"), dpi=150)
        plt.close(fig2)
        logger.info("Saved torque_error.png")
        
        # --- Plot 3: Position tracking ---
        fig3, axes3 = plt.subplots(n_joints, 1, figsize=(14, 3*n_joints), sharex=True)
        if n_joints == 1:
            axes3 = [axes3]
        fig3.suptitle('Position Tracking: Command vs Measured', fontsize=14)
        
        for i, idx in enumerate(active_indices):
            name = joint_names[i]
            axes3[i].plot(time_arr, cmd_q[:, idx], 'b-', label='Commanded', linewidth=1.5)
            axes3[i].plot(time_arr, meas_q[:, idx], 'r--', label='Measured', linewidth=1.5)
            
            # Add joint limits as horizontal lines
            limits = get_motor_joint_limits(idx, safety_margin=0.0)
            if limits:
                lim_min, lim_max = limits
                axes3[i].axhline(y=lim_min, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Joint Limits')
                axes3[i].axhline(y=lim_max, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            
            axes3[i].set_ylabel(f'{name}\n[rad]')
            axes3[i].legend(loc='upper right', fontsize=8)
            axes3[i].grid(True, alpha=0.3)
        
        axes3[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        fig3.savefig(os.path.join(run_dir, "position_tracking.png"), dpi=150)
        plt.close(fig3)
        logger.info("Saved position_tracking.png")
        
        # --- Plot 4: Gravity torque magnitude per pose (bar chart) ---
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        
        # Group by pose and compute mean gravity torque during hold
        unique_poses = sorted(set(pose_idx[phase == "hold"]))
        pose_gravity_means = {idx: [] for idx in active_indices}
        
        for p_idx in unique_poses:
            mask = (pose_idx == p_idx) & (phase == "hold")
            for motor_idx in active_indices:
                mean_tau = np.mean(np.abs(tau_gravity[mask, motor_idx]))
                pose_gravity_means[motor_idx].append(mean_tau)
        
        x = np.arange(len(unique_poses))
        width = 0.8 / n_joints
        
        for i, idx in enumerate(active_indices):
            offset = (i - n_joints/2 + 0.5) * width
            ax4.bar(x + offset, pose_gravity_means[idx], width, label=joint_names[i])
        
        ax4.set_xlabel('Pose Index')
        ax4.set_ylabel('Mean |τ_gravity| [Nm]')
        ax4.set_title('Gravity Compensation Torque per Pose')
        ax4.set_xticks(x)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig4.savefig(os.path.join(run_dir, "gravity_per_pose.png"), dpi=150)
        plt.close(fig4)
        logger.info("Saved gravity_per_pose.png")
        
        # --- Save raw data as .pt ---
        def select_joints(arr):
            return torch.tensor(arr[:, active_indices], dtype=torch.float32)
        
        pt_payload = {
            "time": torch.tensor(time_arr, dtype=torch.float32),
            "pose_idx": torch.tensor(pose_idx, dtype=torch.int32),
            "cmd_q": select_joints(cmd_q),
            "meas_q": select_joints(meas_q),
            "meas_dq": select_joints(meas_dq),
            "tau_gravity": select_joints(tau_gravity),
            "tau_pd": select_joints(tau_pd),
            "tau_measured": select_joints(tau_measured),
            "tau_ff_sent": select_joints(tau_ff_sent),
            "joint_names": joint_names,
            "active_indices": active_indices,
            "sample_rate_hz": self.policy_hz,
        }
        
        torch.save(pt_payload, os.path.join(run_dir, "gravity_test_data.pt"))
        logger.success(f"Data saved to {run_dir}/gravity_test_data.pt")
        

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="G1 Direct Joint Control Deployer")
    parser.add_argument("--device", type=str, 
                       choices=["keyboard", "joystick", "gui", "chirp", "gravity_test"],
                       default="chirp", help="Mode: chirp (sysid), gravity_test (verify grav comp), gui, keyboard")
    parser.add_argument("--step-dt", type=float, default=0.002,
                       help="Control loop update frequency in seconds (default: 0.002)")
    
    # --- Chirp Arguments ---
    parser.add_argument("--min-freq", type=float, default=0.01, help="Chirp Min Freq (Hz)")
    parser.add_argument("--max-freq", type=float, default=1.0, help="Chirp Max Freq (Hz)")
    parser.add_argument("--duration", type=float, default=25.0, help="Chirp Duration (s)")
    
    # --- Gravity Test Arguments ---
    parser.add_argument("--hold-duration", type=float, default=5.0,
                       help="Hold duration per pose in gravity test (default: 5.0s)")
    
    # --- Gain Set Selection ---
    parser.add_argument("--gains", type=str, default="standup",
                       choices=["conservative", "aggressive", "standup", "custom", "inertia"],
                       help="PD gain set: inertia (very soft), conservative, aggressive (stiff), standup (default), custom")
    
    # --- Gravity & Coriolis Compensation Arguments ---
    parser.add_argument("--gravity-comp", action="store_true",
                       help="Enable gravity compensation using MuJoCo model")
    parser.add_argument("--coriolis-comp", action="store_true",
                       help="Enable Coriolis/centrifugal compensation (requires --gravity-comp)")
    parser.add_argument("--gravity-scale", type=float, default=1.0,
                       help="Scale factor for compensation torques (default: 1.0)")
    # ------------------------------------------------

    parser.add_argument("--safety-orientation-limit", type=float, default=1.0,
                       help="Orientation safety limit")
    parser.add_argument("--safety-comm-timeout", type=float, default=0.1,
                       help="Communication timeout threshold")
    parser.add_argument("--log", action="store_true",
                       help="Enable logging")

    args = parser.parse_args()

    # ... [Keep Logger setup code unchanged] ...

    deployer = None
    try:
        deployer = G1DirectJointDeployer(
            device_type=args.device,  # chirp mode now handled directly
            step_dt=args.step_dt,
            safety_orientation_limit=args.safety_orientation_limit,
            safety_communication_timeout=args.safety_comm_timeout,
            gravity_comp_enabled=args.gravity_comp,
            coriolis_comp_enabled=args.coriolis_comp,
            gravity_comp_scale=args.gravity_scale,
            gain_set=args.gains
        )

        # Initialize connection
        deployer._initialize()
        
        # Connect Wait Loop
        logger.info("Connecting to robot...")
        while deployer.low_state.motor_state[0].q == 0.0 and deployer._running:
            time.sleep(0.1)
        
        # Safe Startup (Zero -> FixStand) - skip for modes that handle it internally
        if args.device not in ["chirp", "gravity_test"]:
            deployer._safe_startup()

        # Branch based on mode
        if args.device == "chirp":
            # Run the specific SysID routine
            deployer.run_sysid_chirp(
                min_freq=args.min_freq,
                max_freq=args.max_freq,
                duration=args.duration
            )
        elif args.device == "gravity_test":
            # Run gravity compensation verification
            deployer.run_gravity_test(
                hold_duration=args.hold_duration
            )
        elif args.device == "gui":
            deployer._run_with_gui()
        else:
            deployer._control_loop()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        if deployer is not None:
            deployer.shutdown()


if __name__ == "__main__":
    main()
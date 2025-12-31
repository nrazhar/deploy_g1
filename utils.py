"""
Utility functions for MuJoCo robot control
- Physics computations
- Ring buffer for history
- Heuristic gain calculations
"""

import numpy as np
import config


# ============================================================================
# Physics Computation (RNE Algorithm)
# ============================================================================

def mul_inert_vec(inert, v):
    """Multiply 6D vector by 6D inertia matrix."""
    res = np.zeros(6)
    res[0] = inert[0]*v[0] + inert[3]*v[1] + inert[4]*v[2] - inert[8]*v[4] + inert[7]*v[5]
    res[1] = inert[3]*v[0] + inert[1]*v[1] + inert[5]*v[2] + inert[8]*v[3] - inert[6]*v[5]
    res[2] = inert[4]*v[0] + inert[5]*v[1] + inert[2]*v[2] - inert[7]*v[3] + inert[6]*v[4]
    res[3] = inert[8]*v[1] - inert[7]*v[2] + inert[9]*v[3]
    res[4] = inert[6]*v[2] - inert[8]*v[0] + inert[9]*v[4]
    res[5] = inert[7]*v[0] - inert[6]*v[1] + inert[9]*v[5]
    return res


def cross_force(vel, f):
    """Cross-product for spatial force vectors."""
    res = np.zeros(6)
    res[0] = -vel[2]*f[1] + vel[1]*f[2]
    res[1] = vel[2]*f[0] - vel[0]*f[2]
    res[2] = -vel[1]*f[0] + vel[0]*f[1]
    res[3] = -vel[2]*f[4] + vel[1]*f[5]
    res[4] = vel[2]*f[3] - vel[0]*f[5]
    res[5] = -vel[1]*f[3] + vel[0]*f[4]
    res[0] += -vel[5]*f[4] + vel[4]*f[5]
    res[1] += vel[5]*f[3] - vel[3]*f[5]
    res[2] += -vel[4]*f[3] + vel[3]*f[4]
    return res


def mul_dof_vec(dof_mat, vec, n):
    """Multiply DOF matrix by vector."""
    if n <= 0:
        return np.zeros(6)
    elif n == 1:
        return dof_mat[0] * vec[0]
    return dof_mat[:n].T @ vec[:n]


def compute_qfrc_bias(model, data, include_gravity=True, include_coriolis=True):
    """
    Compute gravity and/or Coriolis forces using RNE algorithm.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data (with current qpos, qvel)
        include_gravity: Include gravity compensation torques
        include_coriolis: Include Coriolis/centrifugal compensation torques
    
    Returns:
        Array of joint torques for compensation
    """
    nbody, nv = model.nbody, model.nv
    cacc = np.zeros((nbody, 6))
    cfrc_body = np.zeros((nbody, 6))
    
    # Set base acceleration to gravity (if enabled)
    if include_gravity:
        cacc[0, 3:6] = -model.opt.gravity

    for i in range(1, nbody):
        pid = model.body_parentid[i]
        bda, dofnum = model.body_dofadr[i], model.body_dofnum[i]
        
        # Velocity-dependent acceleration term (Coriolis-related)
        if dofnum > 0 and include_coriolis:
            cacc[i] = cacc[pid] + mul_dof_vec(
                data.cdof_dot[bda:bda+dofnum], 
                data.qvel[bda:bda+dofnum], 
                dofnum
            )
        else:
            cacc[i] = cacc[pid].copy()
        
        # Inertia * acceleration term
        cfrc_body[i] = mul_inert_vec(data.cinert[i], cacc[i])
        
        # Coriolis/centrifugal term: v x (I * v)
        if include_coriolis:
            cfrc_body[i] += cross_force(
                data.cvel[i], 
                mul_inert_vec(data.cinert[i], data.cvel[i])
            )

    cfrc_body[0] = 0
    for i in range(nbody - 1, 0, -1):
        pid = model.body_parentid[i]
        if pid > 0:
            cfrc_body[pid] += cfrc_body[i]

    return np.array([np.dot(data.cdof[v], cfrc_body[model.dof_bodyid[v]]) for v in range(nv)])


def compute_gravity_only(model, data):
    """Compute only gravity compensation (no Coriolis)."""
    return compute_qfrc_bias(model, data, include_gravity=True, include_coriolis=False)


def compute_coriolis_only(model, data):
    """Compute only Coriolis/centrifugal compensation (no gravity)."""
    return compute_qfrc_bias(model, data, include_gravity=False, include_coriolis=True)


# ============================================================================
# Ring Buffer for Efficient History Storage
# ============================================================================

class RingBuffer:
    """Efficient numpy-based circular buffer."""
    __slots__ = ['capacity', 'data', 'index', 'count']

    def __init__(self, capacity, dtype=np.float64):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=dtype)
        self.index = 0
        self.count = 0

    def append(self, value):
        self.data[self.index] = value
        self.index = (self.index + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def get_array(self):
        if self.count < self.capacity:
            return self.data[:self.count]
        return np.concatenate([self.data[self.index:], self.data[:self.index]])


# ============================================================================
# Heuristic Gain Calculations
# ============================================================================

class GainHeuristics:
    """
    Heuristic PD gain calculations based on:
    1. Torque-limit method: Kp = τ_max / |q_max - q_min|
    2. Natural frequency method: Kp = I * ω², Kd = 2 * I * ζ * ω
    """

    @staticmethod
    def compute_kp_from_torque_limit(tau_max, q_range, conservative=True):
        """
        Kp such that at boundary of range, torque = tau_max
        conservative=True: Kp_min = tau_max / range (full range)
        conservative=False: Kp_max = tau_max / (0.5 * range) (half range, more aggressive)
        """
        if conservative:
            return tau_max / q_range if q_range > 0 else tau_max
        else:
            return tau_max / (0.5 * q_range) if q_range > 0 else tau_max * 2

    @staticmethod
    def compute_kd_from_kp(kp, ratio=20.0):
        """Kd ≈ Kp / ratio (default ratio=20)"""
        return kp / ratio

    @staticmethod
    def compute_gains_from_inertia(armature, omega=10.0, zeta=2.0):
        """
        Natural frequency method:
        Kp = I * ω², Kd = 2 * I * ζ * ω
        omega: natural frequency (Hz), default 10 Hz for compliance
        zeta: damping ratio, default 2.0 (overdamped)
        """
        omega_rad = 2 * np.pi * omega
        kp = armature * omega_rad ** 2
        kd = 2 * armature * zeta * omega_rad
        return kp, kd

    @staticmethod
    def compute_action_scale(tau_max, kp, scale=0.25):
        """α = scale * τ_max / kp (default scale=0.25)"""
        return scale * tau_max / kp if kp > 0 else 0.0


def get_joint_limits(mj_model):
    """Extract joint limits from MuJoCo model with config fallback."""
    n = mj_model.nu
    pos_min, pos_max = np.zeros(n), np.zeros(n)
    vel_limits = np.zeros(n)
    torque_limits = np.zeros(n)
    armatures = np.zeros(n)

    for i in range(n):
        # Position limits
        if i < mj_model.njnt and mj_model.jnt_limited[i]:
            pos_min[i], pos_max[i] = mj_model.jnt_range[i]
        elif i < len(config.POSITION_LIMITS):
            pos_min[i], pos_max[i] = config.POSITION_LIMITS[i]
        else:
            pos_min[i], pos_max[i] = -3.14, 3.14

        # Velocity limits
        vel_limits[i] = config.VELOCITY_LIMITS[i] if i < len(config.VELOCITY_LIMITS) else 10.0

        # Torque limits
        if hasattr(mj_model, 'actuator_forcerange') and i < mj_model.nu:
            fr = mj_model.actuator_forcerange[i]
            torque_limits[i] = max(abs(fr[0]), abs(fr[1])) if (fr[0] or fr[1]) else (
                config.TORQUE_LIMITS[i] if i < len(config.TORQUE_LIMITS) else 100)
        elif i < len(config.TORQUE_LIMITS):
            torque_limits[i] = config.TORQUE_LIMITS[i]
        else:
            torque_limits[i] = 100

        # Armature (reflected inertia)
        if i < mj_model.njnt:
            armatures[i] = mj_model.dof_armature[i] if i < len(mj_model.dof_armature) else 0.01
        else:
            armatures[i] = 0.01

    return {
        'pos_min': pos_min,
        'pos_max': pos_max,
        'pos_range': pos_max - pos_min,
        'vel_limits': vel_limits,
        'torque_limits': torque_limits,
        'armatures': armatures,
        'acc_limits': vel_limits * 20
    }


def compute_heuristic_gains(limits, method='torque', conservative=True, omega=10.0, zeta=2.0):
    """
    Compute heuristic Kp, Kd for all joints.
    method: 'torque' (torque-limit based) or 'inertia' (natural frequency based)
    """
    n = len(limits['torque_limits'])
    kp = np.zeros(n)
    kd = np.zeros(n)

    for i in range(n):
        if method == 'torque':
            kp[i] = GainHeuristics.compute_kp_from_torque_limit(
                limits['torque_limits'][i], limits['pos_range'][i], conservative)
            kd[i] = GainHeuristics.compute_kd_from_kp(kp[i])
        elif method == 'inertia':
            kp[i], kd[i] = GainHeuristics.compute_gains_from_inertia(
                limits['armatures'][i], omega, zeta)
        else:
            kp[i], kd[i] = config.KPS[i], config.KDS[i]

    return kp, kd


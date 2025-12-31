import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import config
import numpy as np


locker = threading.Lock()


def mul_inert_vec(inert, v):
    """
    Multiply 6D vector (rotation, translation) by 6D inertia matrix.
    inert: packed 10-element inertia [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m*cx, m*cy, m*cz, m]
    v: 6D spatial vector [angular(3), linear(3)]
    Returns: 6D force vector
    """
    res = np.zeros(6)
    res[0] = inert[0]*v[0] + inert[3]*v[1] + inert[4]*v[2] - inert[8]*v[4] + inert[7]*v[5]
    res[1] = inert[3]*v[0] + inert[1]*v[1] + inert[5]*v[2] + inert[8]*v[3] - inert[6]*v[5]
    res[2] = inert[4]*v[0] + inert[5]*v[1] + inert[2]*v[2] - inert[7]*v[3] + inert[6]*v[4]
    res[3] = inert[8]*v[1] - inert[7]*v[2] + inert[9]*v[3]
    res[4] = inert[6]*v[2] - inert[8]*v[0] + inert[9]*v[4]
    res[5] = inert[7]*v[0] - inert[6]*v[1] + inert[9]*v[5]
    return res


def cross_force(vel, f):
    """
    Cross-product for force vectors (spatial force cross product).
    vel: 6D velocity [angular(3), linear(3)]
    f: 6D force [torque(3), force(3)]
    Returns: 6D force
    """
    res = np.zeros(6)
    res[0] = -vel[2]*f[1] + vel[1]*f[2]
    res[1] =  vel[2]*f[0] - vel[0]*f[2]
    res[2] = -vel[1]*f[0] + vel[0]*f[1]
    res[3] = -vel[2]*f[4] + vel[1]*f[5]
    res[4] =  vel[2]*f[3] - vel[0]*f[5]
    res[5] = -vel[1]*f[3] + vel[0]*f[4]

    res[0] += -vel[5]*f[4] + vel[4]*f[5]
    res[1] +=  vel[5]*f[3] - vel[3]*f[5]
    res[2] += -vel[4]*f[3] + vel[3]*f[4]
    return res


def mul_dof_vec(dof_mat, vec, n):
    """
    Multiply DOF matrix (6 x n, stored as n x 6) by vector (n).
    dof_mat: (n, 6) array of DOF columns
    vec: (n,) array
    Returns: 6D vector
    """
    if n <= 0:
        return np.zeros(6)
    elif n == 1:
        return dof_mat[0] * vec[0]
    else:
        return dof_mat[:n].T @ vec[:n]


def compute_qfrc_bias(model, data):
    """
    Compute qfrc_bias using the Recursive Newton-Euler (RNE) algorithm.
    This computes C(qpos, qvel) - the Coriolis, centrifugal, and gravity forces.
    
    Algorithm:
    1. Forward pass: compute center of mass acceleration (cacc) for each body
    2. Compute body forces: cfrc = cinert * cacc + cvel × (cinert * cvel)
    3. Backward pass: accumulate forces from children to parents
    4. Project onto DOFs: qfrc_bias = cdof · cfrc
    """
    nbody = model.nbody
    nv = model.nv
    
    # Allocate arrays
    cacc = np.zeros((nbody, 6))
    cfrc_body = np.zeros((nbody, 6))
    
    # Set world acceleration to -gravity (body 0)
    gravity = model.opt.gravity
    cacc[0, 3:6] = -gravity  # linear part is -gravity
    
    # Forward pass over bodies: accumulate cacc, compute cfrc_body
    for i in range(1, nbody):
        parent_id = model.body_parentid[i]
        bda = model.body_dofadr[i]  # body's first DOF address
        dofnum = model.body_dofnum[i]  # number of DOFs for this body
        
        # cacc = cacc_parent + cdof_dot * qvel
        # cdof_dot is stored as (nv, 6), we need columns bda:bda+dofnum
        if dofnum > 0:
            cdof_dot_body = data.cdof_dot[bda:bda+dofnum]  # (dofnum, 6)
            qvel_body = data.qvel[bda:bda+dofnum]
            tmp = mul_dof_vec(cdof_dot_body, qvel_body, dofnum)
            cacc[i] = cacc[parent_id] + tmp
        else:
            cacc[i] = cacc[parent_id].copy()
        
        # cfrc_body = cinert * cacc + cvel × (cinert * cvel)
        cinert_i = data.cinert[i]  # (10,)
        cvel_i = data.cvel[i]  # (6,)
        
        cfrc_body[i] = mul_inert_vec(cinert_i, cacc[i])
        tmp = mul_inert_vec(cinert_i, cvel_i)
        cfrc_body[i] += cross_force(cvel_i, tmp)
    
    # Clear world cfrc_body (should already be zero)
    cfrc_body[0] = 0
    
    # Backward pass over bodies: accumulate cfrc_body from children to parents
    for i in range(nbody - 1, 0, -1):
        parent_id = model.body_parentid[i]
        if parent_id > 0:
            cfrc_body[parent_id] += cfrc_body[i]
    
    # Compute qfrc_bias = cdof · cfrc_body
    qfrc_bias = np.zeros(nv)
    for v in range(nv):
        body_id = model.dof_bodyid[v]
        qfrc_bias[v] = np.dot(data.cdof[v], cfrc_body[body_id])
    
    return qfrc_bias


mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
for i in range(mj_model.ngeom):
    if mj_model.geom_bodyid[i] != 0:  # If not the floor/world
        mj_model.geom_conaffinity[i] = 0

mj_data = mujoco.MjData(mj_model)

# Print joint order
print("Joint order in simulation:")
for i in range(mj_model.nu):
    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  Joint {i}: {joint_name}")

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT

# Assign kp, kd and default values
kp = config.KPS
kd = config.KDS
default_angles = config.DEFAULT_ANGLES
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def SimulationThread():
    global mj_data, mj_model

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        # Get current joint positions and velocities
        # (fixed base, no floating DOF)
        q = mj_data.qpos  # All DOF are joint DOF
        dq = mj_data.qvel  # All velocities are joint velocities

        # PD control to default angles with zero velocity targets
        target_dq = np.zeros_like(dq)
        torques = pd_control(
            np.array(default_angles), q, np.array(kp),
            target_dq, dq, np.array(kd)
        )

        # Gravity compensation using RNE (qfrc_bias = gravity + Coriolis forces)
        gravity_comp_tau = compute_qfrc_bias(mj_model, mj_data) #mj_model.qfrc_bias[:]
        # print(f"Gravity compensation torque: {gravity_comp_tau}")

        # Apply torques with gravity compensation
        mj_data.ctrl[:] = torques + gravity_comp_tau

        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()

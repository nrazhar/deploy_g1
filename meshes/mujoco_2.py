# In humanoidverse/deploy/mujoco_2.py

import torch
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

# Import the new minimal base class
from .urcirobot_2 import URCIRobot_2

# --- Helper functions for control and observation ---

def get_gravity_orientation(quaternion):
    """Calculates the orientation of the gravity vector in the base frame."""
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)
    ])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates PD control torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


# --- The new MujocoRobot class for simple simulation mode ---

class MujocoRobot_2(URCIRobot_2):
    """
    A simplified MuJoCo robot class for a direct simulation mode.
    It inherits the minimal URCIRobot_2 base class and contains
    the logic to run a simple policy-driven simulation loop, plus a tiny GUI
    to edit the command vector live.
    """
    def __init__(self, cfg):
        """
        Initializes the simulation environment from the configuration.
        """
        # Call the initializer of the new, simple base class
        super().__init__(cfg)

        # Load the MuJoCo model and data from the paths specified in the config
        xml_path = self.cfg.xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.simulation_dt

        # --- GUI state (command editor) ---
        # Defaults: derive names from length; ranges [-1, 1] if not provided.
        self._cmd_lock = threading.Lock()
        self._cmd_default = np.array(self.cfg.cmd_init, dtype=np.float32)
        self._cmd = self._cmd_default.copy()

        self._cmd_names = getattr(self.cfg, "cmd_names", None)
        if self._cmd_names is None or len(self._cmd_names) != len(self._cmd_default):
            self._cmd_names = [f"cmd[{i}]" for i in range(len(self._cmd_default))]

        # cmd_ranges: list of (min, max) pairs
        self._cmd_ranges = getattr(self.cfg, "cmd_ranges", None)
        if (self._cmd_ranges is None) or (len(self._cmd_ranges) != len(self._cmd_default)):
            self._cmd_ranges = [(-1.0, 1.0)] * len(self._cmd_default)

        # Launch a small Tkinter GUI in a separate thread
        self._gui_thread = threading.Thread(target=self._launch_cmd_gui, daemon=True)
        self._gui_thread.start()

        print("MujocoRobot_2 Initialized with GUI for live command editing.")

    # ---------- GUI helpers ----------
    def _launch_cmd_gui(self):
        """
        Launches a minimal Tkinter window with sliders for each command dimension.
        Sliders write into self._cmd under a lock so the sim loop can read safely.
        """
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as e:
            print(f"[GUI] Tkinter not available or failed to start: {e}")
            return

        root = tk.Tk()
        root.title("MujocoRobot_2 — Command Editor")

        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)

        # store vars to avoid GC
        self._tk_vars = []

        # Sliders
        for i, (name, (vmin, vmax), val) in enumerate(zip(self._cmd_names, self._cmd_ranges, self._cmd_default)):
            row = ttk.Frame(main)
            row.pack(fill="x", pady=6)

            lbl = ttk.Label(row, text=name, width=18)
            lbl.pack(side="left")

            var = tk.DoubleVar(value=float(val))
            self._tk_vars.append(var)

            def on_change(*_args, idx=i, v=var, lo=vmin, hi=vmax):
                # Clamp and write to self._cmd under lock
                val = float(v.get())
                if val < lo: val = lo
                if val > hi: val = hi
                with self._cmd_lock:
                    self._cmd[idx] = val

            scl = ttk.Scale(
                row,
                from_=vmin,
                to=vmax,
                orient="horizontal",
                variable=var,
                command=lambda _evt=None, cb=on_change: cb()
            )
            scl.pack(side="left", fill="x", expand=True, padx=8)

            val_lbl = ttk.Label(row, textvariable=var, width=8)
            val_lbl.pack(side="right")

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(10, 0))

        def reset():
            for i, v in enumerate(self._tk_vars):
                v.set(float(self._cmd_default[i]))
            with self._cmd_lock:
                self._cmd[:] = self._cmd_default

        def zero():
            for i, v in enumerate(self._tk_vars):
                v.set(0.0)
            with self._cmd_lock:
                self._cmd[:] = 0.0

        reset_btn = ttk.Button(btns, text="Reset to Default", command=reset)
        reset_btn.pack(side="left")

        zero_btn = ttk.Button(btns, text="Zero", command=zero)
        zero_btn.pack(side="left", padx=8)

        info = ttk.Label(
            main,
            text="Tip: Adjust sliders during the sim.\nValues are read every control step.",
            foreground="#666"
        )
        info.pack(fill="x", pady=(8, 0))

        try:
            root.mainloop()
        except Exception as e:
            print(f"[GUI] mainloop ended: {e}")

    def _get_cmd(self):
        with self._cmd_lock:
            return self._cmd.copy()

    # ---------- Simulation ----------
    def run_simple_simulation(self):
        """
        Runs a direct simulation loop using the logic from the standalone
        deploy_mujoco.py script, with live command input from the GUI.
        """
        print("Starting simple simulation loop...")
        
        # Load the policy from the path in the config
        policy = torch.jit.load(self.cfg.policy_path)

        # Initialize variables from the config
        default_angles = np.array(self.cfg.default_angles, dtype=np.float32)
        kps = np.array(self.cfg.kps, dtype=np.float32)
        kds = np.array(self.cfg.kds, dtype=np.float32)

        # NOTE: cmd is now live — pulled from GUI each control step
        # cmd_scale still comes from cfg
        cmd_scale = np.array(self.cfg.cmd_scale, dtype=np.float32)

        # Initialize state variables
        action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        obs = np.zeros(self.cfg.num_obs, dtype=np.float32)
        counter = 0

        # Launch the MuJoCo viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < self.cfg.simulation_duration:
                step_start_time = time.time()
                
                # --- This is the core loop from deploy_mujoco.py ---
                
                # PD Control
                tau = pd_control(
                    target_dof_pos, self.data.qpos[7:], kps,
                    np.zeros_like(kds), self.data.qvel[6:], kds
                )
                self.data.ctrl[:] = tau
                mujoco.mj_step(self.model, self.data)
                counter += 1

                # Policy inference step
                if counter % self.cfg.control_decimation == 0:
                    # Construct observation
                    qj = (self.data.qpos[7:] - default_angles) * self.cfg.dof_pos_scale
                    dqj = self.data.qvel[6:] * self.cfg.dof_vel_scale
                    quat = self.data.qpos[3:7]
                    omega = self.data.qvel[3:6] * self.cfg.ang_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)

                    period = 0.8
                    count = counter * self.cfg.simulation_dt
                    phase = (count % period) / period
                    sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

                    # Pull latest command from the GUI
                    cmd = self._get_cmd().astype(np.float32)

                    # Make sure cmd_scale length matches
                    if cmd_scale.shape[0] != cmd.shape[0]:
                        raise ValueError(
                            f"cmd_scale length ({cmd_scale.shape[0]}) does not match cmd length ({cmd.shape[0]}). "
                            "Update cfg.cmd_scale to match cfg.cmd_init length."
                        )

                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:6+cmd.shape[0]] = cmd * cmd_scale  # (re)packed to fit any cmd length

                    base = 6 + cmd.shape[0]
                    obs[base : base + self.cfg.num_actions] = qj
                    obs[base + self.cfg.num_actions : base + 2 * self.cfg.num_actions] = dqj
                    obs[base + 2 * self.cfg.num_actions : base + 3 * self.cfg.num_actions] = action
                    obs[base + 3 * self.cfg.num_actions : base + 3 * self.cfg.num_actions + 2] = np.array([sin_phase, cos_phase])

                    # Get action from policy
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()

                    # Update target positions
                    target_dof_pos = action * self.cfg.action_scale + default_angles

                viewer.sync()
                
                # Maintain simulation rate
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

"""
MuJoCo Robot GUI Controller - Main Application
"""

import sys
import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QGroupBox, QScrollArea, QSplitter, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor

import config
from utils import RingBuffer, compute_qfrc_bias, get_joint_limits
from gui_widgets import (
    JointSlider, ForcePositionSlider, GainControlWidget, PlotWidget,
    JointDetailPanel, OverviewPanel, TorquePanel, GainHeuristicPanel
)


# ============================================================================
# Shared State
# ============================================================================

class RobotState:
    """Thread-safe shared state between simulation and GUI."""

    def __init__(self, num_joints, max_time_window, sim_dt):
        self.lock = Lock()
        self.num_joints = num_joints
        self.sim_dt = sim_dt
        self.buffer_size = int(max_time_window / sim_dt) + 100

        self.target_positions = np.array(config.DEFAULT_ANGLES, dtype=np.float64)
        self.target_velocities = np.zeros(num_joints)
        self.force_positions = np.zeros(num_joints)
        self.force_position_enabled = np.zeros(num_joints, dtype=bool)

        self.pd_enabled = True
        self.gravity_comp_enabled = True
        self.coriolis_comp_enabled = False

        self.kp = np.array(config.KPS, dtype=np.float64)
        self.kd = np.array(config.KDS, dtype=np.float64)

        self.positions = np.zeros(num_joints)
        self.velocities = np.zeros(num_joints)
        self.accelerations = np.zeros(num_joints)
        self.prev_velocities = np.zeros(num_joints)
        self.torque_pd = np.zeros(num_joints)
        self.torque_gravity = np.zeros(num_joints)
        self.torque_coriolis = np.zeros(num_joints)

        # Buffers
        self.time_buffer = RingBuffer(self.buffer_size)
        self.pos_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.vel_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.acc_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.torque_pd_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.torque_grav_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.torque_cor_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.torque_total_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]
        self.target_pos_buffers = [RingBuffer(self.buffer_size) for _ in range(num_joints)]

        self.sim_time = 0.0
        self.running = True

    def get_control_params(self):
        with self.lock:
            return (self.target_positions.copy(), self.target_velocities.copy(),
                    self.pd_enabled, self.gravity_comp_enabled, self.coriolis_comp_enabled,
                    self.kp.copy(), self.kd.copy())

    def get_force_positions(self):
        with self.lock:
            return self.force_positions.copy(), self.force_position_enabled.copy()

    def set_target_position(self, idx, val):
        with self.lock:
            self.target_positions[idx] = val

    def set_all_targets(self, pos):
        with self.lock:
            self.target_positions[:] = pos

    def set_force_position(self, idx, val):
        with self.lock:
            self.force_positions[idx] = val

    def set_all_force_positions(self, pos):
        with self.lock:
            self.force_positions[:] = pos

    def set_force_position_enabled(self, idx, en):
        with self.lock:
            self.force_position_enabled[idx] = en

    def set_pd_enabled(self, en):
        with self.lock:
            self.pd_enabled = en

    def set_gravity_comp_enabled(self, en):
        with self.lock:
            self.gravity_comp_enabled = en

    def set_coriolis_comp_enabled(self, en):
        with self.lock:
            self.coriolis_comp_enabled = en

    def set_kp(self, idx, val):
        with self.lock:
            self.kp[idx] = val

    def set_kd(self, idx, val):
        with self.lock:
            self.kd[idx] = val

    def set_kp_kd(self, idx, kp, kd):
        with self.lock:
            self.kp[idx] = kp
            self.kd[idx] = kd

    def set_all_kp(self, kp):
        with self.lock:
            self.kp[:] = kp

    def set_all_kd(self, kd):
        with self.lock:
            self.kd[:] = kd

    def update_state(self, positions, velocities, torque_pd, torque_gravity, torque_coriolis, sim_time):
        with self.lock:
            self.accelerations = (velocities - self.prev_velocities) / self.sim_dt
            self.prev_velocities[:] = velocities
            self.positions[:] = positions
            self.velocities[:] = velocities
            self.torque_pd[:] = torque_pd
            self.torque_gravity[:] = torque_gravity
            self.torque_coriolis[:] = torque_coriolis
            self.sim_time = sim_time

            self.time_buffer.append(sim_time)
            for i in range(self.num_joints):
                self.pos_buffers[i].append(positions[i])
                self.vel_buffers[i].append(velocities[i])
                self.acc_buffers[i].append(self.accelerations[i])
                self.torque_pd_buffers[i].append(torque_pd[i])
                self.torque_grav_buffers[i].append(torque_gravity[i])
                self.torque_cor_buffers[i].append(torque_coriolis[i])
                self.torque_total_buffers[i].append(torque_pd[i] + torque_gravity[i] + torque_coriolis[i])
                self.target_pos_buffers[i].append(self.target_positions[i])

    def get_state(self):
        with self.lock:
            return {
                'positions': self.positions.copy(),
                'velocities': self.velocities.copy(),
                'accelerations': self.accelerations.copy(),
                'torque_pd': self.torque_pd.copy(),
                'torque_gravity': self.torque_gravity.copy(),
                'torque_coriolis': self.torque_coriolis.copy(),
                'torque_total': self.torque_pd + self.torque_gravity + self.torque_coriolis,
                'target_positions': self.target_positions.copy(),
                'sim_time': self.sim_time,
                'pd_enabled': self.pd_enabled,
                'gravity_comp_enabled': self.gravity_comp_enabled,
                'coriolis_comp_enabled': self.coriolis_comp_enabled,
                'kp': self.kp.copy(),
                'kd': self.kd.copy()
            }

    def get_history(self, idx, time_window):
        with self.lock:
            t = self.time_buffer.get_array()
            if len(t) == 0:
                e = np.array([])
                return {'time': e, 'position': e, 'velocity': e, 'acceleration': e,
                        'torque_pd': e, 'torque_gravity': e, 'torque_coriolis': e, 
                        'torque_total': e, 'target_position': e}
            mask = t >= (self.sim_time - time_window)
            return {
                'time': t[mask],
                'position': self.pos_buffers[idx].get_array()[mask],
                'velocity': self.vel_buffers[idx].get_array()[mask],
                'acceleration': self.acc_buffers[idx].get_array()[mask],
                'torque_pd': self.torque_pd_buffers[idx].get_array()[mask],
                'torque_gravity': self.torque_grav_buffers[idx].get_array()[mask],
                'torque_coriolis': self.torque_cor_buffers[idx].get_array()[mask],
                'torque_total': self.torque_total_buffers[idx].get_array()[mask],
                'target_position': self.target_pos_buffers[idx].get_array()[mask]
            }

    def get_time_range(self, time_window):
        with self.lock:
            return self.sim_time - time_window, self.sim_time


# ============================================================================
# Simulation Controller
# ============================================================================

class SimulationController:
    """Manages MuJoCo simulation in separate threads."""

    def __init__(self, state: RobotState):
        self.state = state
        self.mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
        for i in range(self.mj_model.ngeom):
            if self.mj_model.geom_bodyid[i] != 0:
                self.mj_model.geom_conaffinity[i] = 0
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = config.SIMULATE_DT
        self.limits = get_joint_limits(self.mj_model)
        self.joint_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"J{i}"
                           for i in range(self.mj_model.nu)]
        self.viewer = None
        self.viewer_lock = Lock()

    def start(self):
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        time.sleep(0.2)
        Thread(target=self._sim_loop, daemon=True).start()
        Thread(target=self._viewer_loop, daemon=True).start()

    def _sim_loop(self):
        while self.state.running and self.viewer.is_running():
            t0 = time.perf_counter()
            with self.viewer_lock:
                tgt_q, tgt_dq, pd_en, grav_en, cor_en, kp, kd = self.state.get_control_params()
                fpos, fen = self.state.get_force_positions()
                for i in range(len(fen)):
                    if fen[i]:
                        self.mj_data.qpos[i], self.mj_data.qvel[i] = fpos[i], 0.0
                q, dq = self.mj_data.qpos.copy(), self.mj_data.qvel.copy()
                tau_pd = (tgt_q - q) * kp + (tgt_dq - dq) * kd if pd_en else np.zeros_like(q)
                
                # Compute gravity and coriolis separately for visualization
                tau_grav = compute_qfrc_bias(self.mj_model, self.mj_data, 
                                             include_gravity=True, include_coriolis=False) if grav_en else np.zeros_like(q)
                tau_cor = compute_qfrc_bias(self.mj_model, self.mj_data, 
                                            include_gravity=False, include_coriolis=True) if cor_en else np.zeros_like(q)
                
                self.mj_data.ctrl[:] = tau_pd + tau_grav + tau_cor
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.state.update_state(q, dq, tau_pd, tau_grav, tau_cor, self.mj_data.time)
            dt = self.mj_model.opt.timestep - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    def _viewer_loop(self):
        while self.state.running and self.viewer.is_running():
            with self.viewer_lock:
                self.viewer.sync()
            time.sleep(config.VIEWER_DT)

    def stop(self):
        self.state.running = False
        if self.viewer:
            self.viewer.close()


# ============================================================================
# Main GUI
# ============================================================================

class RobotGUI(QMainWindow):
    """Main GUI window."""

    def __init__(self, sim: SimulationController, state: RobotState):
        super().__init__()
        self.sim, self.state = sim, state
        self.limits = sim.limits
        self.time_window = config.PLOT_TIME_WINDOW

        n = state.num_joints
        self.armatures = np.array(config.ARMATURES[:n] if hasattr(config, 'ARMATURES') else [0.05] * n)
        self.zetas = np.array(config.DAMPING_RATIOS[:n] if hasattr(config, 'DAMPING_RATIOS') else [2.0] * n)

        self.setWindowTitle("MuJoCo G1 Controller")
        self.setGeometry(20, 20, 1900, 1020)
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#1e1e1e;color:#e0e0e0;}
            QGroupBox{color:#aaa;border:1px solid #444;border-radius:3px;margin-top:6px;padding-top:6px;font-size:10px;}
            QGroupBox::title{subcontrol-origin:margin;left:6px;padding:0 3px;}
            QPushButton{background:#2d5aa0;color:#fff;border:none;padding:5px 10px;border-radius:3px;font-size:10px;}
            QPushButton:hover{background:#3d6ab0;}
            QPushButton:checked{background:#0a4;}
            QTabWidget::pane{border:1px solid #444;}
            QTabBar::tab{background:#2d2d2d;color:#888;padding:5px 12px;border-top-left-radius:3px;border-top-right-radius:3px;}
            QTabBar::tab:selected{background:#0af;color:#fff;}
            QScrollArea{border:none;}
            QCheckBox{color:#e0e0e0;}
            QComboBox{background:#2d2d2d;color:#fff;border:1px solid #444;padding:3px;}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        splitter.addWidget(self._make_ctrl_panel())
        splitter.addWidget(self._make_viz_panel())
        splitter.setSizes([750, 1150])

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(50)
        self._cnt = 0
        self.sel_joint = 0
        self._select_joint(0)

    def _make_ctrl_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)

        # Top controls
        top = QHBoxLayout()
        self.pd_btn = QPushButton("PD")
        self.pd_btn.setCheckable(True)
        self.pd_btn.setChecked(True)
        self.pd_btn.clicked.connect(lambda c: self.state.set_pd_enabled(c))
        self.grav_btn = QPushButton("Grav")
        self.grav_btn.setCheckable(True)
        self.grav_btn.setChecked(True)
        self.grav_btn.clicked.connect(lambda c: self.state.set_gravity_comp_enabled(c))
        self.grav_btn.setToolTip("Gravity compensation")
        self.cor_btn = QPushButton("Cor")
        self.cor_btn.setCheckable(True)
        self.cor_btn.setChecked(False)
        self.cor_btn.clicked.connect(lambda c: self.state.set_coriolis_comp_enabled(c))
        self.cor_btn.setToolTip("Coriolis/centrifugal compensation")
        rst_btn = QPushButton("Reset")
        rst_btn.clicked.connect(self._reset_all)
        zero_btn = QPushButton("Zero")
        zero_btn.clicked.connect(self._zero)
        smooth_rst_btn = QPushButton("⟳Reset")
        smooth_rst_btn.clicked.connect(self._smooth_reset_default)
        smooth_rst_btn.setStyleSheet("background:#2a7;")
        smooth_zero_btn = QPushButton("⟳Zero")
        smooth_zero_btn.clicked.connect(self._smooth_zero)
        smooth_zero_btn.setStyleSheet("background:#2a7;")
        for b in [self.pd_btn, self.grav_btn, self.cor_btn, rst_btn, zero_btn, smooth_rst_btn, smooth_zero_btn]:
            top.addWidget(b)
        layout.addLayout(top)

        # Time window
        tw_layout = QHBoxLayout()
        tw_layout.addWidget(QLabel("Time:"))
        self.tw_spin = QDoubleSpinBox()
        self.tw_spin.setRange(1, 60)
        self.tw_spin.setValue(self.time_window)
        self.tw_spin.setSuffix("s")
        self.tw_spin.valueChanged.connect(lambda v: setattr(self, 'time_window', v))
        self.tw_spin.setFixedWidth(60)
        self.tw_spin.setStyleSheet("background:#2d2d2d;color:#0f8;border:1px solid #444;font-size:10px;")
        tw_layout.addWidget(self.tw_spin)
        tw_layout.addStretch()
        layout.addLayout(tw_layout)

        # Target/Force Tabs (upper section)
        tabs = QTabWidget()
        tabs.addTab(self._make_target_tab(), "Target")
        tabs.addTab(self._make_force_tab(), "Force J")
        layout.addWidget(tabs, 1)

        # Gains section (below tabs, not as a tab)
        gains_group = QGroupBox("PD Gains")
        gains_layout = QVBoxLayout(gains_group)
        gains_layout.setSpacing(2)
        gains_layout.addWidget(self._make_gains_content())
        layout.addWidget(gains_group, 1)

        self.status = QLabel("Running...")
        self.status.setStyleSheet("color:#0f8;font-size:9px;padding:2px;")
        layout.addWidget(self.status)
        return panel

    def _make_target_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(2)

        # Random button row
        btn_row = QHBoxLayout()
        rand_btn = QPushButton("🎲 Random")
        rand_btn.setToolTip("Set random positions within joint limits")
        rand_btn.clicked.connect(self._random_targets)
        rand_btn.setStyleSheet("background:#a52;")
        btn_row.addWidget(rand_btn)
        btn_row.addStretch()
        l.addLayout(btn_row)

        self.sliders = []
        for gn, s, e in [("L.Leg", 0, 6), ("R.Leg", 6, 12), ("Waist", 12, 13), ("L.Arm", 13, 18), ("R.Arm", 18, 23)]:
            g = QGroupBox(gn)
            gl = QVBoxLayout(g)
            gl.setSpacing(1)
            for i in range(s, e):
                nm = self.sim.joint_names[i]
                sl = JointSlider(i, nm, config.DEFAULT_ANGLES[i],
                                self.limits['pos_min'][i], self.limits['pos_max'][i])
                sl.value_changed.connect(lambda idx, v: self.state.set_target_position(idx, v))
                sl.name_label.mousePressEvent = lambda e, idx=i: self._select_joint(idx)
                sl.name_label.setCursor(Qt.PointingHandCursor)
                gl.addWidget(sl)
                self.sliders.append(sl)
            l.addWidget(g)
        l.addStretch()
        scroll.setWidget(w)
        return scroll

    def _make_force_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(2)

        btn_row = QHBoxLayout()
        sync_all_btn = QPushButton("⟳ Get All")
        sync_all_btn.setToolTip("Copy current positions to all force sliders")
        sync_all_btn.clicked.connect(self._sync_all_force_positions)
        sync_all_btn.setStyleSheet("background:#555;")
        btn_row.addWidget(sync_all_btn)
        rand_btn = QPushButton("🎲 Random")
        rand_btn.setToolTip("Set random force positions within limits")
        rand_btn.clicked.connect(self._random_force_positions)
        rand_btn.setStyleSheet("background:#a52;")
        btn_row.addWidget(rand_btn)
        btn_row.addStretch()
        l.addLayout(btn_row)

        info = QLabel("☑ enables forcing. ⟳ copies current. R resets.")
        info.setStyleSheet("color:#f60;font-size:9px;padding:2px;")
        l.addWidget(info)

        self.fsliders = []
        for gn, s, e in [("L.Leg", 0, 6), ("R.Leg", 6, 12), ("Waist", 12, 13), ("L.Arm", 13, 18), ("R.Arm", 18, 23)]:
            g = QGroupBox(gn)
            gl = QVBoxLayout(g)
            gl.setSpacing(1)
            for i in range(s, e):
                nm = self.sim.joint_names[i]
                fs = ForcePositionSlider(i, nm, self.limits['pos_min'][i], self.limits['pos_max'][i],
                                        self._get_cur_pos)
                fs.value_changed.connect(lambda idx, v: self.state.set_force_position(idx, v))
                fs.enabled_changed.connect(lambda idx, en: self.state.set_force_position_enabled(idx, en))
                fs.rst_btn.clicked.connect(lambda _, idx=i: self._reset_force_to_default(idx))
                gl.addWidget(fs)
                self.fsliders.append(fs)
            l.addWidget(g)
        l.addStretch()
        scroll.setWidget(w)
        return scroll

    def _make_gains_content(self):
        """Create gains content widget (not in scroll)."""
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(2)
        l.setContentsMargins(0, 0, 0, 0)

        # Heuristic panel
        self.heur_panel = GainHeuristicPanel(self.limits, self.state.num_joints,
                                             self._get_all_armatures, self._get_all_zetas)
        self.heur_panel.apply_gains.connect(self._apply_heuristic_gains)
        self.heur_panel.set_update_zetas_callback(self._update_all_zetas)
        l.addWidget(self.heur_panel)

        # Reset/Set all row
        rst_row = QHBoxLayout()
        rst_gains_btn = QPushButton("Reset All")
        rst_gains_btn.clicked.connect(self._reset_all_gains)
        rst_gains_btn.setStyleSheet("background:#555;")
        rst_row.addWidget(rst_gains_btn)
        set_all_btn = QPushButton("Set All")
        set_all_btn.clicked.connect(self._set_all_gains_from_widgets)
        set_all_btn.setStyleSheet("background:#2a7;")
        rst_row.addWidget(set_all_btn)
        rst_row.addStretch()
        l.addLayout(rst_row)

        # Scrollable per-joint gain controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_w = QWidget()
        scroll_l = QVBoxLayout(scroll_w)
        scroll_l.setSpacing(1)

        self.gain_widgets = []
        for gn, s, e in [("L.Leg", 0, 6), ("R.Leg", 6, 12), ("Waist", 12, 13), ("L.Arm", 13, 18), ("R.Arm", 18, 23)]:
            g = QGroupBox(gn)
            gl = QVBoxLayout(g)
            gl.setSpacing(1)
            for i in range(s, e):
                nm = self.sim.joint_names[i]
                gw = GainControlWidget(
                    i, nm, config.KPS[i], config.KDS[i],
                    config.KPS[i], config.KDS[i],
                    self.armatures[i], self.zetas[i]
                )
                gw.gains_changed.connect(self._on_gain_set)
                gw.armature_changed.connect(self._on_armature_change)
                gw.zeta_changed.connect(self._on_zeta_change)
                gl.addWidget(gw)
                self.gain_widgets.append(gw)
            scroll_l.addWidget(g)
        scroll_l.addStretch()
        scroll.setWidget(scroll_w)
        l.addWidget(scroll)
        return w

    def _make_viz_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        tabs = QTabWidget()

        self.detail = JointDetailPanel(self.limits)
        tabs.addTab(self.detail, "Detail")

        self.overview = OverviewPanel(self.state.num_joints, self.sim.joint_names, self.limits)
        tabs.addTab(self.overview, "Overview")

        self.torque_panel = TorquePanel(self.state.num_joints, self.sim.joint_names, self.limits)
        tabs.addTab(self.torque_panel, "Torque")

        layout.addWidget(tabs)
        return panel

    # ==================== Callbacks ====================

    def _get_cur_pos(self, idx):
        return self.state.get_state()['positions'][idx]

    def _get_all_armatures(self):
        return np.array([gw.get_armature() for gw in self.gain_widgets])

    def _get_all_zetas(self):
        return np.array([gw.get_zeta() for gw in self.gain_widgets])

    def _on_gain_set(self, idx, kp, kd):
        self.state.set_kp_kd(idx, kp, kd)

    def _on_armature_change(self, idx, val):
        self.armatures[idx] = val

    def _on_zeta_change(self, idx, val):
        self.zetas[idx] = val

    def _update_all_zetas(self, zeta):
        for gw in self.gain_widgets:
            gw.zeta_spin.setValue(zeta)
        self.zetas[:] = zeta

    def _select_joint(self, idx):
        self.sel_joint = idx
        self.detail.set_joint(self.sim.joint_names[idx], idx)
        for i, s in enumerate(self.sliders):
            s.name_label.setStyleSheet("color:#0f8;font-size:10px;background:#333;" if i == idx else "color:#e0e0e0;font-size:10px;")

    def _reset_all(self):
        self.state.set_all_targets(np.array(config.DEFAULT_ANGLES))
        for i, s in enumerate(self.sliders):
            s.set_val(config.DEFAULT_ANGLES[i])

    def _zero(self):
        self.state.set_all_targets(np.zeros(self.state.num_joints))
        for s in self.sliders:
            s.set_val(0)

    def _random_targets(self):
        """Set random target positions within joint limits."""
        rand_pos = np.array([np.random.uniform(self.limits['pos_min'][i], self.limits['pos_max'][i])
                            for i in range(self.state.num_joints)])
        self.state.set_all_targets(rand_pos)
        for i, s in enumerate(self.sliders):
            s.set_val(rand_pos[i])

    def _random_force_positions(self):
        """Set random force positions within joint limits."""
        rand_pos = np.array([np.random.uniform(self.limits['pos_min'][i], self.limits['pos_max'][i])
                            for i in range(self.state.num_joints)])
        self.state.set_all_force_positions(rand_pos)
        for i, fs in enumerate(self.fsliders):
            fs.set_val(rand_pos[i])

    def _smooth_reset(self, target_pos=None, duration=2.0):
        if target_pos is None:
            target_pos = np.array(config.DEFAULT_ANGLES)
        st = self.state.get_state()
        self._smooth_start = st['positions'].copy()
        self._smooth_target = target_pos
        self._smooth_duration = duration
        self._smooth_elapsed = 0.0
        self._smooth_dt = 0.02
        if not hasattr(self, '_smooth_timer'):
            self._smooth_timer = QTimer()
            self._smooth_timer.timeout.connect(self._smooth_step)
        self._smooth_timer.start(int(self._smooth_dt * 1000))

    def _smooth_step(self):
        self._smooth_elapsed += self._smooth_dt
        t = min(1.0, self._smooth_elapsed / self._smooth_duration)
        alpha = (1 - np.cos(np.pi * t)) / 2.0
        interp_pos = self._smooth_start + alpha * (self._smooth_target - self._smooth_start)
        self.state.set_all_targets(interp_pos)
        for i, s in enumerate(self.sliders):
            s.set_val(interp_pos[i])
        if t >= 1.0:
            self._smooth_timer.stop()

    def _smooth_reset_default(self):
        self._smooth_reset(np.array(config.DEFAULT_ANGLES), duration=2.0)

    def _smooth_zero(self):
        self._smooth_reset(np.zeros(self.state.num_joints), duration=2.0)

    def _sync_all_force_positions(self):
        st = self.state.get_state()
        self.state.set_all_force_positions(st['positions'])
        for i, fs in enumerate(self.fsliders):
            fs.set_val(st['positions'][i])

    def _reset_force_to_default(self, idx):
        self.fsliders[idx].set_val(config.DEFAULT_ANGLES[idx])
        self.state.set_force_position(idx, config.DEFAULT_ANGLES[idx])

    def _reset_all_gains(self):
        """Reset spinboxes to defaults. Use 'Set All' to apply."""
        for i, gw in enumerate(self.gain_widgets):
            gw.set_gains(config.KPS[i], config.KDS[i])

    def _set_all_gains_from_widgets(self):
        for gw in self.gain_widgets:
            self.state.set_kp_kd(gw.idx, gw.kp_spin.value(), gw.kd_spin.value())

    def _apply_heuristic_gains(self, kp, kd):
        """Only populate spinboxes with calculated gains. Use 'Set All' to apply."""
        for i, gw in enumerate(self.gain_widgets):
            gw.set_gains(kp[i], kd[i])

    def _update(self):
        if not self.state.running:
            return
        st = self.state.get_state()
        tr = self.state.get_time_range(self.time_window)

        self.pd_btn.setChecked(st['pd_enabled'])
        self.grav_btn.setChecked(st['gravity_comp_enabled'])
        self.cor_btn.setChecked(st['coriolis_comp_enabled'])

        for i, s in enumerate(self.sliders):
            s.update_cur(st['positions'][i])

        if self.sel_joint is not None:
            h = self.state.get_history(self.sel_joint, self.time_window)
            self.detail.update(st, h, tr)

        self._cnt += 1
        if self._cnt % 3 == 0:
            get_h = lambda i: self.state.get_history(i, self.time_window)
            self.overview.update(get_h, tr)
            self.torque_panel.update(get_h, tr)

        pd_s = "PD:ON" if st['pd_enabled'] else "PD:OFF"
        gv_s = "Gv:ON" if st['gravity_comp_enabled'] else "Gv:OFF"
        cr_s = "Cor:ON" if st['coriolis_comp_enabled'] else "Cor:OFF"
        self.status.setText(f"t={st['sim_time']:.1f}s | {pd_s} | {gv_s} | {cr_s} | win={self.time_window:.0f}s")

    def closeEvent(self, e):
        self.state.running = False
        self.sim.stop()
        e.accept()


# ============================================================================
# Main
# ============================================================================

def main():
    n = len(config.DEFAULT_ANGLES)
    state = RobotState(n, 60.0, config.SIMULATE_DT)
    sim = SimulationController(state)
    sim.start()

    if not config.USE_GUI:
        print("GUI disabled. MuJoCo viewer only. Ctrl+C to exit.")
        try:
            while state.running and sim.viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        sim.stop()
        return

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(30, 30, 30))
    pal.setColor(QPalette.WindowText, QColor(224, 224, 224))
    pal.setColor(QPalette.Base, QColor(45, 45, 45))
    pal.setColor(QPalette.Text, QColor(224, 224, 224))
    pal.setColor(QPalette.Button, QColor(45, 45, 45))
    pal.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    pal.setColor(QPalette.Highlight, QColor(0, 168, 255))
    app.setPalette(pal)

    gui = RobotGUI(sim, state)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

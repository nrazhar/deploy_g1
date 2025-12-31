"""
GUI Widget Components for MuJoCo Robot Controller
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QLabel,
    QPushButton, QGroupBox, QDoubleSpinBox, QCheckBox, QComboBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import config


# ============================================================================
# Basic Slider Widgets
# ============================================================================

class JointSlider(QWidget):
    """Slider for setting target joint position."""
    value_changed = pyqtSignal(int, float)

    def __init__(self, idx, name, default, vmin, vmax):
        super().__init__()
        self.idx, self.vmin, self.vmax = idx, vmin, vmax
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(3)

        self.name_label = QLabel(f"{name}:")
        self.name_label.setFixedWidth(95)
        self.name_label.setStyleSheet("color:#e0e0e0;font-size:10px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._v2s(default))
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{height:5px;background:#2d2d2d;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#00a8ff;width:10px;margin:-3px 0;border-radius:5px;}")

        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setDecimals(3)
        self.spin.setValue(default)
        self.spin.valueChanged.connect(self._on_spin)
        self.spin.setFixedWidth(65)
        self.spin.setStyleSheet("background:#2d2d2d;color:#0f8;border:1px solid #444;font-size:9px;")

        self.cur = QLabel("0.000")
        self.cur.setFixedWidth(50)
        self.cur.setStyleSheet("color:#fa0;font-size:9px;")
        self.cur.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.cur)

    def _v2s(self, v):
        return int((max(self.vmin, min(self.vmax, v)) - self.vmin) / (self.vmax - self.vmin) * 1000)

    def _s2v(self, s):
        return self.vmin + s / 1000 * (self.vmax - self.vmin)

    def _on_slider(self, s):
        v = self._s2v(s)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.value_changed.emit(self.idx, v)

    def _on_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._v2s(v))
        self.slider.blockSignals(False)
        self.value_changed.emit(self.idx, v)

    def update_cur(self, v):
        self.cur.setText(f"{v:.3f}")

    def set_val(self, v):
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._v2s(v))
        self.spin.setValue(v)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)


class ForcePositionSlider(QWidget):
    """Slider for forcing joint position with enable checkbox and sync button."""
    value_changed = pyqtSignal(int, float)
    enabled_changed = pyqtSignal(int, bool)

    def __init__(self, idx, name, vmin, vmax, get_current_fn):
        super().__init__()
        self.idx, self.vmin, self.vmax = idx, vmin, vmax
        self.get_current_fn = get_current_fn
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(3)

        self.cb = QCheckBox()
        self.cb.stateChanged.connect(lambda s: self.enabled_changed.emit(idx, s == Qt.Checked))

        self.name_label = QLabel(f"{name}:")
        self.name_label.setFixedWidth(80)
        self.name_label.setStyleSheet("color:#e0e0e0;font-size:10px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{height:5px;background:#2d2d2d;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#f60;width:10px;margin:-3px 0;border-radius:5px;}")

        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setDecimals(3)
        self.spin.setValue((vmin + vmax) / 2)
        self.spin.valueChanged.connect(self._on_spin)
        self.spin.setFixedWidth(65)
        self.spin.setStyleSheet("background:#2d2d2d;color:#f60;border:1px solid #444;font-size:9px;")

        self.sync_btn = QPushButton("⟳")
        self.sync_btn.setFixedSize(20, 18)
        self.sync_btn.setToolTip("Copy current position")
        self.sync_btn.setStyleSheet("QPushButton{background:#555;color:#fff;border:none;font-size:10px;}QPushButton:hover{background:#666;}")
        self.sync_btn.clicked.connect(self._sync_current)

        self.rst_btn = QPushButton("R")
        self.rst_btn.setFixedSize(18, 18)
        self.rst_btn.setToolTip("Reset to default")
        self.rst_btn.setStyleSheet("QPushButton{background:#555;color:#fff;border:none;font-size:9px;}QPushButton:hover{background:#666;}")

        layout.addWidget(self.cb)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.sync_btn)
        layout.addWidget(self.rst_btn)

    def _v2s(self, v):
        return int((max(self.vmin, min(self.vmax, v)) - self.vmin) / (self.vmax - self.vmin) * 1000)

    def _s2v(self, s):
        return self.vmin + s / 1000 * (self.vmax - self.vmin)

    def _on_slider(self, s):
        v = self._s2v(s)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.value_changed.emit(self.idx, v)

    def _on_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._v2s(v))
        self.slider.blockSignals(False)
        self.value_changed.emit(self.idx, v)

    def _sync_current(self):
        v = self.get_current_fn(self.idx)
        self.set_val(v)
        self.value_changed.emit(self.idx, v)

    def set_val(self, v):
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._v2s(v))
        self.spin.setValue(v)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)


# ============================================================================
# Enhanced Gain Control Widget
# ============================================================================

class GainControlWidget(QWidget):
    """
    Compact gain control with:
    - Kp/Kd with +/-10, +/-50 buttons and Set button
    - Armature inertia with double/half/Set buttons
    - Damping ratio with Set button
    """
    gains_changed = pyqtSignal(int, float, float)  # idx, kp, kd
    armature_changed = pyqtSignal(int, float)  # idx, armature
    zeta_changed = pyqtSignal(int, float)  # idx, damping ratio

    def __init__(self, idx, name, kp_init, kd_init, kp_default, kd_default, armature_init, zeta_init):
        super().__init__()
        self.idx = idx
        self.kp_default, self.kd_default = kp_default, kd_default
        self.armature_default = armature_init
        self.zeta_default = zeta_init

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(1)

        # Joint name
        lbl = QLabel(f"{name}:")
        lbl.setFixedWidth(70)
        lbl.setStyleSheet("color:#e0e0e0;font-size:9px;")
        layout.addWidget(lbl)

        btn_css = "QPushButton{background:#444;color:#fff;border:none;font-size:7px;padding:0px;}QPushButton:hover{background:#555;}"
        btn_css_set = "QPushButton{background:#2a7;color:#fff;border:none;font-size:7px;padding:0px;}QPushButton:hover{background:#3b8;}"

        # Kp section
        self.kp_spin = QDoubleSpinBox()
        self.kp_spin.setRange(0, 1000)
        self.kp_spin.setDecimals(1)
        self.kp_spin.setValue(kp_init)
        self.kp_spin.setPrefix("Kp:")
        self.kp_spin.setFixedWidth(62)
        self.kp_spin.setStyleSheet("background:#2d2d2d;color:#4af;border:1px solid #444;font-size:8px;")
        self.kp_spin.setKeyboardTracking(False)
        layout.addWidget(self.kp_spin)

        # Kp +/- buttons (compact: -50 -10 +10 +50)
        for delta in [-50, -10, 10, 50]:
            b = QPushButton(f"{delta:+}")
            b.setFixedSize(22, 15)
            b.setStyleSheet(btn_css)
            b.clicked.connect(lambda _, d=delta: self._adjust_kp(d))
            layout.addWidget(b)

        # Kd section
        self.kd_spin = QDoubleSpinBox()
        self.kd_spin.setRange(0, 100)
        self.kd_spin.setDecimals(2)
        self.kd_spin.setValue(kd_init)
        self.kd_spin.setPrefix("Kd:")
        self.kd_spin.setFixedWidth(58)
        self.kd_spin.setStyleSheet("background:#2d2d2d;color:#f4a;border:1px solid #444;font-size:8px;")
        self.kd_spin.setKeyboardTracking(False)
        layout.addWidget(self.kd_spin)

        # Kd +/- buttons (compact: -10 +10)
        for delta in [-10, 10]:
            b = QPushButton(f"{delta:+}")
            b.setFixedSize(20, 15)
            b.setStyleSheet(btn_css)
            b.clicked.connect(lambda _, d=delta: self._adjust_kd(d))
            layout.addWidget(b)

        # Auto Kd = Kp/20 button
        auto_btn = QPushButton("/20")
        auto_btn.setFixedSize(22, 15)
        auto_btn.setToolTip("Kd=Kp/20")
        auto_btn.setStyleSheet(btn_css)
        auto_btn.clicked.connect(self._auto_kd)
        layout.addWidget(auto_btn)

        # Set button for Kp/Kd (right next to +/- buttons)
        set_btn = QPushButton("Set")
        set_btn.setFixedSize(22, 15)
        set_btn.setToolTip("Apply Kp/Kd")
        set_btn.setStyleSheet(btn_css_set)
        set_btn.clicked.connect(self._emit_gains)
        layout.addWidget(set_btn)

        # Armature inertia
        self.arm_spin = QDoubleSpinBox()
        self.arm_spin.setRange(0.001, 10)
        self.arm_spin.setDecimals(3)
        self.arm_spin.setValue(armature_init)
        self.arm_spin.setPrefix("I:")
        self.arm_spin.setFixedWidth(58)
        self.arm_spin.setStyleSheet("background:#2d2d2d;color:#fa0;border:1px solid #444;font-size:8px;")
        self.arm_spin.setKeyboardTracking(False)
        layout.addWidget(self.arm_spin)

        # Half/Double armature
        half_btn = QPushButton("½")
        half_btn.setFixedSize(14, 15)
        half_btn.setStyleSheet(btn_css)
        half_btn.clicked.connect(lambda: self.arm_spin.setValue(self.arm_spin.value() / 2))
        layout.addWidget(half_btn)

        dbl_btn = QPushButton("2×")
        dbl_btn.setFixedSize(16, 15)
        dbl_btn.setStyleSheet(btn_css)
        dbl_btn.clicked.connect(lambda: self.arm_spin.setValue(self.arm_spin.value() * 2))
        layout.addWidget(dbl_btn)

        # Set button for Armature
        arm_set_btn = QPushButton("Set")
        arm_set_btn.setFixedSize(22, 15)
        arm_set_btn.setToolTip("Apply armature")
        arm_set_btn.setStyleSheet(btn_css_set)
        arm_set_btn.clicked.connect(lambda: self.armature_changed.emit(self.idx, self.arm_spin.value()))
        layout.addWidget(arm_set_btn)

        # Damping ratio (zeta)
        self.zeta_spin = QDoubleSpinBox()
        self.zeta_spin.setRange(0.1, 10)
        self.zeta_spin.setDecimals(2)
        self.zeta_spin.setValue(zeta_init)
        self.zeta_spin.setPrefix("ζ:")
        self.zeta_spin.setFixedWidth(48)
        self.zeta_spin.setStyleSheet("background:#2d2d2d;color:#8f8;border:1px solid #444;font-size:8px;")
        self.zeta_spin.setKeyboardTracking(False)
        layout.addWidget(self.zeta_spin)

        # Set button for zeta
        zeta_set_btn = QPushButton("Set")
        zeta_set_btn.setFixedSize(22, 15)
        zeta_set_btn.setToolTip("Apply ζ")
        zeta_set_btn.setStyleSheet(btn_css_set)
        zeta_set_btn.clicked.connect(lambda: self.zeta_changed.emit(self.idx, self.zeta_spin.value()))
        layout.addWidget(zeta_set_btn)

        # Reset button
        rst_btn = QPushButton("R")
        rst_btn.setFixedSize(14, 15)
        rst_btn.setToolTip("Reset all")
        rst_btn.setStyleSheet(btn_css)
        rst_btn.clicked.connect(self._reset)
        layout.addWidget(rst_btn)

    def _adjust_kp(self, delta):
        new_val = max(0, self.kp_spin.value() + delta)
        self.kp_spin.setValue(new_val)

    def _adjust_kd(self, delta):
        new_val = max(0, self.kd_spin.value() + delta)
        self.kd_spin.setValue(new_val)

    def _auto_kd(self):
        self.kd_spin.setValue(self.kp_spin.value() / 20.0)

    def _emit_gains(self):
        self.gains_changed.emit(self.idx, self.kp_spin.value(), self.kd_spin.value())

    def _reset(self):
        self.kp_spin.setValue(self.kp_default)
        self.kd_spin.setValue(self.kd_default)
        self.arm_spin.setValue(self.armature_default)
        self.zeta_spin.setValue(self.zeta_default)
        self._emit_gains()

    def set_gains(self, kp, kd):
        self.kp_spin.blockSignals(True)
        self.kd_spin.blockSignals(True)
        self.kp_spin.setValue(kp)
        self.kd_spin.setValue(kd)
        self.kp_spin.blockSignals(False)
        self.kd_spin.blockSignals(False)

    def get_armature(self):
        return self.arm_spin.value()

    def get_zeta(self):
        return self.zeta_spin.value()


# ============================================================================
# Plot Widget with Zoom and Pan
# ============================================================================

class PlotWidget(QWidget):
    """Real-time plot with zoom and pan controls."""

    def __init__(self, title=""):
        super().__init__()
        self.default_y_min, self.default_y_max = -1, 1
        self.zoom = 1.0
        self.y_offset = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 1, 4, 0)
        lbl = QLabel(title)
        lbl.setStyleSheet("color:#aaa;font-weight:bold;font-size:10px;")
        ctrl.addWidget(lbl)
        ctrl.addStretch()

        btn_css = "QPushButton{background:#444;color:#fff;border:none;padding:1px 5px;font-size:9px;}QPushButton:hover{background:#555;}"

        for txt, fn in [("+", self._zin), ("-", self._zout), ("R", self._zrst)]:
            b = QPushButton(txt)
            b.setFixedSize(18, 16)
            b.setStyleSheet(btn_css)
            b.clicked.connect(fn)
            if txt == "R":
                b.setToolTip("Reset zoom")
            ctrl.addWidget(b)

        up_btn = QPushButton("▲")
        up_btn.setFixedSize(18, 16)
        up_btn.setStyleSheet(btn_css)
        up_btn.setToolTip("Move up")
        up_btn.clicked.connect(self._pan_up)
        ctrl.addWidget(up_btn)

        dn_btn = QPushButton("▼")
        dn_btn.setFixedSize(18, 16)
        dn_btn.setStyleSheet(btn_css)
        dn_btn.setToolTip("Move down")
        dn_btn.clicked.connect(self._pan_down)
        ctrl.addWidget(dn_btn)

        layout.addLayout(ctrl)

        self.pw = pg.PlotWidget()
        self.pw.setBackground('#1a1a1a')
        self.pw.showGrid(x=True, y=True, alpha=0.3)
        self.pw.setLabel('bottom', '', color='#666', size='7pt')
        self.pw.addLegend(offset=(5, 5))
        self.pw.setMouseEnabled(x=False, y=False)
        self.pw.enableAutoRange('x', False)
        self.pw.enableAutoRange('y', False)
        layout.addWidget(self.pw)
        self.curves = {}

    def set_y_range(self, ymin, ymax):
        self.default_y_min, self.default_y_max = ymin, ymax
        self.y_offset = 0.0
        self._apply_y()

    def _apply_y(self):
        base_center = (self.default_y_min + self.default_y_max) / 2
        center = base_center + self.y_offset
        half = (self.default_y_max - self.default_y_min) / 2 / self.zoom
        self.pw.setYRange(center - half * 1.05, center + half * 1.05, padding=0)

    def _zin(self):
        self.zoom = min(self.zoom * 1.5, 50)
        self._apply_y()

    def _zout(self):
        self.zoom = max(self.zoom / 1.5, 0.1)
        self._apply_y()

    def _zrst(self):
        self.zoom = 1.0
        self.y_offset = 0.0
        self._apply_y()

    def _pan_up(self):
        step = (self.default_y_max - self.default_y_min) / self.zoom * 0.2
        self.y_offset += step
        self._apply_y()

    def _pan_down(self):
        step = (self.default_y_max - self.default_y_min) / self.zoom * 0.2
        self.y_offset -= step
        self._apply_y()

    def set_x_range(self, xmin, xmax):
        self.pw.setXRange(xmin, xmax, padding=0)

    def add_curve(self, name, color, w=2):
        self.curves[name] = self.pw.plot([], [], pen=pg.mkPen(color=color, width=w), name=name)

    def update_curve(self, name, x, y):
        if name in self.curves and len(x):
            self.curves[name].setData(x, y)

    def clear_all(self):
        for c in self.curves.values():
            c.setData([], [])


# ============================================================================
# Detail and Overview Panels
# ============================================================================

class JointDetailPanel(QWidget):
    """Panel showing detailed plots for selected joint."""

    def __init__(self, limits):
        super().__init__()
        self.limits = limits
        self.sel = None

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self.title = QLabel("Select joint")
        self.title.setStyleSheet("font-size:13px;font-weight:bold;color:#0af;")
        layout.addWidget(self.title)

        sw = QWidget()
        sl = QGridLayout(sw)
        sl.setSpacing(2)
        labels = ['Pos:', 'Tgt:', 'Vel:', 'Acc:', 'Kp:', 'Kd:', 'τPD:', 'τGrav:', 'τCor:', 'τTot:']
        self.vals = {}
        cols = 5  # 5 columns per row
        for i, t in enumerate(labels):
            l = QLabel(t)
            l.setStyleSheet("color:#888;font-size:9px;")
            v = QLabel("--")
            # Color Kp/Kd differently
            if t in ['Kp:', 'Kd:']:
                v.setStyleSheet("color:#4af;font-size:9px;")
            else:
                v.setStyleSheet("color:#0f8;font-size:9px;")
            sl.addWidget(l, i // cols, (i % cols) * 2)
            sl.addWidget(v, i // cols, (i % cols) * 2 + 1)
            self.vals[t] = v
        layout.addWidget(sw)

        self.pos_plot = PlotWidget("Position (rad)")
        self.pos_plot.add_curve("Actual", '#0f8')
        self.pos_plot.add_curve("Target", '#f80', 1)
        layout.addWidget(self.pos_plot)

        self.vel_plot = PlotWidget("Velocity (rad/s)")
        self.vel_plot.add_curve("Vel", '#0af')
        layout.addWidget(self.vel_plot)

        self.acc_plot = PlotWidget("Acceleration (rad/s²)")
        self.acc_plot.add_curve("Acc", '#f0f')
        layout.addWidget(self.acc_plot)

        self.tau_plot = PlotWidget("Torque (Nm)")
        self.tau_plot.add_curve("PD", '#f44')
        self.tau_plot.add_curve("Grav", '#4f4')
        self.tau_plot.add_curve("Cor", '#f0f')
        self.tau_plot.add_curve("Tot", '#ff4', 1)
        layout.addWidget(self.tau_plot)

    def set_joint(self, name, idx):
        self.sel = idx
        self.title.setText(f"Joint: {name}")
        self.pos_plot.set_y_range(self.limits['pos_min'][idx], self.limits['pos_max'][idx])
        self.vel_plot.set_y_range(-self.limits['vel_limits'][idx], self.limits['vel_limits'][idx])
        self.acc_plot.set_y_range(-self.limits['acc_limits'][idx], self.limits['acc_limits'][idx])
        self.tau_plot.set_y_range(-self.limits['torque_limits'][idx], self.limits['torque_limits'][idx])
        for p in [self.pos_plot, self.vel_plot, self.acc_plot, self.tau_plot]:
            p.clear_all()

    def update(self, state, hist, tr):
        if self.sel is None:
            return
        i = self.sel
        self.vals['Pos:'].setText(f"{state['positions'][i]:.3f}")
        self.vals['Tgt:'].setText(f"{state['target_positions'][i]:.3f}")
        self.vals['Vel:'].setText(f"{state['velocities'][i]:.3f}")
        self.vals['Acc:'].setText(f"{state['accelerations'][i]:.1f}")
        self.vals['Kp:'].setText(f"{state['kp'][i]:.1f}")
        self.vals['Kd:'].setText(f"{state['kd'][i]:.2f}")
        self.vals['τPD:'].setText(f"{state['torque_pd'][i]:.2f}")
        self.vals['τGrav:'].setText(f"{state['torque_gravity'][i]:.2f}")
        self.vals['τCor:'].setText(f"{state['torque_coriolis'][i]:.2f}")
        self.vals['τTot:'].setText(f"{state['torque_total'][i]:.2f}")

        for p in [self.pos_plot, self.vel_plot, self.acc_plot, self.tau_plot]:
            p.set_x_range(*tr)
        t = hist['time']
        self.pos_plot.update_curve('Actual', t, hist['position'])
        self.pos_plot.update_curve('Target', t, hist['target_position'])
        self.vel_plot.update_curve('Vel', t, hist['velocity'])
        self.acc_plot.update_curve('Acc', t, hist['acceleration'])
        self.tau_plot.update_curve('PD', t, hist['torque_pd'])
        self.tau_plot.update_curve('Grav', t, hist['torque_gravity'])
        self.tau_plot.update_curve('Cor', t, hist['torque_coriolis'])
        self.tau_plot.update_curve('Tot', t, hist['torque_total'])


def _gen_colors(n):
    colors = []
    for i in range(n):
        h = (i * 360 / n) % 360
        c, m = 0.72, 0.18
        x = c * (1 - abs((h / 60) % 2 - 1))
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        colors.append((int((r+m)*255), int((g+m)*255), int((b+m)*255)))
    return colors


class OverviewPanel(QWidget):
    def __init__(self, n, names, limits):
        super().__init__()
        self.n, self.names = n, names
        layout = QVBoxLayout(self)
        colors = _gen_colors(n)

        self.pos_plot = PlotWidget("All Positions")
        self.pos_plot.set_y_range(np.min(limits['pos_min']), np.max(limits['pos_max']))
        self.vel_plot = PlotWidget("All Velocities")
        self.vel_plot.set_y_range(-np.max(limits['vel_limits']), np.max(limits['vel_limits']))
        self.acc_plot = PlotWidget("All Accelerations")
        self.acc_plot.set_y_range(-np.max(limits['acc_limits']), np.max(limits['acc_limits']))
        self.tau_plot = PlotWidget("All Torques")
        self.tau_plot.set_y_range(-np.max(limits['torque_limits']), np.max(limits['torque_limits']))

        for p in [self.pos_plot, self.vel_plot, self.acc_plot, self.tau_plot]:
            layout.addWidget(p, 1)

        for i, nm in enumerate(names):
            c = colors[i]
            self.pos_plot.add_curve(nm, c, 1)
            self.vel_plot.add_curve(nm, c, 1)
            self.acc_plot.add_curve(nm, c, 1)
            self.tau_plot.add_curve(nm, c, 1)

    def update(self, get_hist, tr):
        for p in [self.pos_plot, self.vel_plot, self.acc_plot, self.tau_plot]:
            p.set_x_range(*tr)
        for i in range(self.n):
            h = get_hist(i)
            t, nm = h['time'], self.names[i]
            self.pos_plot.update_curve(nm, t, h['position'])
            self.vel_plot.update_curve(nm, t, h['velocity'])
            self.acc_plot.update_curve(nm, t, h['acceleration'])
            self.tau_plot.update_curve(nm, t, h['torque_total'])


class TorquePanel(QWidget):
    def __init__(self, n, names, limits):
        super().__init__()
        self.n, self.names = n, names
        layout = QVBoxLayout(self)
        colors = _gen_colors(n)
        tmax = np.max(limits['torque_limits'])

        self.pd_plot = PlotWidget("PD Torques")
        self.pd_plot.set_y_range(-tmax, tmax)
        self.grav_plot = PlotWidget("Gravity Comp")
        self.grav_plot.set_y_range(-tmax, tmax)
        self.cor_plot = PlotWidget("Coriolis/Centrifugal")
        self.cor_plot.set_y_range(-tmax, tmax)

        layout.addWidget(self.pd_plot)
        layout.addWidget(self.grav_plot)
        layout.addWidget(self.cor_plot)

        for i, nm in enumerate(names):
            c = colors[i]
            self.pd_plot.add_curve(nm, c, 1)
            self.grav_plot.add_curve(nm, c, 1)
            self.cor_plot.add_curve(nm, c, 1)

    def update(self, get_hist, tr):
        self.pd_plot.set_x_range(*tr)
        self.grav_plot.set_x_range(*tr)
        self.cor_plot.set_x_range(*tr)
        for i in range(self.n):
            h = get_hist(i)
            t, nm = h['time'], self.names[i]
            self.pd_plot.update_curve(nm, t, h['torque_pd'])
            self.grav_plot.update_curve(nm, t, h['torque_gravity'])
            self.cor_plot.update_curve(nm, t, h['torque_coriolis'])


# ============================================================================
# Gain Heuristic Control Panel
# ============================================================================

class GainHeuristicPanel(QWidget):
    """Panel for applying heuristic gain calculations globally."""
    apply_gains = pyqtSignal(object, object)  # kp_array, kd_array

    def __init__(self, limits, num_joints, get_armatures_fn, get_zetas_fn):
        super().__init__()
        self.limits = limits
        self.n = num_joints
        self.get_armatures_fn = get_armatures_fn
        self.get_zetas_fn = get_zetas_fn

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Torque-Limit (Conservative)", "Torque-Limit (Aggressive)", "Inertia-Based"])
        self.method_combo.setStyleSheet("background:#2d2d2d;color:#fff;border:1px solid #444;font-size:10px;")
        self.method_combo.setFixedWidth(180)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        layout.addLayout(method_layout)

        # Global parameters
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("ω(Hz):"))
        self.omega_spin = QDoubleSpinBox()
        self.omega_spin.setRange(1, 100)
        self.omega_spin.setValue(config.DEFAULT_OMEGA)
        self.omega_spin.setStyleSheet("background:#2d2d2d;color:#0f8;border:1px solid #444;font-size:9px;")
        self.omega_spin.setFixedWidth(55)
        self.omega_spin.setKeyboardTracking(False)
        param_layout.addWidget(self.omega_spin)

        param_layout.addWidget(QLabel("Global ζ:"))
        self.global_zeta_spin = QDoubleSpinBox()
        self.global_zeta_spin.setRange(0.1, 10)
        self.global_zeta_spin.setValue(2.0)
        self.global_zeta_spin.setStyleSheet("background:#2d2d2d;color:#0f8;border:1px solid #444;font-size:9px;")
        self.global_zeta_spin.setFixedWidth(50)
        self.global_zeta_spin.setKeyboardTracking(False)
        param_layout.addWidget(self.global_zeta_spin)

        # Button to update all zetas globally
        update_zeta_btn = QPushButton("Set All ζ")
        update_zeta_btn.setToolTip("Set all joint damping ratios to global value")
        update_zeta_btn.setStyleSheet("background:#555;font-size:9px;")
        update_zeta_btn.setFixedWidth(55)
        update_zeta_btn.clicked.connect(self._update_all_zetas)
        param_layout.addWidget(update_zeta_btn)
        param_layout.addStretch()
        layout.addLayout(param_layout)

        # Info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color:#888;font-size:9px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Calculate button (just updates spinboxes, doesn't apply to state)
        calc_btn = QPushButton("🔢 Calculate Heuristic Gains")
        calc_btn.clicked.connect(self._calculate)
        calc_btn.setStyleSheet("background:#555;color:#fff;font-size:10px;padding:5px;")
        calc_btn.setToolTip("Calculate gains and populate spinboxes. Use 'Set All' to apply.")
        layout.addWidget(calc_btn)

        self.method_combo.currentIndexChanged.connect(self._update_info)
        self._update_info()

        # Store callback for updating zetas
        self.update_zetas_callback = None

    def set_update_zetas_callback(self, fn):
        self.update_zetas_callback = fn

    def _update_info(self):
        idx = self.method_combo.currentIndex()
        if idx == 0:
            self.info_label.setText("Kp = τ_max / |q_max - q_min|\nKd = Kp / 20")
        elif idx == 1:
            self.info_label.setText("Kp = τ_max / (0.5 × range)\nKd = Kp / 20")
        else:
            self.info_label.setText("Kp = I × (2πω)²\nKd = 2 × I × ζ × (2πω)\nUses per-joint I and ζ")

    def _update_all_zetas(self):
        if self.update_zetas_callback:
            self.update_zetas_callback(self.global_zeta_spin.value())

    def _calculate(self):
        """Calculate heuristic gains and emit signal to update spinboxes only."""
        from utils import GainHeuristics
        idx = self.method_combo.currentIndex()
        kp = np.zeros(self.n)
        kd = np.zeros(self.n)

        armatures = self.get_armatures_fn()
        zetas = self.get_zetas_fn()

        for i in range(self.n):
            if idx in [0, 1]:
                conservative = (idx == 0)
                kp[i] = GainHeuristics.compute_kp_from_torque_limit(
                    self.limits['torque_limits'][i], self.limits['pos_range'][i], conservative)
                kd[i] = GainHeuristics.compute_kd_from_kp(kp[i])
            else:
                kp[i], kd[i] = GainHeuristics.compute_gains_from_inertia(
                    armatures[i], self.omega_spin.value(), zetas[i])

        self.apply_gains.emit(kp, kd)  # Only updates spinboxes, not state

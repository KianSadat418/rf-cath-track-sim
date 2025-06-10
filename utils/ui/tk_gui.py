import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from ..core.em_functions import biot_savart_loop, compute_emf
from ..core.localization_solver import (
    generate_measured_emf,
    estimate_rx_pose,
    compute_rmse,
)

class Arrow3D(FancyArrowPatch):
    """Utility class for drawing arrows in 3D plots."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class CatheterLocalizationApp:
    """Tkinter GUI application for RF catheter localization."""

    def __init__(self, master):
        self.master = master
        self.master.title("RF Catheter Localization")

        # Simulation parameters
        self.coil_radius = 0.05
        self.num_segments = 50
        self.current = 1.0
        self.frequency = 10000
        self.num_turns = 100
        self.area = np.pi * (0.01) ** 2

        self.tx_coils = [
            {"center": np.array([-0.1, -0.05, -0.08]), "normal": np.array([0, 0, 1]), "color": "r"},
            {"center": np.array([0.0, 0.1, -0.08]), "normal": np.array([0, 0, 1]), "color": "g"},
            {"center": np.array([0.1, -0.05, -0.08]), "normal": np.array([0, 0, 1]), "color": "b"},
        ]

        self.rx_center = np.array([0.1, 0.1, 0.1])
        self.rx_axis = np.array([1.0, 0.0, 0.0])

        self._build_ui()
        self.update_plot()

    def _build_ui(self):
        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=6)

        slider_frame = tk.Frame(self.master)
        slider_frame.grid(row=0, column=1, sticky="n")

        slider_len = 300
        self.s_x = tk.Scale(slider_frame, from_=-0.2, to=0.2, resolution=0.01, length=slider_len, label="X", orient=tk.HORIZONTAL, command=self.on_slider)
        self.s_y = tk.Scale(slider_frame, from_=-0.2, to=0.2, resolution=0.01, length=slider_len, label="Y", orient=tk.HORIZONTAL, command=self.on_slider)
        self.s_z = tk.Scale(slider_frame, from_=-0.2, to=0.2, resolution=0.01, length=slider_len, label="Z", orient=tk.HORIZONTAL, command=self.on_slider)
        self.s_theta = tk.Scale(slider_frame, from_=0, to=2*np.pi, resolution=0.05, length=slider_len, label="theta", orient=tk.HORIZONTAL, command=self.on_slider)
        self.s_phi = tk.Scale(slider_frame, from_=0, to=np.pi, resolution=0.05, length=slider_len, label="phi", orient=tk.HORIZONTAL, command=self.on_slider)

        self.s_x.set(self.rx_center[0])
        self.s_y.set(self.rx_center[1])
        self.s_z.set(self.rx_center[2])
        self.s_theta.set(0.0)
        self.s_phi.set(np.pi/2)

        for i, s in enumerate([self.s_x, self.s_y, self.s_z, self.s_theta, self.s_phi]):
            s.grid(row=i, column=0, sticky="ew")

        self.reset_btn = tk.Button(slider_frame, text="Reset", command=self.reset)
        self.reset_btn.grid(row=5, column=0, sticky="ew", pady=5)

        emf_frame = tk.LabelFrame(self.master, text="Display EMF")
        emf_frame.grid(row=1, column=1, sticky="nw", pady=5)

        self.emf_vars = [tk.BooleanVar(value=True) for _ in range(6)]
        self.emf_labels = []
        idx = 0
        for r in range(2):
            for t in range(3):
                cb = tk.Checkbutton(
                    emf_frame,
                    text=f"Rx{r+1}-Tx{t+1}",
                    variable=self.emf_vars[idx],
                    command=self.update_plot,
                )
                cb.grid(row=idx, column=0, sticky="w")
                lbl = tk.Label(emf_frame, text="")
                lbl.grid(row=idx, column=1, sticky="w")
                self.emf_labels.append(lbl)
                idx += 1

        self.info_label = tk.Label(self.master, text="")
        self.info_label.grid(row=6, column=0, columnspan=2)

    def on_slider(self, value):
        self.master.after_idle(self.update_plot)

    def reset(self):
        self.s_x.set(0.1)
        self.s_y.set(0.1)
        self.s_z.set(0.1)
        self.s_theta.set(0.0)
        self.s_phi.set(np.pi/2)
        self.update_plot()

    def draw_coil(self, center, normal, color, radius=None, num_points=30):
        if radius is None:
            radius = self.coil_radius
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)
        pts = np.vstack((x, y, z)).T
        if not np.allclose(normal, [0, 0, 1]):
            v = np.cross([0, 0, 1], normal)
            c = np.dot([0, 0, 1], normal)
            k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + k + k @ k * (1 / (1 + c + 1e-10))
            pts = (R @ pts.T).T
        pts = pts + center
        pts = np.vstack((pts, pts[0]))
        return pts

    def update_plot(self):
        self.rx_center = np.array([self.s_x.get(), self.s_y.get(), self.s_z.get()])
        theta = self.s_theta.get()
        phi = self.s_phi.get()
        self.rx_axis = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        self.rx_axis = self.rx_axis / np.linalg.norm(self.rx_axis)

        self.ax.clear()

        for i, coil in enumerate(self.tx_coils):
            pts = self.draw_coil(coil["center"], coil["normal"], coil["color"])
            self.ax.plot(pts[:,0], pts[:,1], pts[:,2], color=coil["color"], label=f"Tx {i+1}")
            arrow = Arrow3D(
                [coil["center"][0], coil["center"][0] + 0.1 * coil["normal"][0]],
                [coil["center"][1], coil["center"][1] + 0.1 * coil["normal"][1]],
                [coil["center"][2], coil["center"][2] + 0.1 * coil["normal"][2]],
                mutation_scale=15,
                lw=2,
                arrowstyle="-|>",
                color=coil["color"],
            )
            self.ax.add_artist(arrow)

        d = 0.06
        rx1 = self.rx_center - 0.5 * d * self.rx_axis
        rx2 = self.rx_center + 0.5 * d * self.rx_axis

        p1 = self.draw_coil(rx1, self.rx_axis, "m", radius=0.01)
        p2 = self.draw_coil(rx2, self.rx_axis, "c", radius=0.01)
        self.ax.plot(p1[:,0], p1[:,1], p1[:,2], color="m", lw=2)
        self.ax.plot(p2[:,0], p2[:,1], p2[:,2], color="c", lw=2)
        self.ax.plot([rx1[0], rx2[0]],[rx1[1], rx2[1]],[rx1[2], rx2[2]],"k--",alpha=0.5)

        measured_emf = generate_measured_emf(
            rx1,
            rx2,
            self.rx_axis,
            self.tx_coils,
            self.coil_radius,
            self.num_segments,
            self.current,
            self.area,
            self.num_turns,
            self.frequency,
            noise_std=1e-6,
        )

        for idx, emf in enumerate(measured_emf):
            if self.emf_vars[idx].get():
                self.emf_labels[idx].config(text=f"{emf:.2e} V")
            else:
                self.emf_labels[idx].config(text="")

        est_params = estimate_rx_pose(
            measured_emf,
            self.tx_coils,
            self.coil_radius,
            self.area,
            self.num_turns,
            self.frequency,
            initial_guess=None,
            d=d,
        )
        est_center = est_params[:3]
        th, ph = est_params[3], est_params[4]
        est_axis = np.array([
            np.sin(ph) * np.cos(th),
            np.sin(ph) * np.sin(th),
            np.cos(ph),
        ])
        est_axis = est_axis / np.linalg.norm(est_axis)
        est_rx1 = est_center - 0.5 * d * est_axis
        est_rx2 = est_center + 0.5 * d * est_axis
        p1e = self.draw_coil(est_rx1, est_axis, "y", radius=0.01)
        p2e = self.draw_coil(est_rx2, est_axis, "orange", radius=0.01)
        self.ax.plot(p1e[:,0], p1e[:,1], p1e[:,2], color="y", ls="--")
        self.ax.plot(p2e[:,0], p2e[:,1], p2e[:,2], color="orange", ls="--")
        self.ax.plot([est_rx1[0], est_rx2[0]],[est_rx1[1], est_rx2[1]],[est_rx1[2], est_rx2[2]], color="orange", ls="--", alpha=0.5)

        pos_rmse, ang_err = compute_rmse(self.rx_center, self.rx_axis, est_params)
        self.info_label.config(text=f"Position RMSE: {pos_rmse:.2f} mm  |  Angular Error: {ang_err:.1f} deg")

        self.ax.set_xlim([-0.2, 0.2])
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_zlim([-0.2, 0.2])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_title("RF Catheter Localization")
        self.canvas.draw_idle()

def run_gui():
    root = tk.Tk()
    app = CatheterLocalizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
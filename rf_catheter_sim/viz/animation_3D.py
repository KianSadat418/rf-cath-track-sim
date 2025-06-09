import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import biot_savart_loop, compute_emf

# Define Rx parameters
rx_center = np.array([0.01, 0.0, 0.02])
rx_axis = np.array([0, 0, 1])  # pointing in z
rx_radius = 0.003
rx_area = np.pi * rx_radius**2
rx_turns = 300
frequency = 29220  # Hz

# Define Rx motion: circular path in 3D
n_frames = 50
rx_positions_3d = [
    np.array([
        0.01 + 0.01 * np.sin(theta),
        0.01 * np.sin(2 * theta),
        0.02 + 0.01 * np.cos(theta)
    ])
    for theta in np.linspace(0, 2 * np.pi, n_frames)
]

tx_coils = [
    {"center": np.array([0.05, 0.0, 0.0]), "radius": 0.05, "normal": np.array([0, 0, 1]), "current": 0.1},
    {"center": np.array([-0.05, 0.05, 0.0]), "radius": 0.05, "normal": np.array([0, 0, 1]), "current": 0.1},
    {"center": np.array([-0.05, -0.05, 0.0]), "radius": 0.05, "normal": np.array([0, 0, 1]), "current": 0.1},
]

# 3D animation setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.1, 0.1)
ax.set_zlim(-0.05, 0.1)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('3D Rx Coil Trajectory and EMFs from 3 Tx Coils')

rx_dot, = ax.plot([], [], [], 'ro', label='Rx coil')
tx_dots = [
    ax.plot([tx["center"][0]], [tx["center"][1]], [tx["center"][2]], 'bo')[0]
    for tx in tx_coils
]
emf_texts = [ax.text2D(0.02, 0.9 - 0.05 * i, '', transform=ax.transAxes, fontsize=10) for i in range(len(tx_coils))]
ax.legend()

def init_3d():
    rx_dot.set_data([], [])
    rx_dot.set_3d_properties([])
    for text in emf_texts:
        text.set_text('')
    return [rx_dot] + emf_texts

def animate_3d(i):
    rx_pos = rx_positions_3d[i]
    rx_dot.set_data([rx_pos[0]], [rx_pos[1]])
    rx_dot.set_3d_properties([rx_pos[2]])

    for t, tx in enumerate(tx_coils):
        B = biot_savart_loop(rx_pos, tx["center"], tx["radius"], tx["normal"], current=tx["current"])
        emf = compute_emf(B, rx_axis, rx_area, rx_turns, frequency)
        emf_texts[t].set_text(f"Tx{t+1} EMF: {emf*1000:.2f} mV")

    return [rx_dot] + emf_texts

ani_3d = animation.FuncAnimation(fig, animate_3d, frames=n_frames, init_func=init_3d, blit=True, interval=100)
plt.show()

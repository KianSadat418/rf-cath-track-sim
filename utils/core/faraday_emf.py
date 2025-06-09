import numpy as np
from biot_savart import biot_savart_loop

# Reuse the biot_savart_loop and compute_emf functions from earlier
def compute_emf(B_field, rx_axis, area, num_turns, frequency):
    B_parallel = np.dot(B_field, rx_axis)
    emf_peak = 2 * np.pi * frequency * num_turns * area * B_parallel
    return emf_peak

# Define parameters
rx_center = np.array([0.01, 0.0, 0.02])
rx_axis = np.array([0, 0, 1])  # pointing in z
rx_radius = 0.003
rx_area = np.pi * rx_radius**2
rx_turns = 300
frequency = 29220  # Hz

# Define multiple Tx coils
tx_coils = [
    {"center": np.array([0.05, 0.0, 0.0]), "radius": 0.05, "normal": np.array([0, 0, 1]), "current": 0.1},
    {"center": np.array([-0.05, 0.05, 0.0]), "radius": 0.05, "normal": np.array([0, 0, 1]), "current": 0.1},
    {"center": np.array([-0.05, -0.05, 0.0]), "radius": 0.05, "normal": np.array([0, 0, 1]), "current": 0.1},
]

# Compute EMFs
emfs = []
for tx in tx_coils:
    B = biot_savart_loop(rx_center, tx["center"], tx["radius"], tx["normal"], current=tx["current"])
    emf = compute_emf(B, rx_axis, rx_area, rx_turns, frequency)
    emfs.append(emf)

print("Computed EMFs from multiple Tx coils:", emfs)

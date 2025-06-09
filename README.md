# RF Catheter Tracking Simulation

This repository contains a minimal Python prototype for simulating radio frequency (RF) catheter tracking.  The code explores how a small receive (Rx) coil placed on a catheter tip couples to larger transmit (Tx) coils using basic electromagnetic equations.  Only a handful of features are implemented while many modules are placeholders for future development.

## Repository structure

```
rf_catheter_sim/
  core/         Physics utilities (Biot–Savart, Faraday EMF, etc.)
  sim/          Empty stubs for simulation configuration
  ui/           Empty stub for UI code
  viz/          Simple visualization/animation scripts
main.py         Entry point placeholder
```

### Core modules
- **`biot_savart.py`** – Implements the Biot–Savart law to compute the magnetic field of a circular coil.  The function `biot_savart_loop` iterates over a discretized coil and sums the field contributions:

```python
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability

def biot_savart_loop(field_point, coil_center, coil_radius, coil_normal, current=1.0, num_segments=100):
    n = coil_normal / np.linalg.norm(coil_normal)
    if np.allclose(n, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(n, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    ...
    dB = MU_0 * current / (4 * np.pi) * np.cross(dl, r_vec) / (r_mag ** 3)
```

- **`faraday_emf.py`** – Example script using the Biot–Savart function to compute the induced EMF in an Rx coil from several Tx coils:

```python
rx_center = np.array([0.01, 0.0, 0.02])
rx_axis = np.array([0, 0, 1])
rx_radius = 0.003
rx_area = np.pi * rx_radius**2
rx_turns = 300
frequency = 29220  # Hz

# list of Tx coils ...
for tx in tx_coils:
    B = biot_savart_loop(rx_center, tx["center"], tx["radius"], tx["normal"], current=tx["current"])
    emf = compute_emf(B, rx_axis, rx_area, rx_turns, frequency)
```

- **`utils.py`** – Duplicates the Biot–Savart and EMF helper functions for convenient import by other modules.
- **`localization_solver.py`** – Currently empty placeholder.

### Visualization
The script `viz/animation_3D.py` animates a moving Rx coil in a synthetic 3‑Tx coil setup using Matplotlib's 3D tools.  Each frame computes the field and displays the induced voltage:

```python
rx_positions_3d = [
    np.array([
        0.01 + 0.01 * np.sin(theta),
        0.01 * np.sin(2 * theta),
        0.02 + 0.01 * np.cos(theta)
    ])
    for theta in np.linspace(0, 2 * np.pi, n_frames)
]
ani_3d = animation.FuncAnimation(fig, animate_3d, frames=n_frames, init_func=init_3d)
```

Running this module requires `numpy` and `matplotlib`.

## Usage
1. Install dependencies (Python ≥3.8, `numpy`, `matplotlib`).
2. Execute the animation script:

```bash
python rf_catheter_sim/viz/animation_3D.py
```

It will open a window showing the Rx coil trajectory and the instantaneous EMF values from three Tx coils.

## Status
The project is in a very early state; many folders only contain stubs.  The provided code is primarily for demonstration and experimentation with simple electromagnetic calculations and visualization.

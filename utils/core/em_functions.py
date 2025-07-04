import numpy as np

MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)

# Compute the magnetic field at a point in space due to a circular coil using the Biot-Savart law
def biot_savart_loop(field_point, coil_center, coil_radius, coil_normal, current=1.0, num_segments=100):
    
    n = coil_normal / np.linalg.norm(coil_normal)
    if np.allclose(n, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(n, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    B = np.zeros(3)
    dtheta = 2 * np.pi / num_segments

    for i in range(num_segments):
        theta1 = i * dtheta
        theta2 = (i + 1) * dtheta

        p1_local = coil_radius * (np.cos(theta1) * u + np.sin(theta1) * v)
        p2_local = coil_radius * (np.cos(theta2) * u + np.sin(theta2) * v)

        r1 = coil_center + p1_local
        r2 = coil_center + p2_local

        dl = r2 - r1
        r_mid = 0.5 * (r1 + r2)
        r_vec = field_point - r_mid
        r_mag = np.linalg.norm(r_vec)

        if r_mag == 0:
            continue

        dB = MU_0 * current / (4 * np.pi) * np.cross(dl, r_vec) / (r_mag ** 3)
        B += dB

    return B

# Compute the electromotive force (emf) induced in a coil by a time-varying magnetic field using Faraday's law of electromagnetic induction
def compute_emf(B_field, rx_axis, area, num_turns, frequency):
    B_parallel = np.dot(B_field, rx_axis)
    emf_peak = 2 * np.pi * frequency * num_turns * area * B_parallel
    return emf_peak

# Generate a synthetic EMF matrix for testing purposes
def generate_measured_emf(rx1_center, rx2_center, rx_axis, tx_coils, area, num_turns, freq, noise_std=1e-6):
    emf_matrix = np.zeros((2, len(tx_coils)))
    for i, rx_center in enumerate([rx1_center, rx2_center]):
        for j, tx in enumerate(tx_coils):
            B = biot_savart_loop(rx_center, tx['center'], 0.05, tx['normal'], 1.0, 50)
            emf = compute_emf(B, rx_axis, area, num_turns, freq)
            emf_matrix[i, j] = emf + np.random.normal(0, noise_std)
    return emf_matrix.flatten()
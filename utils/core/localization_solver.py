import numpy as np
from utils.core.em_functions import biot_savart_loop, compute_emf
from scipy.optimize import least_squares

def localization_cost(params, measured_emf, tx_coils, coil_radius, area, num_turns, freq, d=0.06):
    x, y, z, theta, phi = params
    rx_center = np.array([x, y, z])
    rx_axis = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    rx1_center = rx_center - 0.5 * d * rx_axis
    rx2_center = rx_center + 0.5 * d * rx_axis
    
    predicted_emf = []
    for rx in [rx1_center, rx2_center]:
        for tx in tx_coils:
            B = biot_savart_loop(rx, tx['center'], coil_radius, tx['normal'], 1.0, 50)
            emf = compute_emf(B, rx_axis, area, num_turns, freq)
            predicted_emf.append(emf)
    predicted_emf = np.array(predicted_emf)
    return measured_emf - predicted_emf

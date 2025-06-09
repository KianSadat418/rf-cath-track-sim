import numpy as np
from scipy.optimize import least_squares
from utils.core.em_functions import biot_savart_loop, compute_emf

# Direction vector for the receiver coil
def direction_vector(omega, phi):
    return np.array([
        np.cos(omega) * np.cos(phi),
        np.cos(omega) * np.sin(phi),
        -np.sin(omega)
    ])

# Localization cost function
def cost_function(params, measured_emfs, tx_configs, coil_params):
    x, y, z, omega, phi = params
    d = coil_params['rx_distance']
    A = coil_params['rx_area']
    N = coil_params['rx_turns']
    f = coil_params['frequency']

    dir_vec = direction_vector(omega, phi)
    pos1 = np.array([x, y, z])
    pos2 = pos1 + d * dir_vec

    emf_simulated = []

    for tx in tx_configs:
        B1 = biot_savart_loop(pos1, **tx)
        emf1 = compute_emf(B1, dir_vec, A, N, f)
        B2 = biot_savart_loop(pos2, **tx)
        emf2 = compute_emf(B2, dir_vec, A, N, f)
        emf_simulated.extend([emf1, emf2])

    return np.array(emf_simulated) - measured_emfs


# Localization solver using least squares optimization
def solve_localization(measured_emfs, tx_configs, coil_params, initial_guess):
    result = least_squares(
        cost_function,
        x0=initial_guess,
        args=(measured_emfs, tx_configs, coil_params),
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8
    )
    return result.x, result.cost


# Kalman filter for state estimation
def kalman_filter(prev_state, prev_vel, measured_state, P, Q, R, dt):
    A = np.block([
        [np.eye(5), dt * np.eye(5)],
        [np.zeros((5, 5)), np.eye(5)]
    ])
    H = np.block([np.eye(5), np.zeros((5, 5))])
    x_pred = A @ np.hstack([prev_state, prev_vel])
    P_pred = A @ P @ A.T + Q

    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_filt = x_pred + K @ (measured_state - H @ x_pred)
    P_new = (np.eye(10) - K @ H) @ P_pred

    new_state = x_filt[:5]
    new_vel = x_filt[5:]
    return new_state, new_vel, P_new


# Synthetic EMF generation for testing
def generate_synthetic_emf(true_params, tx_configs, coil_params, noise_std=0.0):
    x, y, z, omega, phi = true_params
    d = coil_params['rx_distance']
    A = coil_params['rx_area']
    N = coil_params['rx_turns']
    f = coil_params['frequency']

    dir_vec = direction_vector(omega, phi)
    pos1 = np.array([x, y, z])
    pos2 = pos1 + d * dir_vec

    synthetic_emfs = []
    for tx in tx_configs:
        B1 = biot_savart_loop(pos1, **tx)
        emf1 = compute_emf(B1, dir_vec, A, N, f)
        B2 = biot_savart_loop(pos2, **tx)
        emf2 = compute_emf(B2, dir_vec, A, N, f)
        synthetic_emfs.extend([emf1, emf2])

    if noise_std > 0.0:
        noise = np.random.normal(0, noise_std, size=6)
        synthetic_emfs = np.array(synthetic_emfs) + noise

    return np.array(synthetic_emfs)
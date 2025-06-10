import numpy as np
from utils.core.em_functions import biot_savart_loop, compute_emf
from scipy.optimize import least_squares


def generate_measured_emf(
    rx1_center,
    rx2_center,
    rx_axis,
    tx_coils,
    coil_radius,
    num_segments,
    current,
    area,
    num_turns,
    freq,
    noise_std=1e-6,
):
    """Simulate measured EMF values for a pair of Rx coils.

    Parameters
    ----------
    rx1_center, rx2_center : ndarray
        Center positions of the two receive coils.
    rx_axis : ndarray
        Shared axis direction of the coils (assumed unit length).
    tx_coils : list of dict
        Transmit coil definitions with ``center`` and ``normal`` fields.
    coil_radius : float
        Radius of the transmit coils used by ``biot_savart_loop``.
    num_segments : int
        Number of segments for the Biot–Savart integration.
    current : float
        Current in each Tx coil.
    area : float
        Cross‑sectional area of the Rx coils.
    num_turns : int
        Number of turns in each Rx coil.
    freq : float
        Excitation frequency in Hz.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    numpy.ndarray
        Flattened array of length ``2 * len(tx_coils)`` containing the
        simulated noisy EMF measurements for both Rx coils.
    """

    emf_list = []
    for rx_center in [rx1_center, rx2_center]:
        for tx in tx_coils:
            B = biot_savart_loop(
                rx_center,
                tx["center"],
                coil_radius,
                tx["normal"],
                current,
                num_segments,
            )
            emf = compute_emf(B, rx_axis, area, num_turns, freq)
            emf += np.random.normal(0.0, noise_std)
            emf_list.append(emf)

    return np.array(emf_list)

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


def estimate_rx_pose(
    measured_emf,
    tx_coils,
    coil_radius,
    area,
    num_turns,
    freq,
    initial_guess=None,
    d=0.06,
):
    """Estimate the Rx pose from EMF measurements using non-linear least squares."""

    if initial_guess is None:
        initial_guess = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2])

    result = least_squares(
        localization_cost,
        initial_guess,
        args=(measured_emf, tx_coils, coil_radius, area, num_turns, freq, d),
    )
    return result.x


def compute_rmse(true_center, true_axis, est_params):
    """Return positional RMSE in millimetres and angular error in degrees."""

    est_center = est_params[:3]
    theta, phi = est_params[3:5]
    est_axis = np.array(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    est_axis = est_axis / np.linalg.norm(est_axis)

    pos_error = np.linalg.norm(est_center - true_center)
    pos_rmse = pos_error * 1000.0

    dot_prod = np.clip(np.dot(est_axis, true_axis), -1.0, 1.0)
    ang_error = np.degrees(np.arccos(dot_prod))

    return pos_rmse, ang_error

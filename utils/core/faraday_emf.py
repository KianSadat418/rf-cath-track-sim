import numpy as np

# Compute the electromotive force (emf) induced in a coil by a time-varying magnetic field using Faraday's law of electromagnetic induction
def compute_emf(B_field, rx_axis, area, num_turns, frequency):
    B_parallel = np.dot(B_field, rx_axis)
    emf_peak = 2 * np.pi * frequency * num_turns * area * B_parallel
    return emf_peak
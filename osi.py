"""Osi module."""
import numpy as np

# Dataset: Orientations in degrees and corresponding firing rates (spikes/sec)
orientations = np.array([0, 30, 60, 90, 120, 150])
firing_rates = np.array([10, 25, 80, 30, 15, 8])

def calculate_osi(angles_deg, rates):
    """
    Calculate the Orientation Selectivity Index (OSI).
    """
    r_pref_idx, r_pref = max(enumerate(rates), key=lambda x: x[1])
    r_pref_angle = angles_deg[r_pref_idx]

    r_orth_angle = (r_pref_angle + 90) % 180
    r_orth_idx = np.where((angles_deg == r_orth_angle) | (angles_deg == (r_orth_angle + 180) % 180))[0][0]
    r_orth = rates[r_orth_idx]

    return round(float((r_pref - r_orth) / (r_pref + r_orth)), 6)

def calculate_gosi(angles_deg, rates):
    """
    Calculate the global Orientation Selectivity Index (gOSI).
    """
    numerator = np.abs(np.sum(rates * np.exp(2 * np.deg2rad(angles_deg) * 1j)))
    denominator = np.sum(rates)

    if denominator != 0:
        return round(float(numerator / denominator), 6)
    raise Exception("Invalid data for gOSI")

if __name__ == "__main__":
    osi_val = calculate_osi(orientations, firing_rates)
    gosi_val = calculate_gosi(orientations, firing_rates)
    
    print(f"Calculated OSI:  {osi_val}")
    print(f"Calculated gOSI: {gosi_val}")
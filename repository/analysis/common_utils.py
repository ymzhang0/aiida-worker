import numpy as np

def kbar_to_gpa(value):
    """Convert pressure/modulus from kbar to GPa."""
    return value / 10.0

def calculate_pettifor_ratio(c12, c44, modulus):
    """Calculate Pettifor ratio (C12 - C44) / modulus."""
    if modulus == 0:
        return 0.0
    return (c12 - c44) / modulus

def calculate_pugh_ratio(bulk_modulus, shear_modulus):
    """Calculate Pugh ratio B/G."""
    if shear_modulus == 0:
        return 0.0
    return bulk_modulus / shear_modulus

def calculate_cubic_elastic_averages(c11, c12, c44):
    """
    Calculate Voigt, Reuss, and Hill averages for a cubic system.
    Returns a dictionary with bulk_modulus, shear_modulus, young_modulus, and poisson_ratio.
    """
    # Bulk modulus is the same for Voigt and Reuss in cubic systems
    B = (c11 + 2 * c12) / 3.0
    
    # Shear modulus
    G_V = (c11 - c12 + 3 * c44) / 5.0
    G_R = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    G_H = (G_V + G_R) / 2.0
    
    averages = {}
    for label, G in [("voigt", G_V), ("reuss", G_R), ("hill", G_H)]:
        E = 9.0 * B * G / (3.0 * B + G) if (3.0 * B + G) != 0 else 0.0
        nu = (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G)) if (3.0 * B + G) != 0 else 0.0
        averages[label] = {
            "bulk_modulus": B,
            "shear_modulus": G,
            "young_modulus": E,
            "poisson_ratio": nu,
            "pugh_ratio": B / G if G != 0 else 0.0
        }
    return averages

def standardize_modulus_names(modulus_dict):
    """
    Rename keys from thermo_pw style to a standard internal style.
    Handles mapping like 'bulk_modulus_B' -> 'bulk_modulus'.
    """
    mapping = {
        'bulk_modulus_B': 'bulk_modulus',
        'shear_modulus_G': 'shear_modulus',
        'young_modulus_E': 'young_modulus',
        'poisson_ratio_n': 'poisson_ratio',
        'pugh_ratio_r': 'pugh_ratio',
    }
    new_dict = {}
    for k, v in modulus_dict.items():
        new_key = mapping.get(k, k)
        new_dict[new_key] = v
    return new_dict

import numpy as np
from typing import Dict, List

def calculate_utilities(
    modes_data: List[Dict[str, float]], 
    coeffs: Dict[str, float]
) -> np.ndarray:
    """
    Calculate the utility for each transport mode.
    V_m = ASC_m + beta_time * Time_m + beta_cost * Cost_m
    
    Args:
        modes_data (List[Dict]): List of dicts containing 'asc', 'time', 'cost' for each mode.
        coeffs (Dict): Dictionary with 'beta_time' and 'beta_cost'.
        
    Returns:
        np.ndarray: Array of utility values for each mode.
    """
    utilities = []
    b_time = coeffs.get("beta_time", -0.02)
    b_cost = coeffs.get("beta_cost", -0.05)
    
    for mode in modes_data:
        v = mode.get("asc", 0.0) + (b_time * mode.get("time", 0.0)) + (b_cost * mode.get("cost", 0.0))
        utilities.append(v)
        
    return np.array(utilities)

def multinomial_logit(utilities: np.ndarray) -> np.ndarray:
    """
    Calculate choice probabilities using the Multinomial Logit Model.
    P_m = exp(V_m) / Sum(exp(V_n))
    
    Args:
        utilities (np.ndarray): Array of utilities for each mode.
        
    Returns:
        np.ndarray: Array of probabilities for each mode.
    """
    exp_v = np.exp(utilities)
    probabilities = exp_v / np.sum(exp_v)
    return probabilities

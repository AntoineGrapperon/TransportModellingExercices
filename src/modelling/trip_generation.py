import pandas as pd
import numpy as np
from typing import Dict, List, Any

def calculate_regression_trips(
    population: float, 
    employment: float, 
    coeffs: Dict[str, float]
) -> float:
    """
    Calculate trips based on a simple linear regression model.
    Formula: Trips = C + (P_coeff * Population) + (E_coeff * Employment)
    
    Args:
        population (float): Zonal population.
        employment (float): Zonal employment.
        coeffs (Dict[str, float]): Model coefficients (intercept, pop_coeff, emp_coeff).
        
    Returns:
        float: Estimated number of trips.
    """
    intercept = coeffs.get("intercept", 0)
    p_coeff = coeffs.get("pop_coeff", 0)
    e_coeff = coeffs.get("emp_coeff", 0)
    
    return intercept + (p_coeff * population) + (e_coeff * employment)

def cross_classification_trips(
    hh_data: Dict[str, int], 
    trip_rates: pd.DataFrame
) -> float:
    """
    Calculate trips using category analysis (cross-classification).
    
    Args:
        hh_data (Dict[str, int]): Number of households in each category (e.g., size vs. income).
        trip_rates (pd.DataFrame): Average trips per household for each category.
        
    Returns:
        float: Total estimated trips.
    """
    # Simple implementation: matching household counts to trip rate table
    total_trips = 0
    for category, count in hh_data.items():
        if category in trip_rates.index:
            total_trips += count * trip_rates.loc[category, "rate"]
    return float(total_trips)

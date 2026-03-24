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
    hh_counts: Dict[str, int], 
    trip_rates: Dict[str, float]
) -> float:
    """
    Calculate trips using category analysis (cross-classification).
    
    Args:
        hh_counts (Dict[str, int]): Number of households in each category.
        trip_rates (Dict[str, float]): Average trips per household for each category.
        
    Returns:
        float: Total estimated trips.
    """
    total_trips = 0.0
    for category, count in hh_counts.items():
        rate = trip_rates.get(category, 0.0)
        total_trips += count * rate
    return total_trips

def get_sample_trip_rates() -> pd.DataFrame:
    """Returns a sample trip rate table for educational purposes."""
    data = {
        "0 Cars": [0.5, 1.2, 2.5],
        "1 Car": [1.5, 2.8, 4.2],
        "2+ Cars": [2.2, 4.5, 6.8]
    }
    return pd.DataFrame(data, index=["1 Person", "2-3 Persons", "4+ Persons"])

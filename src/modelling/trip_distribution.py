import numpy as np
import pandas as pd
from typing import Tuple

def gravity_model(
    productions: np.ndarray, 
    attractions: np.ndarray, 
    cost_matrix: np.ndarray, 
    beta: float
) -> np.ndarray:
    """
    Apply a singly-constrained gravity model.
    T_ij = P_i * (A_j * F_ij) / Sum_j(A_j * F_ij)
    where F_ij = exp(-beta * C_ij)
    
    Args:
        productions (np.ndarray): Array of trips produced per zone.
        attractions (np.ndarray): Array of trips attracted per zone.
        cost_matrix (np.ndarray): Matrix of travel costs between zones.
        beta (float): Sensitivity parameter for the friction factor.
        
    Returns:
        np.ndarray: Distributed trip matrix (O-D matrix).
    """
    # Calculate Friction Factors: F_ij = exp(-beta * cost_ij)
    friction_factors = np.exp(-beta * cost_matrix)
    
    # Calculate denominator: Sum_j (A_j * F_ij)
    # attractions is (n,), friction_factors is (n, n)
    # We want to multiply each row of friction_factors by attractions, then sum rows.
    attraction_friction = attractions * friction_factors
    denominators = attraction_friction.sum(axis=1)
    
    # Calculate T_ij
    # Avoid division by zero
    denominators[denominators == 0] = 1
    
    # T_ij = P_i * (A_j * F_ij / Denom_i)
    od_matrix = (productions[:, np.newaxis] * attraction_friction) / denominators[:, np.newaxis]
    
    return od_matrix

def furness_balancing(
    initial_matrix: np.ndarray, 
    target_productions: np.ndarray, 
    target_attractions: np.ndarray, 
    max_iterations: int = 10, 
    tolerance: float = 0.01
) -> Tuple[np.ndarray, int]:
    """
    Balance an O-D matrix using the Furness (Doubly-Constrained) method.
    
    Args:
        initial_matrix (np.ndarray): The initial trip matrix to balance.
        target_productions (np.ndarray): Desired row sums.
        target_attractions (np.ndarray): Desired column sums.
        max_iterations (int): Maximum balancing cycles.
        tolerance (float): Stopping criteria for convergence.
        
    Returns:
        Tuple[np.ndarray, int]: Balanced matrix and number of iterations performed.
    """
    matrix = initial_matrix.copy().astype(float)
    
    for i in range(max_iterations):
        # Row balancing
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        row_factors = target_productions / row_sums
        matrix *= row_factors[:, np.newaxis]
        
        # Column balancing
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1
        col_factors = target_attractions / col_sums
        matrix *= col_factors
        
        # Check convergence (Mean Absolute Error)
        current_p = matrix.sum(axis=1)
        error = np.mean(np.abs(current_p - target_productions))
        if error < tolerance:
            return matrix, i + 1
            
    return matrix, max_iterations

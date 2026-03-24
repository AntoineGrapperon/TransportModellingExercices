import numpy as np
from typing import Tuple

def bpr_function(
    free_flow_time: float, 
    capacity: float, 
    flow: float, 
    alpha: float = 0.15, 
    beta: float = 4.0
) -> float:
    """
    Calculate link travel time using the BPR (Bureau of Public Roads) function.
    T = T0 * (1 + alpha * (V/C)^beta)
    
    Args:
        free_flow_time (float): Travel time at zero flow (T0).
        capacity (float): Practical capacity of the link (C).
        flow (float): Current volume of traffic (V).
        alpha (float): Scaling parameter (default 0.15).
        beta (float): Power parameter (default 4.0).
        
    Returns:
        float: Calculated travel time.
    """
    if capacity <= 0:
        return float('inf')
    return free_flow_time * (1 + alpha * (flow / capacity)**beta)

def solve_2path_equilibrium(
    total_demand: float,
    link1_params: dict,
    link2_params: dict,
    tolerance: float = 0.01,
    max_iter: int = 100
) -> Tuple[float, float, float]:
    """
    Solve for User Equilibrium on a simple 2-path parallel network.
    Wardrop's 1st Principle: Travel times on all used paths are equal and minimal.
    
    Args:
        total_demand (float): Total flow to be assigned.
        link1_params, link2_params: Dicts with 't0', 'cap', 'alpha', 'beta'.
        
    Returns:
        Tuple: (flow1, flow2, travel_time)
    """
    # Simple binary search for flow1 (since t1 increases and t2 decreases with flow1)
    low = 0.0
    high = total_demand
    
    for _ in range(max_iter):
        flow1 = (low + high) / 2
        flow2 = total_demand - flow1
        
        t1 = bpr_function(link1_params['t0'], link1_params['cap'], flow1, link1_params['alpha'], link1_params['beta'])
        t2 = bpr_function(link2_params['t0'], link2_params['cap'], flow2, link2_params['alpha'], link2_params['beta'])
        
        if abs(t1 - t2) < tolerance:
            return flow1, flow2, (t1 + t2) / 2
        
        if t1 < t2:
            low = flow1
        else:
            high = flow1
            
    return flow1, total_demand - flow1, (t1 + t2) / 2

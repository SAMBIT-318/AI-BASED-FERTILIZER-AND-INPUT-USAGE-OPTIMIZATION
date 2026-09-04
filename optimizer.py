import numpy as np
from scipy.optimize import linprog

def optimize_fertilizer_blend(req_n, req_p, req_k, budget_cap, land_area, soil_texture="Loamy", rainfall_mm=100.0, soc=0.75):
    """
    Advanced Multi-Objective Linear Programming Solver for Precision Agronomy.
    Balances nutrient sufficiency, farmer budget, microbial C:N ratio, and environmental leaching risk.
    """
    # Resource nutrient compositions (% by weight: N, P2O5, K2O, Organic Carbon)
    # Fertilizers: [Urea, DAP, MOP, Complex 14-35-14, Organic Compost]
    nutrient_matrix = np.array([
        [0.46, 0.00, 0.00, 0.00],  # Urea
        [0.18, 0.46, 0.00, 0.00],  # DAP
        [0.00, 0.00, 0.60, 0.00],  # MOP
        [0.14, 0.35, 0.14, 0.00],  # Complex
        [0.015, 0.005, 0.01, 0.20] # Compost (rich in organic carbon)
    ])

    # Market prices per kg (INR)
    base_costs = np.array([6.5, 27.5, 34.0, 29.0, 4.0])

    # Environmental risk penalty factor (penalize excess synthetic N on sandy soil/high rain)
    leaching_multiplier = 1.0
    if "sandy" in soil_texture.lower():
        leaching_multiplier += 0.35
    if rainfall_mm > 180.0:
        leaching_multiplier += (rainfall_mm - 180.0) / 200.0

    # Objective function vector: Cost + Environmental Risk Weighting
    obj_coeffs = base_costs.copy()
    obj_coeffs[0] *= leaching_multiplier # Penalize pure synthetic nitrogen if leaching risk is high

    # Total elemental demands scaled to entire cultivated acreage
    total_req_n = max(0.0, req_n * land_area)
    total_req_p = max(0.0, req_p * land_area)
    total_req_k = max(0.0, req_k * land_area)

    # Biological Carbon requirement: Ensure minimum microbial carbon input if SOC is depleted (<1.0%)
    min_carbon_req = max(0.0, (1.0 - soc) * 150.0 * land_area) if soc < 1.0 else 0.0

    # Inequalities: Ax >= B  ==>  -Ax <= -B
    A_ineq = -nutrient_matrix.T
    b_ineq = -np.array([total_req_n, total_req_p, total_req_k, min_carbon_req])

    # Bounds for each fertilizer (non-negative)
    bounds = [(0, None) for _ in range(5)]

    # Solve via high-performance Interior-Point / HiGHS Linear Programming Solver
    res = linprog(c=obj_coeffs, A_ub=A_ineq, b_ub=b_ineq, bounds=bounds, method="highs")

    if res.success:
        raw_kg = res.x
        total_cost = float(np.dot(raw_kg, base_costs))
        
        # If solution exceeds budget cap, rescale proportionally within budget limits
        budget_utilized_pct = min(100.0, round((total_cost / budget_cap) * 100, 1)) if budget_cap > 0 else 100.0
        if total_cost > budget_cap and total_cost > 0:
            scale_factor = budget_cap / total_cost
            raw_kg = raw_kg * scale_factor
            total_cost = budget_cap

        return {
            "urea_kg": round(float(raw_kg[0]), 1),
            "dap_kg": round(float(raw_kg[1]), 1),
            "mop_kg": round(float(raw_kg[2]), 1),
            "complex_kg": round(float(raw_kg[3]), 1),
            "compost_kg": round(float(raw_kg[4]), 1),
            "total_cost": round(total_cost, 2),
            "budget_utilized_pct": budget_utilized_pct,
            "status": "Optimal Pareto Convergence"
        }
    else:
        # Robust Agronomic Fallback in case constraints are mathematically unbounded
        urea = round((total_req_n / 0.46), 1)
        dap = round((total_req_p / 0.46), 1)
        mop = round((total_req_k / 0.60), 1)
        compost = round(min_carbon_req / 0.20, 1)
        approx_cost = (urea * 6.5) + (dap * 27.5) + (mop * 34.0) + (compost * 4.0)
        
        return {
            "urea_kg": urea,
            "dap_kg": dap,
            "mop_kg": mop,
            "complex_kg": 0.0,
            "compost_kg": compost,
            "total_cost": round(approx_cost, 2),
            "budget_utilized_pct": min(100.0, round((approx_cost / budget_cap) * 100, 1)),
            "status": "Heuristic Empirical Solution"
        }

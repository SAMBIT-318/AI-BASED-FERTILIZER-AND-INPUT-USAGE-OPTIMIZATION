import pulp

def optimize_fertilizer_blend(
    req_n: float,
    req_p: float,
    req_k: float,
    budget_cap: float,
    land_area: float
) -> dict:
    total_n = max(0.0, req_n * land_area)
    total_p = max(0.0, req_p * land_area)
    total_k = max(0.0, req_k * land_area)

    # Market prices per kg (INR / USD equivalent)
    # Urea: 46% N | Cost: 6.0/kg
    # DAP: 18% N, 46% P | Cost: 27.0/kg
    # MOP: 60% K | Cost: 34.0/kg
    # Complex (14-35-14): 14% N, 35% P, 14% K | Cost: 29.0/kg
    # Compost: 1.5% N, 1.0% P, 1.5% K | Cost: 3.5/kg
    costs = {
        'Urea': 6.0,
        'DAP': 27.0,
        'MOP': 34.0,
        'Complex': 29.0,
        'Compost': 3.5
    }

    prob = pulp.LpProblem("Fertilizer_Blend_Optimization", pulp.LpMinimize)

    q_urea = pulp.LpVariable("Urea_kg", lowBound=0, cat='Continuous')
    q_dap = pulp.LpVariable("DAP_kg", lowBound=0, cat='Continuous')
    q_mop = pulp.LpVariable("MOP_kg", lowBound=0, cat='Continuous')
    q_comp = pulp.LpVariable("Complex_kg", lowBound=0, cat='Continuous')
    q_org = pulp.LpVariable("Compost_kg", lowBound=0, cat='Continuous')

    slack_n = pulp.LpVariable("Deficit_Slack_N", lowBound=0, cat='Continuous')
    slack_p = pulp.LpVariable("Deficit_Slack_P", lowBound=0, cat='Continuous')
    slack_k = pulp.LpVariable("Deficit_Slack_K", lowBound=0, cat='Continuous')

    # Objective: Minimize cost + penalty for unmet nutrient demand
    prob += (
        costs['Urea'] * q_urea +
        costs['DAP'] * q_dap +
        costs['MOP'] * q_mop +
        costs['Complex'] * q_comp +
        costs['Compost'] * q_org +
        200.0 * (slack_n + slack_p + slack_k)
    ), "Total_Expenditure_Objective"

    # Nutrient Balance Constraints
    prob += (0.46 * q_urea + 0.18 * q_dap + 0.14 * q_comp + 0.015 * q_org + slack_n >= total_n), "Nitrogen_Constraint"
    prob += (0.46 * q_dap + 0.35 * q_comp + 0.010 * q_org + slack_p >= total_p), "Phosphorus_Constraint"
    prob += (0.60 * q_mop + 0.14 * q_comp + 0.015 * q_org + slack_k >= total_k), "Potassium_Constraint"

    # Wallet & Budget Constraint
    prob += (
        costs['Urea'] * q_urea +
        costs['DAP'] * q_dap +
        costs['MOP'] * q_mop +
        costs['Complex'] * q_comp +
        costs['Compost'] * q_org <= budget_cap
    ), "Budget_Cap_Constraint"

    # Leaching and Over-application Upper Bounds (limit to 120% of demand + tolerance)
    prob += (0.46 * q_urea + 0.18 * q_dap + 0.14 * q_comp + 0.015 * q_org <= total_n * 1.20 + 3.0), "N_Leach_Guard"
    prob += (0.46 * q_dap + 0.35 * q_comp + 0.010 * q_org <= total_p * 1.20 + 3.0), "P_Leach_Guard"
    prob += (0.60 * q_mop + 0.14 * q_comp + 0.015 * q_org <= total_k * 1.20 + 3.0), "K_Leach_Guard"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    res_urea = round(max(0.0, pulp.value(q_urea)), 2)
    res_dap = round(max(0.0, pulp.value(q_dap)), 2)
    res_mop = round(max(0.0, pulp.value(q_mop)), 2)
    res_comp = round(max(0.0, pulp.value(q_comp)), 2)
    res_org = round(max(0.0, pulp.value(q_org)), 2)

    total_cost = round(
        costs['Urea'] * res_urea +
        costs['DAP'] * res_dap +
        costs['MOP'] * res_mop +
        costs['Complex'] * res_comp +
        costs['Compost'] * res_org, 2
    )

    supplied_n = round(0.46 * res_urea + 0.18 * res_dap + 0.14 * res_comp + 0.015 * res_org, 2)
    supplied_p = round(0.46 * res_dap + 0.35 * res_comp + 0.010 * res_org, 2)
    supplied_k = round(0.60 * res_mop + 0.14 * res_comp + 0.015 * res_org, 2)

    return {
        "status": pulp.LpStatus[status],
        "urea_kg": res_urea,
        "dap_kg": res_dap,
        "mop_kg": res_mop,
        "complex_kg": res_comp,
        "compost_kg": res_org,
        "total_cost": total_cost,
        "budget_limit": budget_cap,
        "budget_utilized_pct": round((total_cost / budget_cap) * 100, 1) if budget_cap > 0 else 0.0,
        "supplied_n": supplied_n,
        "supplied_p": supplied_p,
        "supplied_k": supplied_k,
        "target_n": round(total_n, 2),
        "target_p": round(total_p, 2),
        "target_k": round(total_k, 2),
        "unmet_n": round(max(0.0, pulp.value(slack_n)), 2),
        "unmet_p": round(max(0.0, pulp.value(slack_p)), 2),
        "unmet_k": round(max(0.0, pulp.value(slack_k)), 2),
    }

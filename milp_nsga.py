# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring, wrong-import-order

import pulp
import pickle
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from optimizer import ChargingProblem

with open("sim_data.pkl", "rb") as f:
    DATA = pickle.load(f)


def run_milp():
    print("Starting MILP Solver...")
    prob = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)
    N, T, F = DATA["N"], DATA["T"], DATA["F"]

    # Variables [cite: 539]
    z_dep = pulp.LpVariable.dicts("z_dep", (range(N), range(1)), cat="Binary")
    p_dep = pulp.LpVariable.dicts("p_dep", (range(N), range(T)), lowBound=0)

    # Objective: Minimize Depot Cost only for simplicity [cite: 542]
    prob += pulp.lpSum(
        [
            p_dep[i][t] * DATA["dt"] * DATA["pi_grid"][t]
            for i in range(N)
            for t in range(T)
        ]
    )

    # Constraints [cite: 552-561]
    for t in range(T):
        # Site Limit
        prob += (
            pulp.lpSum([p_dep[i][t] for i in range(N)])
            + DATA["p_base"][t]
            - DATA["p_pv"][t]
            <= DATA["p_site_max"]
        )

    for i in range(N):
        # SoC Continuity: Must meet energy demand (No shortfall allowed for MILP comparison)
        prob += (
            pulp.lpSum([p_dep[i][t] for i in range(T)]) * DATA["dt"] * 0.9
            >= DATA["fleet"][i]["e_trip"][0]
        )
        for t in range(T):
            # Only charge during dwell
            if t > 60:
                prob += p_dep[i][t] == 0
            prob += p_dep[i][t] <= 22.0 * z_dep[i][0]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Capture MILP Load Profile
    milp_load = np.zeros(T)
    if pulp.LpStatus[prob.status] == "Optimal":
        for i in range(N):
            for t in range(T):
                milp_load[t] += pulp.value(p_dep[i][t])

    return pulp.value(prob.objective), milp_load


# Run NSGA-II
problem = ChargingProblem()
nsga_res = minimize(problem, NSGA2(pop_size=40), ("n_gen", 50), seed=1)

# Run MILP
milp_cost, milp_load = run_milp()

with open("comparison_results.pkl", "wb") as f:
    pickle.dump({"nsga": nsga_res.F, "milp_cost": milp_cost, "milp_load": milp_load}, f)

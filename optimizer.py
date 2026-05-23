# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring, wrong-import-order

import numpy as np
import pickle
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

with open("sim_data.pkl", "rb") as f:
    DATA = pickle.load(f)


class ChargingProblem(ElementwiseProblem):
    def __init__(self):
        # Decisions per window: 0: Skip, 1: Depot, 2..F+1: Public L2, F+2..2F+1: Public DC
        self.n_windows = sum(len(v["dwells"]) for v in DATA["fleet"])
        super().__init__(
            n_var=self.n_windows, n_obj=2, n_constr=1, xl=0, xu=1 + 2 * DATA["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x).astype(int)
        N, T = DATA["N"], DATA["T"]

        # State tracking
        soc = np.array([v["soc_init"] * v["bat_max"] for v in DATA["fleet"]])
        total_cost = 0.0
        total_shortfall = 0.0
        depot_load = np.zeros(T)

        idx = 0
        for i, v in enumerate(DATA["fleet"]):
            for j, dwell in enumerate(v["dwells"]):
                choice = x[idx]
                idx += 1

                start, end = dwell
                duration_hrs = (end - start) * DATA["dt"]

                if choice == 1:  # Depot [cite: 1495, 1586]
                    p_charge = min(
                        22.0, (v["bat_max"] - soc[i]) / DATA["dt"]
                    )  # Simplified power
                    soc[i] += p_charge * DATA["dt"] * 0.9
                    total_cost += np.sum(
                        DATA["pi_grid"][start:end] * p_charge * DATA["dt"]
                    )
                    depot_load[start:end] += p_charge

                elif choice > 1:  # Public Station
                    f_idx = (choice - 2) % DATA["F"]
                    c_type = "L2" if choice < 2 + DATA["F"] else "DC"
                    station = DATA["stations"][f_idx]

                    wait = station["wait_profile"][start]
                    net_time = (duration_hrs * 60) - v["detour"][f_idx] - wait

                    if net_time > 0:
                        energy = min(
                            station["power"][c_type] * (net_time / 60),
                            v["bat_max"] - soc[i],
                        )
                        soc[i] += energy * 0.95
                        total_cost += energy * station["pi_pub"][c_type]
                        total_cost += DATA["c_d_t"] * (
                            v["detour"][f_idx] + wait
                        )  # Detour/Wait Cost

            # Trip consumption
            for trip_e in v["e_trip"]:
                if soc[i] < trip_e:
                    total_shortfall += trip_e - soc[i]
                    soc[i] = 0
                else:
                    soc[i] -= trip_e

        # Constraint: Site limit
        site_violation = np.sum(
            np.maximum(
                0, depot_load + DATA["p_base"] - DATA["p_pv"] - DATA["p_site_max"]
            )
        )

        out["F"] = [total_cost, total_shortfall]
        out["G"] = [site_violation]


# Optimization
problem = ChargingProblem()
algorithm = NSGA2(pop_size=100)
res = minimize(problem, algorithm, ("n_gen", 100), seed=1, verbose=True)

with open("results.pkl", "wb") as f:
    pickle.dump(res, f)
